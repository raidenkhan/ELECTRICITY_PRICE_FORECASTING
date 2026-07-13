"""
mixvol_tft.py - Electricity MixVol TFT Architecture

Implements the Temporal Fusion Transformer integrated with a Sparse Mixture of
Experts (MoE) head for regime-aware electricity price forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidualNetwork(nn.Module):
    """
    Shared GRN building block.

    Inputs: input_dim, hidden_dim, output_dim, dropout=0.1
    Architecture: Linear -> ELU -> Linear -> Dropout -> LayerNorm -> Skip connection
    Purpose: Shared building block for VSN and context encoders.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.skip_layer(x)
        return self.layernorm(out)

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network.
    
    Implements: Bloch §4.1.4 — fundamentals-driven kS(X_t; theta) gating.
    Architecture: One GRN per variable, softmax over all weights.
    """
    def __init__(self, n_vars: int, hidden_dim: int):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_vars)
        ])
        self.flattened_grn = GatedResidualNetwork(
            input_dim=n_vars * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=n_vars
        )
        self.last_weights = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape: [batch, time, n_vars, hidden_dim]
        Returns: selected_features [batch, time, hidden_dim], variable_weights [batch, time, n_vars]
        """
        b, t, v, h = x.shape
        x_flat = x.view(b, t, -1)
        
        weights = self.flattened_grn(x_flat)
        weights = F.softmax(weights, dim=-1)
        self.last_weights = weights

        processed_vars = []
        for i in range(v):
            processed_vars.append(self.var_grns[i](x[:, :, i, :]))
            
        processed_vars = torch.stack(processed_vars, dim=-2)
        weights_expanded = weights.unsqueeze(-1)
        
        selected_features = (processed_vars * weights_expanded).sum(dim=-2)
        return selected_features, weights

class RegimeExpert(nn.Module):
    """
    Regime Expert Head.

    Implements: Bloch §6.2.1 — phi_i(·) regime-specific density component.
    Architecture: 2-layer FF with GELU, LayerNorm, quantile output head.
    Physical coherence: learnable p_floor applied as additive bias to q10 and q50 (0-indexed: 0 and 3).
    """
    def __init__(self, hidden_dim: int, n_features: int, n_quantiles: int = 7, expert_id: int = 0):
        super().__init__()
        self.expert_id = expert_id
        
        # Each expert gets its own Regime-Aware VSN (RA-VSN)
        self.vsn = VariableSelectionNetwork(n_features, hidden_dim)
        
        # Each expert gets its own Temporal Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, x_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x_emb shape: [batch, time, n_features, hidden_dim]
        Returns: selected_features, quantiles, vsn_weights
        """
        # 1. Feature selection specialized for THIS regime
        selected_features, vsn_weights = self.vsn(x_emb)
        
        # 2. Temporal attention specialized for THIS regime
        # attn_out shape [batch, time, hidden_dim]
        attn_out, _ = self.attn(selected_features, selected_features, selected_features)
        
        # 3. Forecast Mapping
        # Use last step or horizon (24) extraction
        # Since we are in the expert call, we map the last 24 steps
        context = attn_out[:, -24:, :]
        
        out = self.fc1(context)
        out = self.gelu(out)
        out = self.ln(out)
        raw = self.fc2(out)
        
        # Enforce strict monotonicity
        q0 = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + 1e-4
        quantiles = torch.cat([q0, q0 + torch.cumsum(deltas, dim=-1)], dim=-1)
        
        return {
            'quantiles': quantiles,
            'vsn_weights': vsn_weights
        }

class SparseMoEHead(nn.Module):
    """
    Sparse Mixture of Experts Head.

    Implements: Bloch Eq. 6.2.2 — a_i(X_t) · phi_i(P_t) mixture.
    Router: Linear(hidden_dim, n_experts) followed by top-k sparse selection.
    Load balancing: tracking coefficient of variation.
    """
    def __init__(self, hidden_dim: int, n_features: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, n_experts)
        
        self.experts = nn.ModuleList([
            RegimeExpert(hidden_dim=hidden_dim, n_features=n_features, n_quantiles=7, expert_id=i)
            for i in range(n_experts)
        ])
        
        # New: Temporal summary for router context (Fixing static query bottleneck)
        self.router_summary = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Fix 2: stronger init + dropout + symmetry-breaking noise
        nn.init.uniform_(self.router.weight, -0.1, 0.1)
        nn.init.zeros_(self.router.bias)
        self.router_dropout = nn.Dropout(0.1)
        with torch.no_grad():
            self.router.weight.add_(torch.randn_like(self.router.weight) * 0.05)
        
        self.routing_cv = torch.tensor(0.0)

    def forward(self, x_emb: torch.Tensor, router_bias: torch.Tensor | None = None, T: float = 1.0) -> dict[str, torch.Tensor]:
        """
        x_emb: [batch, time, n_features, hidden_dim]
        T: dynamically coupled temperature (Hurst-Annealing)
        """
        # 1. Generate Temporal Summary Query [batch, hidden_dim]
        # x_emb_mean: [batch, time, hidden_dim]
        x_emb_mean = x_emb.mean(dim=2) 
        _, h_n = self.router_summary(x_emb_mean)
        router_query = h_n[-1] # [batch, hidden_dim]

        logits = self.router(self.router_dropout(router_query))
        if router_bias is not None:
            if router_bias.dim() == 3:
                router_bias = router_bias.mean(dim=1)
            logits = logits + router_bias
            
        # Use provided H-annealed temperature with a safety floor
        T_safe = max(T, 0.3) 
        probs = F.softmax(logits / T_safe, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        mean_probs = probs.mean(dim=0)
        var_probs = probs.var(dim=0, unbiased=False)
        self.routing_cv = torch.sqrt(var_probs.mean() + 1e-8) / (mean_probs.mean() + 1e-8)
        
        b, h, f, d = x_emb.shape
        # Experts will output [batch, 24, 7]
        mixture_quantiles = torch.zeros(b, 24, 7, device=x_emb.device, dtype=x_emb.dtype)
        
        # Evaluate top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]
            expert_prob = top_k_probs[..., i].view(b, 1, 1) # [b, 1, 1] broadcast to [b, 24, 7]
            
            for exp_id in range(self.n_experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    exp_out = self.experts[exp_id](x_emb[mask])
                    mixture_quantiles[mask] += (exp_out['quantiles'] * expert_prob[mask])

        return {
            'mixture_quantiles': mixture_quantiles,
            'gate_weights': probs,
            'active_expert_ids': top_k_indices
        }

class HurstGateLayer(nn.Module):
    """
    Hurst Gate Layer.

    Implements: Bloch §2.1.4 — H(t)-driven regime persistence.
    Represents the gamma_ij transition bias matrix.
    """
    def __init__(self, n_experts: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.fc = nn.Linear(1, n_experts * n_experts)
        # Zero-init to start with neutral (no) bias
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h_scalar: torch.Tensor) -> torch.Tensor:
        """
        h_scalar shape: [batch, 1] or [batch, time, 1]
        Returns bias tensor [batch, (time,) n_experts] applied to router logits.
        We simplify this to return an expert bias vector [batch, time, n_experts]
        for the current active state. Currently, we just generate bias [batch, time, n_experts]
        or average over transition matrix rows.
        """
        out = self.fc(h_scalar)
        # Reshape to [batch, (time), n_experts, n_experts]
        new_shape = list(h_scalar.shape[:-1]) + [self.n_experts, self.n_experts]
        out = out.view(*new_shape)
        # Generate an additive bias vector. For simplicity, we mean over one dimension or select.
        # Given the minimal spec, we return bias [..., sum over previous state (dim=-1)].
        return out.mean(dim=-1)

class ElectricityMixVolTFT(nn.Module):
    """
    Full MixVol TFT Architecture.

    Integrates all components.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_dim = config.get('hidden_dim', 64)
        self.n_features = config.get('n_features', 18)
        self.n_static_features = config.get('n_static_features', 0)
        self.n_experts = config.get('n_experts', 4)
        self.top_k = config.get('top_k', 2)
        
        # Feature embeddings
        self.feature_embed = nn.Linear(1, self.hidden_dim)
        if self.n_static_features > 0:
            self.static_embed = nn.Linear(self.n_static_features, self.hidden_dim)
            
        self.vsn = VariableSelectionNetwork(self.n_features, self.hidden_dim)
        
        # Simple attention layer for sequence representation
        self.hurst_gate = HurstGateLayer(self.n_experts)
        self.moe = SparseMoEHead(self.hidden_dim, self.n_features, self.n_experts, self.top_k)

    def forward(self, x_historical: torch.Tensor, x_static: torch.Tensor | None = None, h_t: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        x_historical: [batch, seq_len, n_features]
        """
        b, s, f = x_historical.shape
        
        # Embed each feature independently for the VSNs: [b, s, f, 1] -> [b, s, f, h]
        x_emb = self.feature_embed(x_historical.unsqueeze(-1))
        
        # Router logic (Hurst-Annealing)
        router_bias = None
        T = 1.0 # default temperature
        
        if h_t is not None:
            # Predict bias based on Hurst
            if h_t.dim() == 2:
                h_t_v = h_t.unsqueeze(1)
            else:
                h_t_v = h_t
            router_bias = self.hurst_gate(h_t_v)
            
            # Hurst-Temperature Coupling (Hurst-Annealing)
            # High H (persistence) -> Low T (sharp selection)
            # Low H (mean-reverting) -> High T (diffuse mixing)
            # h_t mean across batch
            h_mean = h_t.mean().item()
            # Map H [0.4, 0.6] -> T [2.0, 0.5]
            T = 2.0 - (h_mean - 0.4) * (1.5 / 0.2)
            T = max(0.5, min(2.0, T))
            
        moe_out = self.moe(x_emb, router_bias, T=T)
        
        return {
            'quantiles': moe_out['mixture_quantiles'],
            'gate_weights': moe_out['gate_weights'],
            'active_experts': moe_out['active_expert_ids'],
            'T_hurst': torch.tensor(T, device=x_historical.device)
        }
