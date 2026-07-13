# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# encoder.py — BiLSTM Recognition Network (Amortised Variational Encoder)
#
# Role (Layer 0 in the full pipeline):
#   Encodes a 168-hour price/exogenous history window together with 24-hour
#   future exogenous forecasts (renewable generation, load forecast, etc.)
#   into two outputs consumed by downstream modules:
#
#     1. z0_mean, z0_logvar  →  initial condition distribution q(z_0 | x)
#                                passed to LatentSDE for reparameterised sampling
#
#     2. context             →  (batch, horizon, hidden_dim) time-varying
#                                context injected into the SDE drift f_θ(z, x, ctx)
#                                at each future timestep
#
# Architecture summary:
#   history_lstm   BiLSTM(18 → 128, 2 layers) — processes the full 168h window
#   future_lstm    UniLSTM(6  →  64, 1 layer)  — processes the 24h exog forecasts
#   z0_mean_net    Linear(256 → 3)             — posterior mean of z_0
#   z0_logvar_net  Linear(256 → 3)             — posterior log-variance of z_0
#   context_proj   Linear(256 + 64 → 128)      — per-timestep context projection
#   context_norm   LayerNorm(128)              — stabilise context magnitudes
#
# Dimension key (defaults):
#   history_feat_dim  = 18    (price lags + calendar + commodity + flow features)
#   future_feat_dim   = 6     (load_forecast, solar_forecast, wind_onshore_forecast,
#                              wind_offshore_forecast, day_ahead_load, calendar_hot)
#   latent_dim        = 3     (renewable surplus / thermal scarcity / demand surge)
#   hidden_dim        = 128   (matches SDE context dimension — no bottleneck)
#
# References:
#   Krishnan et al. (2015) "Deep Kalman Filters"
#   Rubanova et al. (2019) "Latent ODEs for Irregularly-Sampled Time Series"
#   LF_GP_NRF_Research_Plan.md §4.1–4.3
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LatentForceEncoder(nn.Module):
    """BiLSTM recognition network for the LF-GP-NRF model.

    Encodes a history window together with future exogenous forecasts into:
      - A variational posterior over the SDE initial condition z_0 ~ q(z_0|x)
        parameterised by (z0_mean, z0_logvar).
      - A time-varying context vector for each forecast horizon step, injected
        into the Neural SDE drift at inference time.

    Parameters
    ----------
    history_feat_dim : int
        Number of features in the history tensor (default 18).  Includes lagged
        prices, calendar dummies, commodity prices, cross-border flow features,
        and any derived technical indicators.
    future_feat_dim : int
        Number of exogenous features available for the forecast horizon
        (default 6).  Typically deterministic forecasts: load, solar, wind
        onshore, wind offshore, day-ahead load, and a calendar indicator.
    latent_dim : int
        Dimensionality of the latent force vector z(t).  Default 3, reflecting
        three physically interpretable forces: renewable surplus, thermal
        scarcity, and demand surge.  Ablate over {2, 3, 4, 5}.
    hidden_dim : int
        Hidden size of the BiLSTM.  Also the output context dimension fed to
        the SDE.  Default 128 matches §4.3 of the research plan.
    num_layers : int
        Number of stacked BiLSTM layers for the history encoder.  Default 2.
        Dropout is applied between layers when num_layers > 1.
    dropout : float
        Inter-layer dropout probability for the history BiLSTM.  Default 0.1.
        Ignored (set to 0) when num_layers == 1 to satisfy PyTorch's constraint.
    """

    def __init__(
        self,
        history_feat_dim: int = 18,
        future_feat_dim: int = 6,
        latent_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.history_feat_dim = history_feat_dim
        self.future_feat_dim = future_feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ------------------------------------------------------------------
        # 1. History BiLSTM
        #    Input  : (batch, history_len, history_feat_dim)
        #    Output : (batch, history_len, hidden_dim * 2)   [bidirectional]
        #    We use the LAST TIMESTEP's concatenated [fwd | bwd] hidden as the
        #    global history summary → shape (batch, hidden_dim * 2 = 256).
        #
        #    PyTorch LSTM note: dropout is only valid between layers, so it
        #    must be 0 when num_layers == 1.
        # ------------------------------------------------------------------
        self.history_lstm = nn.LSTM(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ------------------------------------------------------------------
        # 2. Future Exogenous UniLSTM
        #    Input  : (batch, horizon, future_feat_dim)
        #    Output : (batch, horizon, hidden_dim // 2 = 64)
        #    All hidden states are retained (not just the last) because the
        #    context projection needs a per-timestep future representation.
        # ------------------------------------------------------------------
        self.future_lstm = nn.LSTM(
            input_size=future_feat_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # ------------------------------------------------------------------
        # 3. z0 Posterior Head
        #    Maps the global history summary (hidden_dim * 2 = 256) to the
        #    mean and log-variance of the initial latent force z_0.
        #
        #    z0_logvar is left unbounded (no Softplus/clamp here); clipping is
        #    applied in reparameterise() for numerical safety.
        # ------------------------------------------------------------------
        self.z0_mean_net = nn.Linear(hidden_dim * 2, latent_dim)
        self.z0_logvar_net = nn.Linear(hidden_dim * 2, latent_dim)

        # ------------------------------------------------------------------
        # 4. Context Projection
        #    Per future timestep, we concatenate:
        #      - The global history summary broadcast to (batch, horizon, 256)
        #      - The per-step future LSTM output              (batch, horizon, 64)
        #    Total input dim: 256 + 64 = 320   →   hidden_dim = 128
        #
        #    LayerNorm is applied after projection to keep context magnitudes
        #    in a well-conditioned range for the SDE drift network.
        # ------------------------------------------------------------------
        context_in_dim = hidden_dim * 2 + hidden_dim // 2  # 256 + 64 = 320
        self.context_proj = nn.Linear(context_in_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

        # Weight initialisation — orthogonal for recurrent weights, Xavier for
        # linear projections.  This follows common practice for deep LSTMs to
        # mitigate vanishing/exploding gradients at the start of training.
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise LSTM and linear layer weights."""
        for name, param in self.history_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 to encourage long-range memory
                # (bias vector layout: [input, forget, cell, output] gates)
                n = param.shape[0] // 4
                param.data[n : 2 * n].fill_(1.0)

        for name, param in self.future_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.shape[0] // 4
                param.data[n : 2 * n].fill_(1.0)

        for linear in (
            self.z0_mean_net,
            self.z0_logvar_net,
            self.context_proj,
        ):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, history: Tensor, future_exog: Tensor) -> dict[str, Tensor]:
        """Encode history and future exogenous features.

        Parameters
        ----------
        history : Tensor, shape (batch, history_len, history_feat_dim)
            The look-back window.  For the default configuration this is a
            168-step (one week of hourly data) tensor containing lagged prices
            and exogenous covariates observed in the past.
        future_exog : Tensor, shape (batch, horizon, future_feat_dim)
            Deterministic exogenous forecasts for the prediction horizon.  For
            the default configuration this is a 24-step (one day ahead) tensor
            containing renewable generation and load forecasts.

        Returns
        -------
        dict with keys:
            'z0_mean'   : Tensor (batch, latent_dim)
                          Posterior mean of the SDE initial condition z_0.
            'z0_logvar' : Tensor (batch, latent_dim)
                          Posterior log-variance of z_0 (log σ²).
            'context'   : Tensor (batch, horizon, hidden_dim)
                          Time-varying context vector for each forecast step.
                          Injected into the SDE drift f_θ(z_t, ctx_t) at each
                          integration step by the LatentSDE module.

        Shapes (with defaults):
            history    : (B, 168, 18)
            future_exog: (B,  24,  6)
            z0_mean    : (B,   3)
            z0_logvar  : (B,   3)
            context    : (B,  24, 128)
        """
        batch_size = history.size(0)
        horizon = future_exog.size(1)

        # ------------------------------------------------------------------
        # Step 1: Encode history with bidirectional LSTM
        #
        # history_out   : (B, history_len, hidden_dim * 2)
        #                  All timestep outputs — not used directly.
        # h_n           : (num_layers * 2, B, hidden_dim)
        #                  Final hidden states for all layers and directions.
        #
        # We extract the last layer's forward and backward final hidden states
        # and concatenate them to form the global history summary.
        #
        # PyTorch stores h_n as:
        #   h_n[0]  — layer 0, forward
        #   h_n[1]  — layer 0, backward
        #   h_n[2]  — layer 1, forward    (if num_layers == 2)
        #   h_n[3]  — layer 1, backward   (if num_layers == 2)
        #
        # We want the LAST layer, which is at indices:
        #   forward  : h_n[-2]  (= h_n[2 * num_layers - 2])
        #   backward : h_n[-1]  (= h_n[2 * num_layers - 1])
        # ------------------------------------------------------------------
        _, (h_n, _) = self.history_lstm(history)
        #  h_n : (num_layers * 2, B, hidden_dim)

        # Last layer's final forward and backward hidden states
        fwd_hidden = h_n[-2]  # (B, hidden_dim)
        bwd_hidden = h_n[-1]  # (B, hidden_dim)

        # Global history summary: concatenate forward + backward
        history_summary = torch.cat([fwd_hidden, bwd_hidden], dim=-1)
        # history_summary : (B, hidden_dim * 2 = 256)

        # ------------------------------------------------------------------
        # Step 2: Derive z0 posterior parameters from history summary
        # ------------------------------------------------------------------
        z0_mean = self.z0_mean_net(history_summary)  # (B, latent_dim)
        z0_logvar = self.z0_logvar_net(history_summary)  # (B, latent_dim)

        # ------------------------------------------------------------------
        # Step 3: Encode future exogenous features
        #
        # future_out : (B, horizon, hidden_dim // 2 = 64)
        #              Per-timestep hidden states — retain all of them.
        # ------------------------------------------------------------------
        future_out, _ = self.future_lstm(future_exog)
        # future_out : (B, horizon, hidden_dim // 2)

        # ------------------------------------------------------------------
        # Step 4: Build context vector for each future timestep
        #
        # Broadcast the scalar history summary across the forecast horizon so
        # we can concatenate it with the per-step future LSTM outputs.
        #
        # history_summary_exp : (B, horizon, hidden_dim * 2)
        # future_out          : (B, horizon, hidden_dim // 2)
        # concat              : (B, horizon, hidden_dim * 2 + hidden_dim // 2)
        #                       = (B, horizon, 320) with defaults
        # ------------------------------------------------------------------
        history_summary_exp = history_summary.unsqueeze(1).expand(
            batch_size, horizon, self.hidden_dim * 2
        )
        # Concatenate along feature dimension
        context_in = torch.cat([history_summary_exp, future_out], dim=-1)
        # context_in : (B, horizon, 320)

        # Project and normalise
        context = self.context_proj(context_in)  # (B, horizon, hidden_dim)
        context = self.context_norm(context)  # (B, horizon, hidden_dim)

        return {
            "z0_mean": z0_mean,  # (B, latent_dim)
            "z0_logvar": z0_logvar,  # (B, latent_dim)
            "context": context,  # (B, horizon, hidden_dim)
        }

    # ------------------------------------------------------------------
    # Reparameterisation trick
    # ------------------------------------------------------------------

    def reparameterise(self, z0_mean: Tensor, z0_logvar: Tensor) -> Tensor:
        """Draw a z_0 sample via the reparameterisation trick.

        z = μ + ε · exp(0.5 · log σ²)   where   ε ~ N(0, I)

        The log-variance is clamped to [-10, 10] before exponentiation to
        prevent numerical overflow/underflow during early training when the
        posterior has not yet contracted.

        Parameters
        ----------
        z0_mean : Tensor, shape (batch, latent_dim)
        z0_logvar : Tensor, shape (batch, latent_dim)

        Returns
        -------
        Tensor, shape (batch, latent_dim)
            A single sample from q(z_0 | x).  Gradients flow through μ and
            log σ² (not through ε), enabling backpropagation through the
            sampling operation.
        """
        # Clamp for numerical stability — avoids exp() overflow
        z0_logvar_clamped = z0_logvar.clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * z0_logvar_clamped)  # (B, latent_dim)
        eps = torch.randn_like(std)  # (B, latent_dim)
        return z0_mean + eps * std  # (B, latent_dim)


# =============================================================================
# __main__ — Shape verification test block
#
# Run with:
#   python -m src.models.lf_gp_nrf.encoder
# or directly:
#   python encoder.py
#
# All assertions verify the tensor shapes documented in the class docstrings.
# A summary table is printed on success.
# =============================================================================

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # Test configuration (mirrors the research plan defaults)
    # ------------------------------------------------------------------
    BATCH = 4
    HISTORY_LEN = 168  # one week of hourly observations
    HORIZON = 24  # day-ahead forecast horizon

    HISTORY_FEAT = 18  # price + calendar + commodity + flow features
    FUTURE_FEAT = 6  # load / solar / wind / calendar forecasts
    LATENT_DIM = 3  # renewable surplus, thermal scarcity, demand surge
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running encoder shape tests on: {device}\n")

    # ------------------------------------------------------------------
    # Instantiate encoder
    # ------------------------------------------------------------------
    encoder = LatentForceEncoder(
        history_feat_dim=HISTORY_FEAT,
        future_feat_dim=FUTURE_FEAT,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # ------------------------------------------------------------------
    # Synthetic random inputs (uniform for interpretable magnitude checks)
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    history = torch.randn(BATCH, HISTORY_LEN, HISTORY_FEAT, device=device)
    future_exog = torch.randn(BATCH, HORIZON, FUTURE_FEAT, device=device)

    # ------------------------------------------------------------------
    # Forward pass — eval mode (dropout disabled, deterministic)
    # ------------------------------------------------------------------
    encoder.eval()
    with torch.no_grad():
        out = encoder(history, future_exog)

    z0_mean = out["z0_mean"]
    z0_logvar = out["z0_logvar"]
    context = out["context"]

    # ------------------------------------------------------------------
    # Shape assertions
    # ------------------------------------------------------------------
    expected_z0_shape = (BATCH, LATENT_DIM)
    expected_context_shape = (BATCH, HORIZON, HIDDEN_DIM)

    assert z0_mean.shape == expected_z0_shape, (
        f"z0_mean shape mismatch: got {tuple(z0_mean.shape)}, "
        f"expected {expected_z0_shape}"
    )
    assert z0_logvar.shape == expected_z0_shape, (
        f"z0_logvar shape mismatch: got {tuple(z0_logvar.shape)}, "
        f"expected {expected_z0_shape}"
    )
    assert context.shape == expected_context_shape, (
        f"context shape mismatch: got {tuple(context.shape)}, "
        f"expected {expected_context_shape}"
    )

    # ------------------------------------------------------------------
    # Reparameterisation test
    # ------------------------------------------------------------------
    z0_sample = encoder.reparameterise(z0_mean, z0_logvar)
    assert z0_sample.shape == expected_z0_shape, (
        f"z0_sample shape mismatch: got {tuple(z0_sample.shape)}, "
        f"expected {expected_z0_shape}"
    )

    # Two draws from the same posterior should differ (stochastic)
    z0_sample_2 = encoder.reparameterise(z0_mean, z0_logvar)
    assert not torch.allclose(z0_sample, z0_sample_2), (
        "reparameterise() returned identical samples on two consecutive calls — "
        "randomness is broken"
    )

    # ------------------------------------------------------------------
    # Gradient flow test — ensure backprop reaches all parameters
    # ------------------------------------------------------------------
    encoder.train()
    out_train = encoder(history, future_exog)
    loss = (
        out_train["z0_mean"].sum()
        + out_train["z0_logvar"].sum()
        + out_train["context"].sum()
    )
    loss.backward()

    params_without_grad = [
        name
        for name, p in encoder.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(params_without_grad) == 0, (
        f"Parameters with no gradient after backward(): {params_without_grad}"
    )

    # ------------------------------------------------------------------
    # Numeric sanity checks
    # ------------------------------------------------------------------
    encoder.eval()
    with torch.no_grad():
        out_eval = encoder(history, future_exog)

    # Context should be finite after LayerNorm
    assert torch.isfinite(out_eval["context"]).all(), (
        "context contains NaN or Inf after LayerNorm"
    )

    # LayerNorm guarantees zero mean / unit variance across the last dim
    ctx = out_eval["context"]
    ctx_mean = ctx.mean(dim=-1).abs()  # (B, horizon)
    ctx_std = ctx.std(dim=-1)  # (B, horizon)
    assert (ctx_mean < 1e-4).all(), (
        f"LayerNorm mean not near zero; max abs mean = {ctx_mean.max():.6f}"
    )
    assert ((ctx_std - 1.0).abs() < 0.1).all(), (
        f"LayerNorm std not near 1; max deviation = {(ctx_std - 1.0).abs().max():.6f}"
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │               LatentForceEncoder — Shape Test Results            │")
    print("  ├──────────────────┬──────────────────────────────────────────────┤")
    print(f"  │ Input: history   │ {str(tuple(history.shape)):<42} │")
    print(f"  │ Input: future    │ {str(tuple(future_exog.shape)):<42} │")
    print("  ├──────────────────┼──────────────────────────────────────────────┤")
    print(f"  │ z0_mean          │ {str(tuple(z0_mean.shape)):<42} │")
    print(f"  │ z0_logvar        │ {str(tuple(z0_logvar.shape)):<42} │")
    print(f"  │ context          │ {str(tuple(context.shape)):<42} │")
    print(f"  │ z0_sample (reparam) │ {str(tuple(z0_sample.shape)):<39} │")
    print("  ├──────────────────┼──────────────────────────────────────────────┤")
    print(f"  │ Total params     │ {total_params:<42,} │")
    print(f"  │ Gradient flow    │ {'OK — all parameters receive gradients':<42} │")
    print(
        f"  │ LayerNorm mean   │ {f'max |mean| = {ctx_mean.max():.2e}  (< 1e-4 ✓)':<42} │"
    )
    print(
        f"  │ LayerNorm std    │ {f'max |std-1| = {(ctx_std - 1.0).abs().max():.4f}  (< 0.10 ✓)':<42} │"
    )
    print("  └──────────────────┴──────────────────────────────────────────────┘")
    print("\n  All assertions passed. LatentForceEncoder is ready.\n")

    sys.exit(0)
