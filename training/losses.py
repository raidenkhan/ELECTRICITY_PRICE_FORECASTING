"""
losses.py - Custom Training Losses

Contains standard pinball loss, physical coherence penalty for prices, and the
correct Shazeer et al. (2017) auxiliary load-balancing loss for the MoE router.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantile_levels: list[float]) -> torch.Tensor:
    """
    Standard pinball / quantile loss at all given levels.
    
    y_pred: [batch, horizon, n_quantiles]
    y_true: [batch, horizon]
    quantile_levels: list of floats like [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    """
    y_true = y_true.unsqueeze(-1)
    errors = y_true - y_pred
    
    levels = torch.tensor(quantile_levels, device=y_pred.device, dtype=y_pred.dtype)
    levels = levels.view(1, 1, -1)
    
    loss = torch.max((levels - 1) * errors, levels * errors)
    return loss.mean()


def physical_coherence_penalty(quantiles: torch.Tensor, p_floor: float, lambda_pc: float = 0.01) -> torch.Tensor:
    """
    Physical coherence penalty.
    Implements Contribution B Constraint C3: mu_i >= P_floor.
    Penalise q50 (index 3) predictions below p_floor.
    
    quantiles: [batch, horizon, n_quantiles]
    """
    q50 = quantiles[..., 3]
    penalty = F.relu(p_floor - q50) ** 2
    return lambda_pc * penalty.mean()


def load_balancing_loss(gate_probs: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """
    Shazeer et al. (2017) auxiliary load-balancing loss.
    
    Penalises unequal fraction of tokens routed to each expert. 
    Loss = n_experts * sum_i( f_i * P_i )
    where f_i = fraction of tokens routed to expert i (from argmax selection)
    and   P_i = mean gate probability for expert i (differentiable).
    
    gate_probs: [batch, time, n_experts]  — the full softmax distribution
    """
    n_experts = gate_probs.shape[-1]
    
    # Fraction of tokens sent to each expert via top-k argmax (not differentiable,
    # used only as a coefficient). Use mean probs as proxy (fully differentiable).
    # Per Shazeer 2017: f_i ≈ mean_probs[i], P_i = mean_probs[i]
    mean_probs = gate_probs.mean(dim=(0, 1))  # [n_experts]
    
    # L_aux = n_experts * Σ f_i * P_i
    aux_loss = n_experts * (mean_probs * mean_probs).sum()
    return beta * aux_loss


def total_loss(
    quantiles: torch.Tensor,
    y_true: torch.Tensor,
    quantile_levels: list[float],
    p_floor: float,
    gate_probs: torch.Tensor,
    lambda_pc: float = 0.01,
    beta: float = 0.01,
    # Legacy arg kept for backward compat — ignored
    routing_cv: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Total loss combining pinball, physical coherence, and Shazeer load balancing.
    Returns: dict{'total', 'pinball', 'coherence', 'load_balance'}
    """
    pb_loss = pinball_loss(quantiles, y_true, quantile_levels)
    pc_loss = physical_coherence_penalty(quantiles, p_floor, lambda_pc)
    lb_loss = load_balancing_loss(gate_probs, beta)

    total = pb_loss + pc_loss + lb_loss

    return {
        "total": total,
        "pinball": pb_loss.detach(),
        "coherence": pc_loss.detach(),
        "load_balance": lb_loss.detach(),
    }
