"""
metrics.py - Evaluation Metrics

Implements CRPS exact computation from quantiles and regime-specific evaluation.
"""

from __future__ import annotations

import torch

def crps_from_quantiles(y_true: torch.Tensor, quantile_preds: torch.Tensor, quantile_levels: list[float]) -> float:
    """
    Implements CRPS from discrete quantiles based on Gneiting & Raftery (2007) Eq 21.
    For discrete quantile levels, CRPS can be approximated by averaging the pinball loss 
    across all predicted quantiles (scaled by 2).
    
    CRPS = 2 * sum_i (alpha_i - 1_y_true<q_i) * (y_true - q_i) / n_levels
    
    y_true: [batch, horizon]
    quantile_preds: [batch, horizon, n_quantiles]
    quantile_levels: list of floats e.g., [0.1, 0.2, ... 0.9]
    """
    levels = torch.tensor(quantile_levels, device=y_true.device, dtype=y_true.dtype).view(1, 1, -1)
    y_true_expanded = y_true.unsqueeze(-1)
    
    errors = y_true_expanded - quantile_preds
    # Standard Pinball Loss definition
    loss = torch.max((levels - 1) * errors, levels * errors)
    
    # 2 * expectation of quantile pinball loss corresponds exactly to CRPS
    crps = 2.0 * loss.mean(dim=-1) # average over quantiles
    return float(crps.mean().item()) # average over batch & horizon

def regime_crps(y_true: torch.Tensor, quantile_preds: torch.Tensor, regime_labels: torch.Tensor, quantile_levels: list[float]) -> dict[str, float]:
    """
    Return CRPS decomposed per regime.
    
    y_true: [batch, horizon]
    quantile_preds: [batch, horizon, n_quantiles]
    regime_labels: [batch, horizon]
    """
    metrics = {}
    levels = torch.tensor(quantile_levels, device=y_true.device, dtype=y_true.dtype).view(1, 1, -1)
    y_true_expanded = y_true.unsqueeze(-1)
    
    errors = y_true_expanded - quantile_preds
    loss = torch.max((levels - 1) * errors, levels * errors)
    crps_all = 2.0 * loss.mean(dim=-1) # [batch, horizon]
    
    for r in range(4): # 4 regimes mapped: 0, 1, 2, 3
        mask = (regime_labels == r)
        if mask.sum() > 0:
            metrics[f'regime_{r}'] = float(crps_all[mask].mean().item())
        else:
            # Handle empty regimes
            metrics[f'regime_{r}'] = float('nan')
            
    return metrics
