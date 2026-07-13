import torch
import torch.nn as nn

class TiltedLoss(nn.Module):
    """
    Tilted (Pinball) Loss for Quantile Regression.
    L(y, yp) = tau * max(y-yp, 0) + (1-tau) * max(yp-y, 0)
    """
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, input, target):
        error = target - input
        return torch.mean(torch.max(self.quantile * error, (self.quantile - 1) * error))

class AsymmetricHuberLoss(nn.Module):
    """
    Asymmetric Huber Loss to penalize under/over-estimation differently 
    while maintaining robustness to outliers.
    """
    def __init__(self, delta=1.0, alpha=0.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha # Weights error if target > pred (under-estimation)

    def forward(self, input, target):
        error = target - input
        abs_error = torch.abs(error)
        
        # Standard Huber
        quad = torch.min(abs_error, torch.tensor(self.delta))
        lin = abs_error - quad
        huber = 0.5 * quad**2 + self.delta * lin
        
        # Apply asymmetry
        weight = torch.where(error > 0, self.alpha, 1.0 - self.alpha)
        return torch.mean(weight * huber)
