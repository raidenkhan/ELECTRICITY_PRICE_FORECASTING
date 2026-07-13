import torch
import torch.nn as nn
import torch.nn.functional as F

class EVTExpert(nn.Module):
    """
    CNP-1: Extreme Value Theory (EVT) Expert for Spike/Dark Doldrums Regime.
    Outputs parameters for a Generalized Pareto Distribution (GPD):
    - threshold (mu)
    - scale (sigma) > 0
    - shape (xi) > 0 (heavy tail)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # mu, log_sigma, xi_raw
        )
        
    def forward(self, x):
        out = self.net(x)
        mu = out[..., 0]
        # Add softplus/exp for strictly positive scale
        sigma = F.softplus(out[..., 1]) + 1e-4
        # Shape parameter usually small positive for heavy tails
        xi = F.softplus(out[..., 2]) + 1e-4 
        return mu, sigma, xi

    def gpd_nll_loss(self, y_true, mu, sigma, xi):
        """
        Negative Log-Likelihood for GPD.
        y_true should be > mu for valid GPD, but we soft-penalize violations
        or mask them in practice.
        """
        # y - mu
        excess = y_true - mu
        # Mask where excess > 0
        valid = (excess > 0)
        
        # GPD log prob: -ln(sigma) - (1 + 1/xi) * ln(1 + xi * excess / sigma)
        term1 = torch.log(sigma[valid])
        term2 = (1.0 + 1.0 / xi[valid]) * torch.log(1.0 + xi[valid] * excess[valid] / sigma[valid])
        nll = term1 + term2
        
        # For non-exceedances, penalize heavily to force mu below y_true
        penalty = F.mse_loss(mu[~valid], y_true[~valid]) * 10.0
        
        if valid.sum() > 0:
            return nll.mean() + penalty
        return penalty

class SolarConditionedExpert(nn.Module):
    """
    CNP-3: Solar-Curve Conditioned Regression for Negative Price Regime.
    Prices in this regime are heavily driven by convex interactions between
    residual load and solar penetration.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # We explicitly model the non-linear interaction
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # We assume x contains features, including solar and residual load.
        # The network learns the convex mapping natively.
        features = self.feature_net(x)
        pred_price = self.out_net(features)
        return pred_price.squeeze(-1)

if __name__ == '__main__':
    print("Testing EVT and Solar Experts...")
    x = torch.randn(32, 20)
    y = torch.randn(32) * 100 + 400 # Spike prices
    
    evt = EVTExpert(20)
    mu, sigma, xi = evt(x)
    loss = evt.gpd_nll_loss(y, mu, sigma, xi)
    print("EVT Loss:", loss.item())
    
    solar = SolarConditionedExpert(20)
    pred_neg = solar(x)
    print("Solar Regressor output shape:", pred_neg.shape)
