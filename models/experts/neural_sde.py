import torch
import torch.nn as nn
import numpy as np

class NeuralSDEExpert(nn.Module):
    """
    Stabilized Neural SDE Expert.
    Fixed theta for strong mean reversion to the predicted equilibrium.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        
        # Fixed Reversion Speed (conservative 0.1)
        self.theta = 0.1 
        
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Use Tanh for more stable gradients in drift
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Volatility capped at 1.0 (asinh units)
        )
        
    def drift(self, y, x):
        mu_x = self.mu_net(x)
        return self.theta * (mu_x - y)
        
    def diffusion(self, y, x):
        return self.sigma_net(x) * 0.5 + 0.05 # Range [0.05, 0.55]
        
    def forward(self, x, y_start, dt=1.0, steps=24, n_paths=1, sub_steps=2):
        batch_size = x.shape[0]
        paths = []
        curr_y = y_start.repeat(1, n_paths) 
        h = dt / sub_steps
        
        for _ in range(steps):
            for _ in range(sub_steps):
                d = self.drift(curr_y.view(-1, 1), x.repeat_interleave(n_paths, dim=0))
                s = self.diffusion(curr_y.view(-1, 1), x.repeat_interleave(n_paths, dim=0))
                d = d.view(batch_size, n_paths); s = s.view(batch_size, n_paths)
                
                # We use a deterministic path for training/mean-inference to avoid noise
                if self.training:
                    curr_y = curr_y + d * h
                else:
                    dW = torch.randn_like(curr_y) * np.sqrt(h)
                    curr_y = curr_y + d * h + s * dW
                    
                curr_y = torch.clamp(curr_y, -5.0, 10.0)
            paths.append(curr_y.clone())
            
        return torch.stack(paths, dim=2)

    def compute_loss(self, x, y_seq):
        """
        MSE path loss for stability.
        """
        y_start = y_seq[:, 0:1]
        target_path = y_seq[:, 1:] # Target for hour 1..24
        
        self.train()
        pred_path = self.forward(x, y_start, steps=target_path.shape[1], n_paths=1)
        pred_path = pred_path.squeeze(1) # (batch, steps)
        
        return nn.MSELoss()(pred_path, target_path)

if __name__ == '__main__':
    # Final sanity check
    input_dim = 15
    model = NeuralSDEExpert(input_dim=input_dim)
    x = torch.randn(2, input_dim)
    y_seq = torch.randn(2, 25) # 0 to 24
    loss = model.compute_loss(x, y_seq)
    print(f"Path Loss: {loss.item():.4f}")
