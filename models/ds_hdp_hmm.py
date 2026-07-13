import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VariationalStickyHMM(nn.Module):
    """
    A PyTorch implementation of a finite approximation to the Sticky HDP-HMM.
    We use a large number of states K, and a sparse Dirichlet prior on transitions
    with an added sticky parameter to encourage state persistence.
    This serves as the 'DS-HDP-HMM' proxy suitable for gradient-based training.
    """
    def __init__(self, obs_dim, K_max=15, alpha=0.1, kappa=10.0, gamma=1.0):
        super().__init__()
        self.K = K_max
        self.obs_dim = obs_dim
        
        # Dirichlet hyperparameters
        self.alpha = alpha  # Concentration for cross-transitions (sparse)
        self.kappa = kappa  # Sticky parameter (self-transition bias)
        self.gamma = gamma  # Global concentration
        
        # Trainable parameters
        # Unnormalized transition matrix
        self.log_trans = nn.Parameter(torch.randn(self.K, self.K))
        # Initial state distribution
        self.log_init = nn.Parameter(torch.randn(self.K))
        
        # Emission parameters (Gaussian for daily stats)
        self.means = nn.Parameter(torch.randn(self.K, self.obs_dim))
        # Log-variances (diagonal covariance)
        self.log_vars = nn.Parameter(torch.zeros(self.K, self.obs_dim))
        # Degrees of freedom for Student's T (initialize around 5.0)
        self.log_dfs = nn.Parameter(torch.ones(self.K, self.obs_dim) * np.log(5.0))
        
    def get_transition_matrix(self):
        return F.softmax(self.log_trans, dim=1)
        
    def get_init_dist(self):
        return F.softmax(self.log_init, dim=0)
        
    def emission_log_prob(self, x):
        """
        x: (T, obs_dim)
        Returns: (T, K) log probabilities
        """
        T = x.shape[0]
        # x is (T, 1, obs_dim)
        x_exp = x.unsqueeze(1)
        # means is (1, K, obs_dim)
        mu = self.means.unsqueeze(0)
        # log_vars is (1, K, obs_dim)
        lv = self.log_vars.unsqueeze(0)
        
        # Student's t-distribution log density
        # More robust to outliers than Gaussian.
        # log_prob = log(Gamma((df+1)/2)) - log(Gamma(df/2)) - 0.5*log(pi*df*var) 
        #            - ((df+1)/2) * log(1 + (x-mu)^2 / (df*var))
        df = torch.exp(self.log_dfs.unsqueeze(0)) # (1, K, obs_dim) or (1, K, 1)
        var = torch.exp(lv)
        
        # log(Gamma((df+1)/2)) - log(Gamma(df/2)) approx logic
        # For simplicity in this proxy, we use a fixed df per regime or trainable
        
        diff = (x_exp - mu)**2 / (df * var)
        log_prob = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0) \
                   - 0.5 * (torch.log(np.pi * df) + lv) \
                   - ((df + 1.0) / 2.0) * torch.log(1.0 + diff)
        
        return log_prob.sum(dim=-1)

    def forward_algorithm(self, x):
        """
        Compute marginal log-likelihood of observation sequence x.
        x: (T, obs_dim)
        """
        T = x.shape[0]
        log_a = F.log_softmax(self.log_trans, dim=1) # (K, K)
        log_pi = F.log_softmax(self.log_init, dim=0) # (K,)
        log_emissions = self.emission_log_prob(x)    # (T, K)
        
        # Initialize forward variable alpha_0
        alpha_t = log_pi + log_emissions[0] # (K,)
        
        for t in range(1, T):
            # alpha_{t-1} + log A -> (K, 1) + (K, K) -> (K, K)
            # logsumexp over previous states (dim=0) -> (K,)
            alpha_t = torch.logsumexp(alpha_t.unsqueeze(1) + log_a, dim=0) + log_emissions[t]
            
        # Total log likelihood
        return torch.logsumexp(alpha_t, dim=0)
        
    def prior_loss(self):
        """
        Sticky Dirichlet Prior log density (up to a constant).
        Encourages sparsity in transitions but puts mass on the diagonal.
        """
        trans = self.get_transition_matrix()
        
        # Target prior alpha for row i:
        # P(row) ~ Dir(alpha + kappa * delta_{i=j})
        loss = 0
        for i in range(self.K):
            prior_alphas = torch.ones(self.K, device=trans.device) * self.alpha
            prior_alphas[i] += self.kappa
            
            # Log density of Dirichlet (ignoring beta terms since they are constant wrt parameters)
            # sum_j (prior_alpha_j - 1) * log(trans_ij)
            row_loss = torch.sum((prior_alphas - 1.0) * torch.log(trans[i] + 1e-8))
            loss += row_loss
        return loss

    def compute_loss(self, x):
        """
        Negative MAP loss to minimize.
        loss = -LogLikelihood - LogPrior
        """
        ll = self.forward_algorithm(x)
        prior = self.prior_loss()
        return -(ll + prior) / x.shape[0] # Normalize by sequence length

    def viterbi(self, x):
        """
        Viterbi algorithm for most likely state sequence.
        x: (T, obs_dim)
        Returns: 1D tensor of shape (T,)
        """
        T = x.shape[0]
        log_a = F.log_softmax(self.log_trans, dim=1)
        log_pi = F.log_softmax(self.log_init, dim=0)
        log_emissions = self.emission_log_prob(x)
        
        V = torch.zeros(T, self.K, device=x.device)
        ptr = torch.zeros(T, self.K, dtype=torch.long, device=x.device)
        
        V[0] = log_pi + log_emissions[0]
        
        for t in range(1, T):
            # V_{t-1} + log A -> (K, K)
            val = V[t-1].unsqueeze(1) + log_a
            max_val, max_idx = torch.max(val, dim=0)
            V[t] = max_val + log_emissions[t]
            ptr[t] = max_idx
            
        states = torch.zeros(T, dtype=torch.long, device=x.device)
        states[-1] = torch.argmax(V[-1])
        for t in range(T-1, 0, -1):
            states[t-1] = ptr[t, states[t]]
            
        return states

if __name__ == '__main__':
    # Test the HMM
    print("Testing DS-HDP-HMM proxy...")
    obs_dim = 5
    K = 10
    model = VariationalStickyHMM(obs_dim=obs_dim, K_max=K)
    x = torch.randn(100, obs_dim) # 100 days of daily stats
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        optimizer.zero_grad()
        loss = model.compute_loss(x)
        loss.backward()
        optimizer.step()
        
    states = model.viterbi(x)
    print("Optimization test complete. Extracted states shape:", states.shape)
    print("Unique states discovered:", torch.unique(states).tolist())
