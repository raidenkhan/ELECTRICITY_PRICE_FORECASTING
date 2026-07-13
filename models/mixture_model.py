import torch
import torch.nn as nn
import numpy as np

class EPFMixtureCombiner(nn.Module):
    """
    Combines the predictions from individual CNP experts weighted by the 
    regime probabilities from the DS-HDP-HMM.
    
    Formula: P_final(t) = sum_k ( p(regime=k | X) * Expert_k(X) )
    """
    def __init__(self, mapping_dict):
        """
        mapping_dict maps HMM state indices to their corresponding expert model identifiers.
        E.g., { 0: 'evt', 1: 'tft', 2: 'solar', 3: 'tft' }
        Because HDP-HMM discovers states organically, we must map them dynamically 
        after interpretation (e.g. by checking average price or volatility in that state).
        """
        super().__init__()
        self.mapping_dict = mapping_dict

    def align_regimes(self, hmm_states_stats):
        """
        Helper method to systematically map discovered HMM states to specific experts.
        """
        # Logic: High median price / HIGH IQR -> EVT
        # Highly negative price -> Solar Regressor
        # Standard range -> TFT
        # This will populate self.mapping_dict based on data post-hoc.
        pass

    def forward(self, hmm_probs, expert_preds, asinh_transform=True, scaling_constant=50.0):
        """
        hmm_probs: Tensor of shape (batch, K) representing regime probabilities
        expert_preds: Dict mapping expert names to their point predictions (batch,)
                      e.g., {'evt': pred_1, 'tft': pred_2, 'solar': pred_3}
        """
        batch_size = hmm_probs.shape[0]
        K = hmm_probs.shape[1]
        
        # Compute the weighted sum
        final_prediction = torch.zeros(batch_size, device=hmm_probs.device)
        
        for k in range(K):
            if k in self.mapping_dict:
                expert_key = self.mapping_dict[k]
                pred_k = expert_preds[expert_key]
                final_prediction += hmm_probs[:, k] * pred_k
        
        if asinh_transform:
            # We predict in asinh space, convert back to nominal EUR/MWh
            final_prediction = scaling_constant * torch.sinh(final_prediction)
            
        return final_prediction

if __name__ == '__main__':
    print("Testing Mixture Combiner...")
    mapping = {0: 'tft', 1: 'evt', 2: 'solar'}
    combiner = EPFMixtureCombiner(mapping)
    
    hmm_probs = torch.tensor([[0.8, 0.1, 0.1], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])
    expert_preds = {
        'tft': torch.tensor([1.2, 1.1, 1.3]), # ~ 75 Eur
        'evt': torch.tensor([4.5, 5.0, 4.8]), # Spiky ~ 3700 Eur
        'solar': torch.tensor([-2.5, -2.1, -1.8]) # Negative ~ -300 Eur
    }
    
    out = combiner(hmm_probs, expert_preds)
    print("Combinder outputs (EUR/MWh):", out.tolist())
