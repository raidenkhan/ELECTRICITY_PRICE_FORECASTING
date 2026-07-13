import numpy as np

class HawkesIntensity:
    """
    Hawkes Process with Exponential Kernel.
    intensity(t) = baseline + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    """
    def __init__(self, baseline=0.01, alpha=0.5, beta=1.0):
        self.baseline = baseline
        self.alpha = alpha
        self.beta = beta

    def fit_and_predict(self, events):
        """
        Calculates intensity sequence given binary event indicators.
        events: binary array (1 if spike, 0 otherwise)
        """
        T = len(events)
        intensity = np.zeros(T)
        current_intensity = self.baseline
        
        for t in range(T):
            intensity[t] = current_intensity
            # Update intensity for next step
            # Decay + Jump if event occurred
            current_intensity = self.baseline + (current_intensity - self.baseline) * np.exp(-self.beta)
            if events[t] > 0:
                current_intensity += self.alpha
                
        return intensity

def detect_spikes(prices, threshold_std=2.0):
    """Simple spike detector based on standard deviations from mean."""
    mean = prices.mean()
    std = prices.std()
    return (prices > (mean + threshold_std * std)).astype(int)
