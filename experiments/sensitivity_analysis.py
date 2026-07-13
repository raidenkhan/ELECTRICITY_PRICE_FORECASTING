"""
sensitivity_analysis.py — Sensitivity Analysis of SPI Threshold (tau)
=====================================================================
Evaluates the impact of the Stability-Preserving Indicator (SPI) 
threshold tau on the 24-hour Energy Score.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

# Hack for pickle
try:
    from models.hr_lear_residuals import ResidualExpert
    import __main__
    __main__.ResidualExpert = ResidualExpert
except ImportError:
    pass

def compute_energy_score(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """Fast Energy Score computation."""
    N, M, H = y_samples.shape
    diff_y = y_samples - y_true[:, np.newaxis, :]
    dist_y = np.linalg.norm(diff_y, axis=2)
    term1 = np.mean(dist_y)
    
    idx1 = np.random.choice(M, size=(N, M), replace=True)
    idx2 = np.random.choice(M, size=(N, M), replace=True)
    row_idx = np.arange(N)[:, np.newaxis]
    X1 = y_samples[row_idx, idx1, :]
    X2 = y_samples[row_idx, idx2, :]
    dist_xx = np.linalg.norm(X1 - X2, axis=2)
    term2 = 0.5 * np.mean(dist_xx)
    return term1 - term2

def path_bootstrap_predict(model, test_df, n_bootstrap=200):
    """Day-wise path bootstrap."""
    test_df['date'] = test_df.index.date
    days = test_df['date'].unique()
    all_res = [model.residuals[h] for h in range(24)]
    min_len = min(len(r) for r in all_res)
    stacked_res = np.zeros((min_len, 24))
    for h in range(24):
        stacked_res[:, h] = all_res[h][:min_len]
        
    y_samples = np.zeros((len(days), n_bootstrap, 24))
    y_true_days = np.zeros((len(days), 24))
    
    for i, d in enumerate(days):
        day_data = test_df[test_df['date'] == d]
        if len(day_data) != 24: continue
        y_true_days[i, :] = day_data['price'].values
        preds = np.zeros(24)
        for h in range(24):
            X = model._prepare_features(day_data[day_data.index.hour == h])
            X_scaled = model.scalers[h].transform(X)
            preds[h] = model.models[h].predict(X_scaled)
        res_idx = np.random.choice(min_len, size=n_bootstrap, replace=True)
        y_samples[i, :, :] = preds[np.newaxis, :] + stacked_res[res_idx, :]
        
    return y_true_days, y_samples

def run_sensitivity():
    print("[sensitivity] Starting SPI Threshold Sensitivity Analysis...")
    
    # Load Data
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    df = df.join(labels[['hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']], how='left')
    test_df = df[df['split'] == 'test'].copy()
    
    # Load Models
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    with open(ROOT / 'models/hr_lear_residual_experts.pkl', 'rb') as f:
        experts = pickle.load(f)
        
    # Baseline paths (tau=1.0 / no correction)
    y_true_days, y_samples_base = path_bootstrap_predict(anchor, test_df)
    
    # Get corrections for all days
    prob_cols = ['hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    test_days = test_df.index.date
    unique_days = np.unique(test_days)
    
    day_corrections = []
    day_max_probs = []
    for d in unique_days:
        day_data = test_df[test_df.index.date == d]
        if len(day_data) != 24: 
            day_corrections.append(None)
            day_max_probs.append(None)
            continue
        
        corrs = np.zeros((24, 3))
        for k in range(3):
            if k in experts:
                corrs[:, k] = experts[k].predict(day_data)
        probs = day_data[prob_cols].values
        weighted_corr = np.sum(probs * corrs, axis=1)
        day_corrections.append(weighted_corr)
        day_max_probs.append(probs.max(axis=1))

    # Sweep tau
    taus = np.linspace(0.0, 1.0, 11)
    results = []
    
    for tau in taus:
        print(f"  Testing tau = {tau:.1f}...")
        y_samples_tau = y_samples_base.copy()
        
        for i, d in enumerate(unique_days):
            if day_corrections[i] is None: continue
            
            corr = day_corrections[i].copy()
            max_p = day_max_probs[i]
            # Uncertainty Guard logic
            guard_mask = max_p < tau
            corr[guard_mask] = 0.0
            
            y_samples_tau[i, :, :] += corr[np.newaxis, :]
            
        es = compute_energy_score(y_true_days, y_samples_tau)
        results.append({'tau': tau, 'energy_score': es})
        
    res_df = pd.DataFrame(results)
    print("\n[sensitivity] Results:")
    print(res_df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x='tau', y='energy_score', marker='o', color='darkred', lw=2)
    plt.axvline(0.6, color='black', linestyle='--', label='Selected Threshold (0.6)')
    plt.title("Sensitivity Analysis: Impact of SPI Threshold (tau) on Energy Score", fontsize=12)
    plt.xlabel("Stability Threshold (tau)", fontsize=10)
    plt.ylabel("24-hour Energy Score", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = ROOT / 'results/figures/fig_07_tau_sensitivity.png'
    plt.savefig(save_path, dpi=200)
    print(f"  Saved plot to {save_path}")
    
    # Copy to papers/figures
    (ROOT / 'papers/figures').mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(save_path, ROOT / 'papers/figures/fig_07_tau_sensitivity.png')

if __name__ == '__main__':
    run_sensitivity()
