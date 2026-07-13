"""
mphil_audit.py — MPhil Research Audit for HR-LEAR
==================================================
1. Energy Score Comparison (LEAR vs HR-LEAR)
2. Ablation Check (Flat-LEAR vs HR-LEAR)
3. Visualization: Residual Distribution in Spike Regime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.baselines.lear import LEARBaseline
from models.hr_lear_residuals import ResidualExpert
from evaluation.metrics import crps_from_quantiles

# =============================================================================
# 1. Energy Score Implementation
# =============================================================================

def compute_energy_score(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """
    y_true: [N, 24]
    y_samples: [N, M, 24]
    """
    N, M, H = y_samples.shape
    
    # Term 1: E||X - y||
    # ||X - y|| for each sample: [N, M]
    diff_y = y_samples - y_true[:, np.newaxis, :]
    dist_y = np.linalg.norm(diff_y, axis=2) # [N, M]
    term1 = np.mean(dist_y)
    
    # Term 2: 0.5 * E||X - X'||
    # To avoid M^2 calculation, we use a trick: 
    # Sample two indices for each day
    idx1 = np.random.choice(M, size=(N, M), replace=True)
    idx2 = np.random.choice(M, size=(N, M), replace=True)
    
    # Correct indexing for numpy
    # We need to pick one sample per day, but M samples per day
    # X1[i, j, :] = y_samples[i, idx1[i, j], :]
    
    # Faster way with fancy indexing
    row_idx = np.arange(N)[:, np.newaxis]
    X1 = y_samples[row_idx, idx1, :]
    X2 = y_samples[row_idx, idx2, :]
    
    dist_xx = np.linalg.norm(X1 - X2, axis=2) # [N, M]
    term2 = 0.5 * np.mean(dist_xx)
    
    return term1 - term2

def path_bootstrap_predict(model, test_df, n_bootstrap=500):
    """
    Generates 24-hour path samples using day-wise residual bootstrap.
    Assumes model has .residuals[h] which are residual vectors.
    """
    # Group test_df by day
    test_df['date'] = test_df.index.date
    days = test_df['date'].unique()
    
    # We need to make sure residuals have the same "days" across all hours
    all_res = []
    for h in range(24):
        res = model.residuals[h]
        all_res.append(res)
    
    # Find min length
    min_len = min(len(r) for r in all_res)
    stacked_res = np.zeros((min_len, 24))
    for h in range(24):
        stacked_res[:, h] = all_res[h][:min_len]
        
    y_samples = np.zeros((len(days), n_bootstrap, 24))
    y_true_days = np.zeros((len(days), 24))
    
    print(f"[audit] Generating samples for {len(days)} days...")
    
    for i, d in enumerate(days):
        day_data = test_df[test_df['date'] == d]
        if len(day_data) != 24: continue
        
        y_true_days[i, :] = day_data['price'].values
        
        # Point forecasts
        preds = np.zeros(24)
        for h in range(24):
            hour_data = day_data[day_data.index.hour == h]
            X = model._prepare_features(hour_data)
            X_scaled = model.scalers[h].transform(X)
            preds[h] = model.models[h].predict(X_scaled)
            
        # Sample residual vectors
        res_idx = np.random.choice(min_len, size=n_bootstrap, replace=True)
        sampled_res_vecs = stacked_res[res_idx, :] # [M, 24]
        
        y_samples[i, :, :] = preds[np.newaxis, :] + sampled_res_vecs
        
    return y_true_days, y_samples

# =============================================================================
# 2. Flat-LEAR Baseline
# =============================================================================

class FlatLEARBaseline:
    def __init__(self):
        self.regime_models = {} # (regime, hour) -> model
        
    def fit(self, df):
        print("[audit] Fitting Flat-LEAR (Regime-Specific Partitioning)...")
        for r in range(3):
            regime_df = df[df['hmm_regime'] == r]
            if len(regime_df) < 50: 
                print(f"  Regime {r} too small ({len(regime_df)}), skipping.")
                continue
            
            # Simple LEAR per hour within this regime
            for h in range(24):
                h_df = regime_df[regime_df.index.hour == h]
                if len(h_df) < 10: continue
                
                model_helper = LEARBaseline() # Using a sub-instance to reuse feature prep
                X = model_helper._prepare_features(h_df)
                y = h_df['price'].values
                
                from sklearn.linear_model import ElasticNet
                m = ElasticNet(alpha=0.1)
                m.fit(X, y)
                self.regime_models[(r, h)] = m
                
    def predict(self, df):
        preds = np.zeros(len(df))
        model_helper = LEARBaseline()
        for (r, h), m in self.regime_models.items():
            idx = (df['hmm_regime'] == r) & (df.index.hour == h)
            if not idx.any(): continue
            
            X = model_helper._prepare_features(df[idx])
            preds[idx] = m.predict(X)
        return preds

# =============================================================================
# Main Audit Execution
# =============================================================================

def run_audit():
    print("[audit] Starting MPhil Research Audit...")
    
    # Load Data
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    df = df.join(labels[['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']], how='left')
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test'].copy()
    
    # Load Models
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    with open(ROOT / 'models/hr_lear_residual_experts.pkl', 'rb') as f:
        experts = pickle.load(f)
        
    # --- Task 1: Energy Score ---
    print("\n--- Task 1: Energy Score Comparison ---")
    y_true_days, y_samples_lear = path_bootstrap_predict(anchor, test_df)
    es_lear = compute_energy_score(y_true_days, y_samples_lear)
    
    # HR-LEAR Energy Score (apply corrections to samples)
    prob_cols = [f'hmm_prob_{k}' for k in range(3)]
    test_df['date'] = test_df.index.date
    unique_days = np.unique(test_df['date'].values)
    
    y_samples_hr = y_samples_lear.copy()
    
    for i, d in enumerate(unique_days):
        day_data = test_df[test_df['date'] == d]
        if len(day_data) != 24: continue
        
        # Calculate corrections
        corrs = np.zeros((24, 3))
        for k in range(3):
            if k in experts:
                corrs[:, k] = experts[k].predict(day_data)
        
        probs = day_data[prob_cols].values
        max_probs = probs.max(axis=1)
        guard_mask = max_probs < 0.6
        
        weighted_corr = np.sum(probs * corrs, axis=1)
        weighted_corr[guard_mask] = 0.0
        
        # Apply to all samples for this day
        y_samples_hr[i, :, :] += weighted_corr[np.newaxis, :]
        
    es_hr = compute_energy_score(y_true_days, y_samples_hr)
    
    print(f"  Energy Score (Global LEAR): {es_lear:.4f}")
    print(f"  Energy Score (HR-LEAR)   : {es_hr:.4f}")
    print(f"  Improvement: {((es_lear - es_hr)/es_lear)*100:.2f}%")

    # --- Task 2: Ablation (Flat-LEAR) ---
    print("\n--- Task 2: Ablation (Flat-LEAR vs HR-LEAR) ---")
    flat_model = FlatLEARBaseline()
    flat_model.fit(train_df)
    y_pred_flat = flat_model.predict(test_df)
    
    # Calculate MAE for Flat-LEAR
    y_true = test_df['price'].values
    # Note: Flat-LEAR might have zeros where it didn't fit. Handle that?
    # For now, just compare on valid predictions
    valid_idx = y_pred_flat != 0
    mae_flat = np.mean(np.abs(y_pred_flat[valid_idx] - y_true[valid_idx]))
    
    # Load HR-LEAR metrics from file
    hr_res = pd.read_csv(ROOT / 'results/tables/hr_lear_results.csv')
    mae_hr = hr_res['MAE'].values[0]
    
    print(f"  MAE (Flat-LEAR) : {mae_flat:.4f}")
    print(f"  MAE (HR-LEAR)   : {mae_hr:.4f}")
    print(f"  HR-LEAR is {((mae_flat - mae_hr)/mae_flat)*100:.2f}% better than discrete partitioning.")

    # --- Task 3: Residual Visualizations ---
    print("\n--- Task 3: Residual Distribution in Spike Regime ---")
    spike_idx = test_df['hmm_regime'] == 2
    spike_df = test_df[spike_idx].copy()
    
    # Global Anchor Residuals
    y_anchor = anchor.predict(spike_df)['point']
    res_anchor = spike_df['price'].values - y_anchor
    
    # HR-LEAR Residuals
    corrs = np.zeros((len(spike_df), 3))
    for k in range(3):
        if k in experts:
            corrs[:, k] = experts[k].predict(spike_df)
    probs = spike_df[prob_cols].values
    weighted_corr = np.sum(probs * corrs, axis=1)
    y_hr = y_anchor + weighted_corr
    res_hr = spike_df['price'].values - y_hr
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(res_anchor, label='Global Anchor Residuals', fill=True, alpha=0.3, color='red')
    sns.kdeplot(res_hr, label='HR-LEAR Residuals', fill=True, alpha=0.3, color='blue')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Residual Distribution (Spike Regime 2)")
    plt.xlabel("Forecast Error (€/MWh)")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = ROOT / 'results/figures/fig_06d_residual_correction.png'
    plt.savefig(save_path)
    print(f"  Saved residual distribution plot to {save_path}")
    
    # Save audit results to a text file
    with open(ROOT / 'results/mphil_audit_results.txt', 'w') as f:
        f.write("MPhil Research Audit Results\n")
        f.write("============================\n\n")
        f.write(f"Task 1: Energy Score (24h Coherence)\n")
        f.write(f"  Global LEAR: {es_lear:.4f}\n")
        f.write(f"  HR-LEAR:     {es_hr:.4f}\n")
        f.write(f"  Improvement: {((es_lear - es_hr)/es_lear)*100:.2f}%\n\n")
        f.write(f"Task 2: Ablation (Flat-LEAR vs HR-LEAR)\n")
        f.write(f"  Flat-LEAR MAE: {mae_flat:.4f}\n")
        f.write(f"  HR-LEAR MAE:   {mae_hr:.4f}\n")
        f.write(f"  Improvement:   {((mae_flat - mae_hr)/mae_flat)*100:.2f}%\n\n")
        f.write(f"Task 3: Residual Variance in Spike Regime\n")
        f.write(f"  Global Anchor Residual Std: {np.std(res_anchor):.4f}\n")
        f.write(f"  HR-LEAR Residual Std:       {np.std(res_hr):.4f}\n")
        f.write(f"  Variance Reduction:         {((np.var(res_anchor) - np.var(res_hr))/np.var(res_anchor))*100:.2f}%\n")

if __name__ == '__main__':
    run_audit()
