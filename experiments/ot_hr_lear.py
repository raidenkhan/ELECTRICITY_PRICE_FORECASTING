"""
src/experiments/ot_hr_lear.py
==============================
Optimal Transport (OT) Adaptation for HR-LEAR.
Transports 'Base' residuals to 'Spike' space to augment training data.
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.mrs_lear_ensemble import crps_from_quantiles

def run_ot_hr_lear():
    print("[ot_hr_lear] Loading data...")
    feature_df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    res_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    
    data = res_df.join(feature_df.drop(columns=['price', 'split']), how='left')
    data = data.join(labels[[f'hmm_prob_{k}' for k in range(3)]], how='left')
    
    train_data = data[data['split'] == 'train'].copy()
    test_data = data[data['split'] == 'test'].copy()
    
    # 1. Optimal Transport Adaptation (Simplified 1D / Gaussian Mapping)
    # Goal: Use Regime 1 (Base) data to help Regime 2 (Spike)
    # Map residuals from R1 distribution to R2 distribution
    r1_mask = train_data['hmm_prob_1'] > 0.8
    r2_mask = train_data['hmm_prob_2'] > 0.8
    
    r1_res = train_data.loc[r1_mask, 'residual']
    r2_res = train_data.loc[r2_mask, 'residual']
    
    print(f"[ot_hr_lear] Augmenting Spike Regime (n={len(r2_res)}) with Base data (n={len(r1_res)})...")
    
    # 1D OT Mapping: y_new = (std2/std1) * (y - mu1) + mu2
    mu1, std1 = r1_res.mean(), r1_res.std()
    mu2, std2 = r2_res.mean(), r2_res.std()
    
    # Augmented dataset for Regime 2 specialist
    # Use actual R2 data + mapped R1 data
    aug_r2_res = (std2/std1) * (r1_res - mu1) + mu2
    
    # Weights for the augmented specialist
    # R2 data gets weight 1.0, Augmented data gets weight 0.5
    X_r2 = train_data.loc[r2_mask].select_dtypes(include=[np.number]).drop(columns=['residual', 'anchor_pred'], errors='ignore')
    X_r1_mapped = train_data.loc[r1_mask].select_dtypes(include=[np.number]).drop(columns=['residual', 'anchor_pred'], errors='ignore')
    
    X_combined = pd.concat([X_r2, X_r1_mapped])
    Y_combined = pd.concat([r2_res, aug_r2_res])
    W_combined = np.concatenate([np.ones(len(r2_res)), 0.3 * np.ones(len(aug_r2_res))])

    # 2. Train Augmented Specialist
    print("[ot_hr_lear] Training Augmented Specialist...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    expert_r2 = ElasticNetCV(cv=5)
    expert_r2.fit(X_scaled, Y_combined, sample_weight=W_combined)
    
    # 3. Predict and Fuse (only update Regime 2)
    # First get original ElasticNet corrections for all regimes
    from models.hr_lear_residuals import ResidualExpert
    corrections = np.zeros((len(test_data), 3))
    for k in range(3):
        expert = ResidualExpert(k)
        weights = train_data[f'hmm_prob_{k}'].values
        expert.fit(train_data, train_data['residual'], weights)
        if k == 2:
            # Use our OT-augmented expert
            X_test_scaled = scaler.transform(test_data[X_combined.columns])
            corrections[:, k] = expert_r2.predict(X_test_scaled)
        else:
            corrections[:, k] = expert.predict(test_data)
            
    # 4. Fusion with Anchor
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    
    anchor_out = anchor.predict(test_data, n_bootstrap=100)
    y_anchor = anchor_out['point']
    q_anchor = anchor_out['quantiles']
    y_true = test_data['price'].values
    
    prob_matrix = test_data[[f'hmm_prob_{k}' for k in range(3)]].values
    max_probs = prob_matrix.max(axis=1)
    guard_mask = max_probs < 0.6
    
    weighted_corr = np.sum(prob_matrix * corrections, axis=1)
    weighted_corr[guard_mask] = 0.0
    
    y_ot = y_anchor + weighted_corr
    q_ot = q_anchor + weighted_corr[:, np.newaxis]
    
    mae = np.mean(np.abs(y_true - y_ot))
    crps = crps_from_quantiles(y_true, q_ot, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    
    print(f"\n[ot_hr_lear] RESULTS: MAE={mae:.4f}, CRPS={crps:.4f}")
    
    # Save
    res = {'Approach': 'OT-Adaptation', 'MAE': mae, 'CRPS': crps}
    pd.DataFrame([res]).to_csv(ROOT / 'results/tables/ot_hr_lear_results.csv', index=False)

if __name__ == '__main__':
    run_ot_hr_lear()
