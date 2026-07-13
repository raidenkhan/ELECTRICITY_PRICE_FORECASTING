"""
src/experiments/pi_hr_lear.py
==============================
Physics-Informed HR-LEAR implementation.
Uses net load ramps and fuel interactions in specialists.
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.hr_lear_residuals import ResidualExpert
from models.mrs_lear_ensemble import crps_from_quantiles

def run_pi_hr_lear():
    print("[pi_hr_lear] Loading physics data and residuals...")
    feature_df = pd.read_parquet(ROOT / 'data/processed/feature_matrix_physics.parquet')
    res_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    
    data = res_df.join(feature_df.drop(columns=['price', 'split']), how='left')
    data = data.join(labels[[f'hmm_prob_{k}' for k in range(3)]], how='left')
    
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    
    # 1. Load Anchor
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    
    anchor_out = anchor.predict(test_data, n_bootstrap=100)
    y_anchor = anchor_out['point']
    q_anchor = anchor_out['quantiles']
    y_true = test_data['price'].values
    
    # 2. Train Physics-Informed Residual Experts
    print("[pi_hr_lear] Training Physics-Informed Specialists...")
    experts = {}
    corrections = np.zeros((len(test_data), 3))
    
    for k in range(3):
        expert = ResidualExpert(k)
        weights = train_data[f'hmm_prob_{k}'].values
        expert.fit(train_data, train_data['residual'], weights)
        corrections[:, k] = expert.predict(test_data)
        
    # 3. Fusion
    prob_matrix = test_data[[f'hmm_prob_{k}' for k in range(3)]].values
    max_probs = prob_matrix.max(axis=1)
    guard_mask = max_probs < 0.6
    
    weighted_corr = np.sum(prob_matrix * corrections, axis=1)
    weighted_corr[guard_mask] = 0.0
    
    y_pi = y_anchor + weighted_corr
    q_pi = q_anchor + weighted_corr[:, np.newaxis]
    
    mae = np.mean(np.abs(y_true - y_pi))
    crps = crps_from_quantiles(y_true, q_pi, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    
    print(f"\n[pi_hr_lear] RESULTS: MAE={mae:.4f}, CRPS={crps:.4f}")
    
    # Save results
    res = {'Approach': 'PI-HR-LEAR', 'MAE': mae, 'CRPS': crps}
    pd.DataFrame([res]).to_csv(ROOT / 'results/tables/pi_hr_lear_results.csv', index=False)

if __name__ == '__main__':
    run_pi_hr_lear()
