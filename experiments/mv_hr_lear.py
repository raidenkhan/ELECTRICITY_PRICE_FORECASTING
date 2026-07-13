"""
src/experiments/mv_hr_lear.py
==============================
Multivariate HR-LEAR implementation.
Uses multi-output specialists to capture intra-day dependencies.
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.mrs_lear_ensemble import crps_from_quantiles

def run_mv_hr_lear():
    print("[mv_hr_lear] Loading data...")
    feature_df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    res_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    
    # Pivot residuals to 24h vector format
    res_df['hour'] = res_df.index.hour
    res_df['date'] = res_df.index.date
    res_vec = res_df.pivot(index='date', columns='hour', values='residual')
    
    # Also get features at daily level (average or morning features)
    daily_features = feature_df.groupby(feature_df.index.date).mean(numeric_only=True)
    daily_labels = labels.groupby(labels.index.date).mean(numeric_only=True)
    daily_split = feature_df['split'].groupby(feature_df.index.date).first()
    
    train_idx = daily_split[daily_split == 'train'].index
    test_idx = daily_split[daily_split == 'test'].index
    
    X_train = daily_features.loc[train_idx].drop(columns=['price'], errors='ignore')
    Y_train = res_vec.loc[train_idx].fillna(0)
    
    X_test = daily_features.loc[test_idx].drop(columns=['price'], errors='ignore')
    Y_true_res = res_vec.loc[test_idx].fillna(0)

    # 1. Train Multivariate Specialists
    print("[mv_hr_lear] Training Multi-output Specialists...")
    experts = {}
    for k in range(3):
        weights = daily_labels.loc[train_idx, f'hmm_prob_{k}'].values
        if weights.sum() < 1e-3: continue
        
        # Multi-output GBM
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=50))
        model.fit(X_train, Y_train, sample_weight=weights)
        experts[k] = model

    # 2. Predict and Fuse
    corrections = np.zeros((len(test_idx), 24, 3))
    for k, model in experts.items():
        corrections[:, :, k] = model.predict(X_test)
        
    prob_matrix = daily_labels.loc[test_idx, [f'hmm_prob_{k}' for k in range(3)]].values
    weighted_corr = np.zeros((len(test_idx), 24))
    for i in range(len(test_idx)):
        weighted_corr[i, :] = np.dot(corrections[i, :, :], prob_matrix[i, :])
        
    # Flatten back to hourly
    mv_corr_hourly = weighted_corr.flatten()
    
    # 3. Apply to Anchor
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    
    # We need the hourly test set for anchor predictions
    test_df_hourly = feature_df[feature_df['split'] == 'test']
    anchor_out = anchor.predict(test_df_hourly, n_bootstrap=100)
    y_anchor = anchor_out['point']
    q_anchor = anchor_out['quantiles']
    y_true = test_df_hourly['price'].values
    
    # Ensure length matches (might have dropped some days if incomplete)
    # Re-align
    y_mv = y_anchor + mv_corr_hourly[:len(y_anchor)]
    q_mv = q_anchor + mv_corr_hourly[:len(y_anchor), np.newaxis]
    
    mae = np.mean(np.abs(y_true - y_mv))
    crps = crps_from_quantiles(y_true, q_mv, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    
    print(f"\n[mv_hr_lear] RESULTS: MAE={mae:.4f}, CRPS={crps:.4f}")
    
    # Save
    res = {'Approach': 'MV-HR-LEAR', 'MAE': mae, 'CRPS': crps}
    pd.DataFrame([res]).to_csv(ROOT / 'results/tables/mv_hr_lear_results.csv', index=False)

if __name__ == '__main__':
    run_mv_hr_lear()
