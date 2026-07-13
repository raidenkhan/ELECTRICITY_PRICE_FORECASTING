"""
hr_lear_ensemble.py — HR-LEAR Step 3: Fusion Ensemble
======================================================
Fuses Global LEAR anchor with Regime Residual Experts.
Implements the uncertainty guard and final evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Fix for pickle unpickling
sys.path.append(str(Path(__file__).parent))
try:
    from hr_lear_residuals import ResidualExpert
    import __main__
    __main__.ResidualExpert = ResidualExpert
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.mrs_lear_ensemble import crps_from_quantiles

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

def evaluate_hr_lear():
    print("[hr_lear_fuse] Loading models and data...")
    models_dir = ROOT / 'models'
    
    with open(models_dir / 'hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    with open(models_dir / 'hr_lear_residual_experts.pkl', 'rb') as f:
        experts = pickle.load(f)
        
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    df = df.join(labels[[f'hmm_prob_{k}' for k in range(3)] + ['hmm_regime']], how='left')
    
    test_df = df[df['split'] == 'test'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    
    # 1. Global Anchor Predictions (with full quantiles)
    print("[hr_lear_fuse] Step 1: Getting Global Anchor Predictions...")
    anchor_out = anchor.predict(test_df, n_bootstrap=500)
    y_anchor = anchor_out['point']
    q_anchor = anchor_out['quantiles']
    
    # 2. Residual Corrections
    print("[hr_lear_fuse] Step 2: Computing Regime-Residual Corrections...")
    corrections = np.zeros((len(test_df), 3))
    for k in range(3):
        if k in experts:
            corrections[:, k] = experts[k].predict(test_df)
            
    # 3. Soft-Blend Fusion
    print("[hr_lear_fuse] Step 3: Fusing Predictions with Uncertainty Guard...")
    prob_matrix = test_df[[f'hmm_prob_{k}' for k in range(3)]].values
    max_probs = prob_matrix.max(axis=1)
    
    # Uncertainty Guard: If confidence < 0.6, correction is 0 (keep global anchor)
    guard_mask = max_probs < 0.6
    
    weighted_corr = np.sum(prob_matrix * corrections, axis=1)
    weighted_corr[guard_mask] = 0.0
    
    # Final HR-LEAR Point Forecast
    y_hr = y_anchor + weighted_corr
    
    # Final HR-LEAR Quantiles: Shift the anchor quantiles by the correction
    q_hr = np.zeros_like(q_anchor)
    for j in range(len(QUANTILE_LEVELS)):
        q_hr[:, j] = q_anchor[:, j] + weighted_corr
        
    # Metrics
    y_true = test_df['price'].values
    mae = float(np.mean(np.abs(y_hr - y_true)))
    rmse = float(np.sqrt(np.mean((y_hr - y_true) ** 2)))
    crps = crps_from_quantiles(y_true, q_hr, QUANTILE_LEVELS)
    
    print(f"\n[hr_lear_fuse] === HR-LEAR TEST RESULTS ===")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  CRPS : {crps:.4f}")
    
    # Save metrics
    out_dir = ROOT / 'results/tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {'model': 'HR-LEAR', 'MAE': mae, 'RMSE': rmse, 'CRPS': crps}
    pd.DataFrame([res]).to_csv(out_dir / 'hr_lear_results.csv', index=False)
    print(f"[hr_lear_fuse] Saved metrics -> {out_dir / 'hr_lear_results.csv'}")

    # Save full predictions
    pred_dir = ROOT / 'results/predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        'datetime': test_df.index,
        'y_true': y_true,
        'y_pred': y_hr,
        'regime': test_df['hmm_regime'].values
    })
    for j, q in enumerate(QUANTILE_LEVELS):
        pred_df[f'q{int(q*100):02d}'] = q_hr[:, j]
    
    pred_df.to_parquet(pred_dir / 'hr_lear_predictions.parquet')
    
    return crps

if __name__ == '__main__':
    evaluate_hr_lear()
