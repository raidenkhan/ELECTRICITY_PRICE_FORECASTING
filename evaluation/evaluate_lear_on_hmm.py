"""
evaluate_lear_on_hmm.py — Fair Comparison Runner
==================================================
Runs the standard LEAR baseline (trained on all data) 
and evaluates it on the HMM-detected regimes.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.baselines.lear import LEARBaseline
from models.mrs_lear_ensemble import crps_from_quantiles

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

def run_fair_comparison():
    print("[fair_comp] Loading data...")
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    df = df.join(labels[['hmm_regime']], how='left')

    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print("[fair_comp] Training standard LEAR baseline (Global)...")
    lear = LEARBaseline()
    lear.fit(train_df)

    print("[fair_comp] Predicting on test set...")
    out = lear.predict(test_df, n_bootstrap=500)
    
    y_true = test_df['price'].values
    y_pred = out['point']
    y_quantiles = out['quantiles']
    regimes = test_df['hmm_regime'].fillna(1).astype(int).values

    # Compute Overall CRPS
    crps_all = crps_from_quantiles(y_true, y_quantiles, QUANTILE_LEVELS)
    print(f"\n[fair_comp] Global LEAR Overall CRPS: {crps_all:.4f}")

    # Compute Regime-Stratified CRPS
    regime_results = []
    for k in range(3):
        mask = regimes == k
        if mask.any():
            rc = crps_from_quantiles(y_true[mask], y_quantiles[mask], QUANTILE_LEVELS)
            regime_results.append({'regime': k, 'LEAR_CRPS': rc, 'count': mask.sum()})
            print(f"  Regime {k}: CRPS={rc:.4f}  (n={mask.sum()})")

    # Load MRS-LEAR results for direct comparison
    mrs_results_path = ROOT / 'results/tables/mrs_lear_results.csv'
    if mrs_results_path.exists():
        mrs_df = pd.read_csv(mrs_results_path)
        print("\n[fair_comp] Head-to-Head Comparison (CRPS):")
        print(f"{'Regime':<10} | {'Global LEAR':<12} | {'MRS-LEAR':<12} | {'Winner'}")
        print("-" * 50)
        
        for k in range(3):
            lear_v = next((r['LEAR_CRPS'] for r in regime_results if r['regime'] == k), np.nan)
            mrs_v = mrs_df[f'CRPS_regime_{k}'].values[0] if f'CRPS_regime_{k}' in mrs_df.columns else np.nan
            
            winner = "MRS-LEAR" if mrs_v < lear_v else "LEAR"
            print(f"{k:<10} | {lear_v:<12.4f} | {mrs_v:<12.4f} | {winner}")

if __name__ == '__main__':
    run_fair_comparison()
