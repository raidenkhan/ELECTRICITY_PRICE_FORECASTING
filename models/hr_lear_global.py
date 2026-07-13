"""
hr_lear_global.py — HR-LEAR Step 1: Global Anchor
===================================================
Trains a standard LEAR baseline on the full training set 
and extracts residuals for subsequent hierarchical refinement.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.baselines.lear import LEARBaseline

def train_global_anchor():
    print("[hr_lear_global] Loading feature matrix...")
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"[hr_lear_global] Training Global Anchor on {len(train_df)} rows...")
    anchor = LEARBaseline()
    anchor.fit(train_df)
    
    # Save the anchor model
    models_dir = ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / 'hr_lear_anchor.pkl', 'wb') as f:
        pickle.dump(anchor, f)
    print(f"[hr_lear_global] Saved Global Anchor to {models_dir / 'hr_lear_anchor.pkl'}")
    
    # Extract Point Forecasts for all splits
    print("[hr_lear_global] Generating anchor forecasts for all splits...")
    full_out = anchor.predict(df, n_bootstrap=1) # Just need point forecast for residuals
    
    results_df = df[['price', 'split']].copy()
    results_df['anchor_pred'] = full_out['point']
    results_df['residual'] = results_df['price'] - results_df['anchor_pred']
    
    # Save residuals for Level 1 training
    residuals_path = ROOT / 'data/processed/hr_lear_residuals.parquet'
    results_df.to_parquet(residuals_path)
    print(f"[hr_lear_global] Saved anchor forecasts and residuals to {residuals_path}")
    
    # Quick metrics to verify anchor sanity
    test_results = results_df[results_df['split'] == 'test']
    mae = (test_results['price'] - test_results['anchor_pred']).abs().mean()
    print(f"[hr_lear_global] Anchor Test MAE: {mae:.4f}")

if __name__ == '__main__':
    train_global_anchor()
