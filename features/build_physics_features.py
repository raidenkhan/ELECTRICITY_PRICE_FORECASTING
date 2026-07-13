"""
src/features/build_physics_features.py
======================================
Extends the feature matrix with physical market drivers:
1. Net Load Ramps (capture flexibility limits)
2. Fuel-Load Interactions (capture merit-order elbow shifts)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

def build_physics_features():
    print("[physics_features] Loading existing feature matrix...")
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    
    # 1. Net Load Ramps
    # Net Load = T_demand (approx) or actual demand - renewables
    # Since T_demand is already scaled, we can use it to compute the ramp.
    df['net_load_ramp'] = df['T_demand'].diff().fillna(0)
    
    # 2. Fuel-Load Interactions (SHAP-informed)
    # The merit order elbow shifts based on fuel prices and demand level.
    df['gas_load_interact'] = df['G_price_zscore'] * df['T_demand']
    
    # 3. Signed Square of Net Load (capture convexity of merit order)
    df['demand_sq'] = np.sign(df['T_demand']) * (df['T_demand']**2)

    # Save as a separate physics matrix to avoid corrupting baseline
    out_path = ROOT / 'data/processed/feature_matrix_physics.parquet'
    df.to_parquet(out_path)
    print(f"[physics_features] Saved physics-enhanced feature matrix -> {out_path}")

if __name__ == '__main__':
    build_physics_features()
