"""
bess_backtest.py — Battery Energy Storage System (BESS) Economic Evaluation
===========================================================================
Simulates a daily arbitrage strategy using point forecasts.
Compares LEAR vs. HR-LEAR profit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

# Hack for pickle
try:
    from models.hr_lear_residuals import ResidualExpert
    import __main__
    __main__.ResidualExpert = ResidualExpert
except ImportError:
    pass

def simulate_bess(y_true: np.ndarray, y_pred: np.ndarray, 
                  capacity_mwh: float = 1.0, 
                  power_mw: float = 0.5, 
                  efficiency: float = 0.9) -> float:
    """Simple daily arbitrage simulation."""
    n_days = len(y_true) // 24
    total_profit = 0.0
    
    for d in range(n_days):
        day_true = y_true[d*24 : (d+1)*24]
        day_pred = y_pred[d*24 : (d+1)*24]
        
        # Pick 2 lowest hours to charge, 2 highest to discharge (approx 1MWh)
        sorted_idx = np.argsort(day_pred)
        charge_hours = sorted_idx[:2]
        discharge_hours = sorted_idx[-2:]
        
        cost = np.sum(day_true[charge_hours] * 0.5)
        revenue = np.sum(day_true[discharge_hours] * 0.5 * efficiency)
        
        total_profit += (revenue - cost)
        
    return total_profit

def run_backtest():
    print("[bess] Starting BESS Backtest...")
    
    hr_path = ROOT / 'results/predictions/hr_lear_predictions_improved.parquet'
    if not hr_path.exists():
        hr_path = ROOT / 'results/predictions/hr_lear_predictions.parquet'
        
    df_hr = pd.read_parquet(hr_path)
    y_true = df_hr['y_true'].values
    y_hr = df_hr['y_pred'].values
    
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    
    df_feat = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    test_df = df_feat[df_feat['split'] == 'test']
    
    print("[bess] Running Global LEAR (Anchor) simulation...")
    y_lear = anchor.predict(test_df)['point']
    
    profit_lear = simulate_bess(y_true, y_lear)
    profit_hr = simulate_bess(y_true, y_hr)
    
    print(f"\n[bess] === BESS ARBITRAGE RESULTS (Test Set) ===")
    print(f"  Total Profit (Global LEAR) : â‚¬{profit_lear:,.2f}")
    print(f"  Total Profit (HR-LEAR)    : â‚¬{profit_hr:,.2f}")
    if profit_lear != 0:
        improvement = ((profit_hr - profit_lear)/profit_lear)*100
        print(f"  Improvement               : â‚¬{profit_hr - profit_lear:,.2f} ({improvement:.2f}%)")
    
    # Save results
    with open(ROOT / 'results/tables/bess_results.txt', 'w') as f:
        f.write(f"BESS Backtest Results\n")
        f.write(f"Profit LEAR: {profit_lear}\n")
        f.write(f"Profit HR  : {profit_hr}\n")

if __name__ == '__main__':
    run_backtest()
