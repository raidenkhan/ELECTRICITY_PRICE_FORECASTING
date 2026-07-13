"""
src/evaluation/extended_ablations.py
====================================
Performs deeper ablations and metrics requested by the user:
1. Regime-Specific Metrics (MAE, CRPS)
2. Directional Accuracy (DA)
3. Spike Tracking (Top 5% metrics)
4. Trading Backtest (Profit per MWh)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.mrs_lear_ensemble import crps_from_quantiles

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

def run_extended_eval():
    preds_path = ROOT / 'results/predictions/hr_lear_predictions.parquet'
    if not preds_path.exists():
        print("Predictions not found. Run the HR-LEAR pipeline first.")
        return

    df = pd.read_parquet(preds_path)
    
    # 1. Basic Metrics
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    quantiles = df[[f'q{int(q*100):02d}' for q in QUANTILE_LEVELS]].values
    
    # 2. Directional Accuracy (DA)
    # Price change direction compared to previous hour
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    da = np.mean((np.sign(y_true_diff) == np.sign(y_pred_diff)))
    
    # 3. Spike Evaluation (Top 5%)
    threshold = np.quantile(y_true, 0.95)
    spike_mask = y_true >= threshold
    
    spike_mae = np.mean(np.abs(y_true[spike_mask] - y_pred[spike_mask]))
    spike_crps = crps_from_quantiles(y_true[spike_mask], quantiles[spike_mask], QUANTILE_LEVELS)
    
    # 4. Regime Specifics
    regime_metrics = []
    for r in range(3):
        mask = df['regime'] == r
        if not mask.any(): continue
        
        r_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        r_crps = crps_from_quantiles(y_true[mask], quantiles[mask], QUANTILE_LEVELS)
        regime_metrics.append({
            'regime': r,
            'MAE': r_mae,
            'CRPS': r_crps,
            'count': mask.sum()
        })
    
    # 5. Trading Backtest (Simple Arbitrage)
    # Buy at min price of day, sell at max price
    # Simplified: hourly profitability
    # Assume 1MW battery, 90% efficiency
    # If pred says price(t+k) > price(t)/0.9, we buy at t and sell at t+k
    # We'll use a daily cycle approach like in the paper
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    daily_profits = []
    for date, group in df.groupby('date'):
        if len(group) < 24: continue
        
        # Point forecast based decisions
        idx_buy = group['y_pred'].idxmin()
        idx_sell = group['y_pred'].idxmax()
        
        p_buy = group.loc[idx_buy, 'y_true']
        p_sell = group.loc[idx_sell, 'y_true']
        
        # Profit = 0.9 * p_sell - (1/0.9) * p_buy
        profit = 0.9 * p_sell - (1.111) * p_buy
        daily_profits.append(profit)
    
    avg_profit = np.mean(daily_profits)

    # Output Results
    print("\n" + "="*40)
    print("      EXTENDED RESEARCH METRICS")
    print("="*40)
    print(f"Directional Accuracy: {da:.2%}")
    print(f"Spike MAE (Top 5%):   {spike_mae:.2f}")
    print(f"Spike CRPS (Top 5%):  {spike_crps:.2f}")
    print(f"Avg Daily Trading Profit: €{avg_profit:.2f}/MW")
    print("-" * 40)
    for rm in regime_metrics:
        print(f"Regime {rm['regime']}: MAE={rm['MAE']:.2f}, CRPS={rm['CRPS']:.2f} (n={rm['count']})")
    print("="*40)

    # Save to CSV
    results_df = pd.DataFrame([{
        'DirectionalAccuracy': da,
        'SpikeMAE': spike_mae,
        'SpikeCRPS': spike_crps,
        'AvgTradingProfit': avg_profit,
        **{f"Regime_{rm['regime']}_MAE": rm['MAE'] for rm in regime_metrics},
        **{f"Regime_{rm['regime']}_CRPS": rm['CRPS'] for rm in regime_metrics}
    }])
    out_file = ROOT / 'results/tables/extended_ablations.csv'
    results_df.to_csv(out_file, index=False)
    print(f"Saved extended metrics to {out_file}")

if __name__ == '__main__':
    run_extended_eval()
