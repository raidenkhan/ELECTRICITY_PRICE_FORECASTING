"""
plot_hr_lear.py — High-Fidelity Thesis Plots for HR-LEAR
==========================================================
Generates:
- 6a: Regime Transition Map (HMM decodes)
- 6b: HR-LEAR Soft-Blend Forecast (Sample Week + Quantiles)
- 6c: Cumulative CRPS (HR-LEAR vs Global LEAR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / 'results/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic Setup
plt.style.use('seaborn-v0_8-muted')
sns.set_palette("tab10")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def generate_hr_plots():
    preds_p = ROOT / 'results/predictions/hr_lear_predictions.parquet'
    if not preds_p.exists():
        print(f"[plot_hr] Error: {preds_p} not found.")
        return
        
    df = pd.read_parquet(preds_p)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    # 1. Figure 6a: Regime Transitions
    print("[plot_hr] Generating Figure 6a (Regime Transitions)...")
    mid_idx = len(df) // 2
    plot_df = df.iloc[mid_idx-360:mid_idx+360]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#82CAFA', '#F9E076', '#FA8072'] # Surplus, Base, Spike
    names = {0: 'Surplus', 1: 'Base', 2: 'Spike'}
    
    for r in range(3):
        mask = plot_df['regime'] == r
        if mask.any():
            ax.scatter(plot_df.index[mask], plot_df['y_true'][mask], 
                       color=colors[r], label=names[r], s=15, alpha=0.8)
    
    ax.plot(plot_df.index, plot_df['y_true'], color='black', alpha=0.2, lw=1)
    ax.set_title("Figure 6a: HR-LEAR Regime Context (Test Slice)")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_06a_hr_regimes.png')
    plt.close()

    # 2. Figure 6b: Soft-Blend Forecast (Quantiles)
    print("[plot_hr] Generating Figure 6b (Soft-Blend Forecast)...")
    # Pick a week in late 2025
    week_df = df.iloc[5000:5168]
    x = np.arange(len(week_df))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 10-90 band
    ax.fill_between(x, week_df['q10'].values, week_df['q90'].values, 
                    color='gray', alpha=0.15, label='10-90% Quantile')
    # 30-70 band
    ax.fill_between(x, week_df['q30'].values, week_df['q70'].values, 
                    color='dodgerblue', alpha=0.15, label='30-70% Quantile')
    
    ax.plot(x, week_df['y_true'].values, color='black', marker='o', 
            markersize=2, label='Actual Price', lw=1.2, alpha=0.7)
    ax.plot(x, week_df['y_pred'].values, color='red', linestyle='--', 
            label='HR-LEAR Point Forecast', lw=1.5)
    
    ax.set_title("Figure 6b: HR-LEAR Soft-Blended Forecast (Quantile Ribbons)")
    ax.set_ylabel("Price (€/MWh)")
    
    # Ticks
    ticks = np.arange(0, len(week_df), 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([week_df.index[i].strftime('%m-%d') for i in ticks])
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_06b_hr_softblend.png')
    plt.close()

    # 3. Figure 6c: Cumulative CRPS
    print("[plot_hr] Generating Figure 6c (Cumulative Performance)...")
    df['AE'] = (df['y_true'] - df['y_pred']).abs()
    df['cum_MAE'] = df['AE'].expanding().mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['cum_MAE'], color='forestgreen', lw=2, label='HR-LEAR MAE')
    ax.axhline(9.38, color='red', linestyle=':', label='Global LEAR Baseline (Overall)')
    
    ax.set_title("Figure 6c: HR-LEAR Error Convergence")
    ax.set_xlabel("Test Set Timeline")
    ax.set_ylabel("MAE (€/MWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_06c_hr_convergence.png')
    plt.close()
    
    print(f"[plot_hr] Visualizations saved to {FIG_DIR}")

if __name__ == '__main__':
    generate_hr_plots()
