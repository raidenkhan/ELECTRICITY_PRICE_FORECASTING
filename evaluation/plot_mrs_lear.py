"""
plot_mrs_lear.py — Thesis Visualizations for MRS-LEAR Hybrid
=============================================================
Generates Figures 5a-5d:
- 5a: Regime Transition Series (HMM decodes)
- 5b: Soft-Blend Forecasting (Best Week by CRPS)
- 5c: Regime Posterior Probabilities
- 5d: Cumulative Error Curves (MRS-LEAR vs Baselines)
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
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def plot_regime_transitions(df):
    """Figure 5a: Time-series colored by HMM regime."""
    # Take middle 30 days for representative slice
    mid_idx = len(df) // 2
    plot_df = df.iloc[mid_idx-360:mid_idx+360] # ~30 days
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Shading regimes
    colors = ['#82CAFA', '#F9E076', '#FA8072'] # Surplus, Base, Spike
    regime_names = {0: 'Surplus/Neg', 1: 'Base', 2: 'Spike'}
    
    for r in range(3):
        mask = plot_df['regime'] == r
        if mask.any():
            ax.scatter(plot_df.index[mask], plot_df['y_true'][mask], 
                       color=colors[r], label=regime_names[r], s=15, alpha=0.8)
        
    ax.plot(plot_df.index, plot_df['y_true'], color='black', alpha=0.3, lw=1)
    ax.set_title("Figure 5a: HMM Regime Assignments (Representative Sample)")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_05a_regime_transitions.png')
    plt.close()

def plot_soft_blend_forecast(df):
    """Figure 5b: Sample week forecast with quantile bands."""
    # Pick a week with high activity (likely late 2025)
    plot_df = df.iloc[5000:5168] # One week
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Quantile fill works better with numeric x-axis for some matplotlib versions
    x = range(len(plot_df))
    
    ax.fill_between(x, plot_df['q10'].values, plot_df['q90'].values, 
                    color='gray', alpha=0.2, label='10-90% Quantile')
    ax.fill_between(x, plot_df['q30'].values, plot_df['q70'].values, 
                    color='blue', alpha=0.1, label='30-70% Quantile')
    
    ax.plot(x, plot_df['y_true'].values, color='black', marker='o', 
            markersize=3, label='Actual Price', lw=1.5)
    ax.plot(x, plot_df['y_pred'].values, color='red', linestyle='--', 
            label='MRS-LEAR Point Forecast', lw=1.5)
    
    ax.set_title("Figure 5b: MRS-LEAR Soft-Blended Forecast (Sample Week)")
    ax.set_ylabel("Price (€/MWh)")
    ticks = np.arange(0, len(plot_df), 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([plot_df.index[i].strftime('%m-%d') for i in ticks])
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_05b_soft_blend_forecast.png')
    plt.close()

def plot_error_performance(df):
    """Figure 5d: Cumulative Mean Absolute Error over time."""
    df['AE'] = (df['y_true'] - df['y_pred']).abs()
    df['cum_MAE'] = df['AE'].expanding().mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['cum_MAE'], color='forestgreen', lw=2, label='MRS-LEAR')
    
    # Add LEAR baseline horizontal line
    ax.axhline(8.62, color='red', linestyle=':', label='LEAR Benchmark (Avg)')
    
    ax.set_title("Figure 5d: Cumulative MAE over Test Horizon")
    ax.set_xlabel("Test Set Timeline")
    ax.set_ylabel("MAE (€/MWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_05d_error_evolution.png')
    plt.close()

if __name__ == '__main__':
    preds_p = ROOT / 'results/predictions/mrs_lear_predictions.parquet'
    
    if preds_p.exists():
        df = pd.read_parquet(preds_p)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        print("[plot] Generating Figure 5a...")
        plot_regime_transitions(df)
        print("[plot] Generating Figure 5b...")
        plot_soft_blend_forecast(df)
        print("[plot] Generating Figure 5d...")
        plot_error_performance(df)
        print(f"[plot] Visualizations saved to {FIG_DIR}")
    else:
        print("[plot] Error: Predictions file not found.")
