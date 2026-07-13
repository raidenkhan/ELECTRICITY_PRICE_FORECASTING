"""
plot_comparison.py — Head-to-Head Visual Comparison
=========================================================
Plots the predictions of Global LEAR, HR-LEAR, and CNN-BiLSTM-AR 
against the actual price for targeted 7-day windows.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as torch
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from data.sequence_dataset import get_dataloaders
from models.cnn_bilstm import CNNBiLSTM_AR

# Aesthetic Setup
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def plot_window(hr_df, anchor_test_df, df, model, scaler, device, start_idx, end_idx, title, out_filename):
    window_df = hr_df.iloc[start_idx:end_idx].copy()
    dates = window_df.index
    
    # Get anchor predictions
    window_anchor = anchor_test_df.loc[dates]['anchor_pred'].values
    
    # Generated DL predictions for exact dates
    dl_window_preds = []
    features_scaled = scaler.transform(df.select_dtypes(include=[np.number]).drop(columns=['price'], errors='ignore'))
    
    with torch.no_grad():
        for dt in dates:
            df_idx = df.index.get_loc(dt)
            x_window = features_scaled[df_idx - 168 : df_idx]
            x_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x_tensor).cpu().numpy()[0, 3]
            dl_window_preds.append(pred)
            
    window_dl = np.array(dl_window_preds)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(168)
    
    ax.plot(x, window_df['y_true'].values, color='black', lw=2.5, alpha=1.0, label='Actual Price', zorder=5)
    ax.plot(x, window_anchor, color='mediumseagreen', lw=2.0, linestyle=':', alpha=0.9, label='Global LEAR', zorder=2)
    ax.plot(x, window_dl, color='darkviolet', lw=2.0, linestyle='--', alpha=0.7, label='CNN-BiLSTM-AR', zorder=3)
    ax.plot(x, window_df['y_pred'].values, color='crimson', lw=2.5, linestyle='-', alpha=0.85, label='HR-LEAR (Ours)', zorder=4)
    
    ax.set_title(f"{title}: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}", fontsize=14, pad=15)
    ax.set_ylabel("Price (€/MWh)", fontsize=12)
    
    # X-ticks every 24 hours
    ticks = np.arange(0, 168, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i].strftime('%a\n%m-%d') for i in ticks], fontsize=10)
    
    # Add Metrics Text Box (Overall Test Set Metrics from our previous runs)
    metrics_text = (
        "Overall Test Set Performance:\n\n"
        "HR-LEAR (Ours):\n"
        "  CRPS = 7.95 | MAE = 9.92\n\n"
        "Global LEAR:\n"
        "  CRPS = 9.38 | MAE = 11.46\n\n"
        "CNN-BiLSTM-AR:\n"
        "  CRPS = 10.06 | MAE = 12.62"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, zorder=6, fontfamily='monospace')
    
    ax.legend(loc='upper right', frameon=True, fontsize=11, title="Models")
    ax.grid(True, alpha=0.4, linestyle='--')
    
    out_path = ROOT / f'results/figures/{out_filename}'
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot_comp] Saved comparison plot to {out_path}")


def generate_comparison_plots():
    print("[plot_comp] Loading data...")
    hr_df = pd.read_parquet(ROOT / 'results/predictions/hr_lear_predictions.parquet')
    anchor_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    anchor_test_df = anchor_df[anchor_df['split'] == 'test'].copy()
    
    hr_df = hr_df.set_index('datetime')
    
    # Feature matrix for DL model
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    num_features = df.select_dtypes(include=[np.number]).drop(columns=['price'], errors='ignore').shape[1]
    
    loaders, scaler = get_dataloaders(df, window_size=168, batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM_AR(input_dim=num_features, n_quantiles=7).to(device)
    model.load_state_dict(torch.load(ROOT / 'models/cnn_bilstm_best.pt', map_location=device))
    model.eval()

    # --- Plot 1: Standard Max Variance ---
    hr_df['rolling_var'] = hr_df['y_true'].rolling(168).var()
    var_peak_idx = hr_df['rolling_var'].argmax()
    var_start_idx = max(0, var_peak_idx - 168)
    plot_window(hr_df, anchor_test_df, df, model, scaler, device, 
                var_start_idx, var_start_idx + 168, 
                "High-Volatility Week Comparison (Test Set)", 
                "fig_07_model_comparison.png")

    # --- Plot 2: Extreme Swing (Negative to Positive) ---
    print("[plot_comp] Searching for extreme negative-to-positive swing...")
    hr_df['rolling_min'] = hr_df['y_true'].rolling(168).min()
    hr_df['rolling_max'] = hr_df['y_true'].rolling(168).max()
    hr_df['swing'] = hr_df['rolling_max'] - hr_df['rolling_min']
    
    # Filter for windows that go deeply negative
    valid_windows_mask = hr_df['rolling_min'] < -30
    if valid_windows_mask.any():
        valid_windows = hr_df[valid_windows_mask]
        swing_peak_dt = valid_windows['swing'].idxmax()
        swing_end_idx = hr_df.index.get_loc(swing_peak_dt)
        swing_start_idx = max(0, swing_end_idx - 168)
        
        plot_window(hr_df, anchor_test_df, df, model, scaler, device, 
                    swing_start_idx, swing_start_idx + 168, 
                    "Extreme Regime Transition (Negative to Spike)", 
                    "fig_08_extreme_volatility.png")
    else:
        print("[plot_comp] No deep negative window found. Falling back to max swing.")
        swing_peak_dt = hr_df['swing'].idxmax()
        swing_end_idx = hr_df.index.get_loc(swing_peak_dt)
        swing_start_idx = max(0, swing_end_idx - 168)
        
        plot_window(hr_df, anchor_test_df, df, model, scaler, device, 
                    swing_start_idx, swing_start_idx + 168, 
                    "Maximum Price Swing Week (Test Set)", 
                    "fig_08_extreme_volatility.png")


if __name__ == '__main__':
    generate_comparison_plots()
