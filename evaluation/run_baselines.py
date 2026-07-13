"""
run_baselines.py - Unified Evaluation Runner for Baselines

Runs all 4 baselines on the test set.
Computes MAE, RMSE, CRPS, Pinball (q10, q90), and Winkler Score.
Computes regime-stratified MAE and CRPS.
Saves results to baseline_results.csv and produces fig_03a_baseline_comparison.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.baselines.naive import NaivePersistence
from src.models.baselines.lear import LEARBaseline
from src.models.baselines.sarima_garch import SarimaGarchBaseline
from src.models.baselines.tft_standard import TFTStandardBaseline
from src.evaluation.metrics import crps_from_quantiles

def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2) -> float:
    """Winkler score for (1-alpha)*100% prediction interval."""
    # alpha = 0.2 for 80% PI (q10, q90)
    delta = upper - lower
    penalty_lower = (2.0 / alpha) * (lower - y) * (y < lower)
    penalty_upper = (2.0 / alpha) * (y - upper) * (y > upper)
    winkler = delta + penalty_lower + penalty_upper
    return np.mean(winkler)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray, q_levels: list[float], regime_labels: np.ndarray):
    """
    Computes unified metrics block.
    quantiles shape: [N, 7]
    q_levels: [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    """
    import torch
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Setup for torch metrics
    y_t = torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1) # [N, 1]
    q_t = torch.tensor(quantiles, dtype=torch.float32).unsqueeze(1) # [N, 1, 7]
    
    crps = crps_from_quantiles(y_t, q_t, q_levels)
    
    # Pinball at q10 and q90
    levels = np.array(q_levels)
    idx_10 = np.where(np.isclose(levels, 0.1))[0][0]
    idx_90 = np.where(np.isclose(levels, 0.9))[0][0]
    
    # manually calculate pinball for numpy
    def pb(y, q_val, tau):
        err = y - q_val
        return np.mean(np.maximum(tau * err, (tau - 1) * err))
        
    pb_10 = pb(y_true, quantiles[:, idx_10], 0.1)
    pb_90 = pb(y_true, quantiles[:, idx_90], 0.9)
    
    # Winkler 80% interval (alpha=0.2)
    winkler = winkler_score(y_true, quantiles[:, idx_10], quantiles[:, idx_90], alpha=0.2)
    
    # Stratified
    regimes = np.unique(regime_labels)
    strat_mae = {}
    strat_crps = {}
    for r in range(4):
        mask = (regime_labels == r)
        if mask.sum() > 0:
            strat_mae[f"regime_{r}"] = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            strat_crps[f"regime_{r}"] = crps_from_quantiles(y_t[mask], q_t[mask], q_levels)
        else:
            strat_mae[f"regime_{r}"] = np.nan
            strat_crps[f"regime_{r}"] = np.nan
            
    return {
        'MAE': mae,
        'RMSE': rmse,
        'CRPS': crps,
        'Pinball_10': pb_10,
        'Pinball_90': pb_90,
        'Winkler_80': winkler,
        **( {f'MAE_regime_{r}': strat_mae[f'regime_{r}'] for r in range(4)} ),
        **( {f'CRPS_regime_{r}': strat_crps[f'regime_{r}'] for r in range(4)} )
    }

def main():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
        
    fm_path = ROOT / cfg["data"]["processed_path"]
    rl_path = ROOT / cfg["data"]["regimes_dir"] / "regime_labels.parquet"
    
    print("[evaluation] Loading dataset for baselines ...")
    fm = pd.read_parquet(fm_path)
    rl = pd.read_parquet(rl_path)
    
    train_mask = fm["split"] == "train"
    test_mask = fm["split"] == "test"
    val_mask = fm["split"] == "val"
    
    train_df = fm[train_mask]
    test_df = fm[test_mask]
    val_df = fm[val_mask]
    
    # Baselines often require a contiguous history: use val as part of train for simplicity?
    # Actually, we should combine train + val for classical models fitting right before the test period.
    train_val_df = pd.concat([train_df, val_df])
    
    test_regimes = rl[rl["split"] == "test"]["regime"].values
    y_test = test_df['price'].values
    
    models = {
        "Naive D-7": NaivePersistence(),
        "LEAR": LEARBaseline(),
        "SARIMA-GARCH": SarimaGarchBaseline(),
        "Standard TFT": TFTStandardBaseline(seq_len=168, horizon=24)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n[evaluation] --------------------")
        print(f"[evaluation] Running Baseline: {name}")
        
        # Fit
        if name == "Standard TFT":
            model.fit(train_df, val_df)
        else:
            model.fit(train_val_df)
            
        # Predict
        print(f"[evaluation] Generating test predictions ...")
        preds = model.predict(test_df)
        
        y_pred = preds['point']
        quantiles = preds['quantiles']
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, quantiles, model.quantile_levels, test_regimes)
        metrics["Model"] = name
        results.append(metrics)
        print(f"[evaluation] {name} MAE: {metrics['MAE']:.3f} | CRPS: {metrics['CRPS']:.3f}")
        
    # Save CSV
    df_res = pd.DataFrame(results)
    # Reorder columns
    cols = ['Model', 'MAE', 'RMSE', 'CRPS', 'Pinball_10', 'Pinball_90', 'Winkler_80'] + \
           [f'MAE_regime_{r}' for r in range(4)] + \
           [f'CRPS_regime_{r}' for r in range(4)]
    df_res = df_res[[c for c in cols if c in df_res.columns]]
    
    out_csv = ROOT / "results/tables/baseline_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\n[evaluation] Baseline testing complete. Saved results -> {out_csv}")
    
    # Figure 3a: Grouped Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regime_names = ["Low-vol Off-peak", "Thermal Peak", "Renewable Surplus", "Extreme Spike"]
    
    x = np.arange(4)
    width = 0.15
    multiplier = 0
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, row in df_res.iterrows():
        model_name = row['Model']
        crps_vals = [row[f'CRPS_regime_{r}'] for r in range(4)]
        
        offset = width * multiplier
        rects = ax.bar(x + offset, crps_vals, width, label=model_name)
        multiplier += 1
        
    ax.set_ylabel('CRPS')
    ax.set_title('Figure 3a - CRPS by Model and Regime (Test Set)', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(regime_names)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    out_pdf = ROOT / "results/figures/fig_03a_baseline_comparison.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)
    print(f"[evaluation] Figure 3a saved -> {out_pdf}")

if __name__ == "__main__":
    main()
