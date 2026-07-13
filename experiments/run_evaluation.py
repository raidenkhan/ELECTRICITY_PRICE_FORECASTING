#!/usr/bin/env python
# =============================================================================
# run_evaluation.py — LF-GP-NRF evaluation runner
# Electricity Price Forecasting — MPhil Research
#
# Usage:
#   python src/experiments/run_evaluation.py [--checkpoint PATH] [--split test|val]
#
# What this script does:
#   1. Loads the trained LF-GP-NRF model from the Phase 3 checkpoint
#   2. Runs the EPFEvaluator on the test set (2024) or validation set (2023)
#   3. Computes all metrics and saves results (CSV + LaTeX)
#   4. Generates all diagnostic plots
#   5. Runs three targeted tests:
#        - Spike anticipation       (does uncertainty grow BEFORE spikes?)
#        - Regime transition smoothness (are consecutive-hour distributions smooth?)
#        - Negative price CRPS      (does the flow handle negative prices well?)
# =============================================================================

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import
import matplotlib.gridspec as gridspec  # noqa: F401 — available for plots
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the working directory from
# which this script is invoked.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset import build_dataloaders
from src.experiments.evaluation import (
    EPFEvaluator,
    compare_models,
    compute_crps_samples,
    compute_energy_score,
    compute_pinball,
    compute_winkler_score,
    dm_test,
)
from src.models.lf_gp_nrf.model import LFGPNRFModel

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
_PALETTE = sns.color_palette("muted")

# ---------------------------------------------------------------------------
# Fallback feature dimensions used when feature_dims.json is absent
# ---------------------------------------------------------------------------
_DEFAULT_HISTORY_FEAT_DIM = 20
_DEFAULT_FUTURE_FEAT_DIM = 6


# =============================================================================
# 1.  Model loader
# =============================================================================


def load_model(
    checkpoint_path: str,
    feature_dims_path: str,
    device: str,
) -> LFGPNRFModel:
    """Instantiate LFGPNRFModel and load a saved state dict.

    Parameters
    ----------
    checkpoint_path :
        Path to the ``.pt`` checkpoint file written by the training script.
        The file must contain a mapping loadable by ``torch.load`` that is
        compatible with ``model.load_state_dict``.  Both raw state-dict files
        and PyTorch-Lightning checkpoint dicts (``{'state_dict': ...}``) are
        handled automatically.
    feature_dims_path :
        Path to ``feature_dims.json`` — a small JSON file written by the
        training pipeline that records ``history_feat_dim`` and
        ``future_feat_dim`` so the architecture matches exactly.
    device :
        Torch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    LFGPNRFModel
        Model loaded from the checkpoint, moved to *device* and set to eval
        mode.
    """
    # ------------------------------------------------------------------
    # A.  Read architecture dimensions from feature_dims.json
    # ------------------------------------------------------------------
    if os.path.isfile(feature_dims_path):
        with open(feature_dims_path, "r", encoding="utf-8") as fh:
            dims = json.load(fh)
        history_feat_dim = int(dims.get("history_feat_dim", _DEFAULT_HISTORY_FEAT_DIM))
        future_feat_dim = int(dims.get("future_feat_dim", _DEFAULT_FUTURE_FEAT_DIM))
        print(
            f"[load_model] feature_dims.json found → "
            f"history_feat_dim={history_feat_dim}, future_feat_dim={future_feat_dim}"
        )
    else:
        history_feat_dim = _DEFAULT_HISTORY_FEAT_DIM
        future_feat_dim = _DEFAULT_FUTURE_FEAT_DIM
        print(
            f"[load_model] WARNING: '{feature_dims_path}' not found. "
            f"Falling back to defaults: "
            f"history_feat_dim={history_feat_dim}, future_feat_dim={future_feat_dim}"
        )

    # ------------------------------------------------------------------
    # B.  Build the model with standard Phase-3 hyperparameters
    # ------------------------------------------------------------------
    model = LFGPNRFModel(
        history_feat_dim=history_feat_dim,
        future_feat_dim=future_feat_dim,
        latent_dim=3,
        encoder_hidden=128,
        sde_hidden=64,
        n_inducing=256,
        n_flow_transforms=4,
        n_flow_bins=8,
        n_paths_train=8,
        n_paths_infer=64,
        n_samples_infer=100,
    )

    # ------------------------------------------------------------------
    # C.  Load the saved weights
    # ------------------------------------------------------------------
    raw = torch.load(checkpoint_path, map_location=device)

    # Support both:
    #   (a) plain state-dict saved by torch.save(model.state_dict(), ...)
    #   (b) Lightning checkpoint: {'state_dict': ..., 'epoch': ..., ...}
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[load_model] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device)
    model.eval()
    return model


# =============================================================================
# 2.  Targeted test — spike anticipation
# =============================================================================


def run_targeted_test_spike_anticipation(
    results: dict,
    save_dir: str,
) -> dict:
    """Spike anticipation test — does uncertainty grow BEFORE spikes?

    A "spike window" is a 24-hour day where at least one hour has a price
    above 200 EUR/MWh.  We compare the mean 90th-percentile forecast profile
    (across hours 0–23) for spike windows vs. non-spike (base) windows.

    Parameters
    ----------
    results :
        Dict returned by ``EPFEvaluator.run()``.  Must contain:
        ``'samples'`` (N, S, 24) and ``'y_true'`` (N, 24).
    save_dir :
        Directory in which to write ``spike_anticipation.png``.

    Returns
    -------
    dict with keys:
        ``'n_spike_windows'``  : int
        ``'mean_p90_spike'``   : np.ndarray (24,)
        ``'mean_p90_base'``    : np.ndarray (24,)
    """
    os.makedirs(save_dir, exist_ok=True)

    samples = np.asarray(results["samples"], dtype=np.float64)  # (N, S, 24)
    y_true = np.asarray(results["y_true"], dtype=np.float64)  # (N, 24)

    N, S, H = samples.shape

    # ------------------------------------------------------------------
    # Identify spike / base windows
    # ------------------------------------------------------------------
    spike_mask = np.any(y_true > 200.0, axis=1)  # (N,) bool
    base_mask = ~spike_mask

    n_spike = int(spike_mask.sum())
    n_base = int(base_mask.sum())

    print(
        f"[spike_anticipation] spike windows: {n_spike} / {N}  ({100 * n_spike / N:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Compute 90th-percentile forecast at each hour for each window
    # p90_all: (N, 24)
    # ------------------------------------------------------------------
    # samples shape (N, S, 24) → we want quantile over S axis
    p90_all = np.quantile(samples, 0.90, axis=1)  # (N, 24)

    # ------------------------------------------------------------------
    # Mean & std of P90 trajectory
    # ------------------------------------------------------------------
    if n_spike > 0:
        p90_spike = p90_all[spike_mask]  # (n_spike, 24)
        mean_p90_spike = p90_spike.mean(axis=0)
        std_p90_spike = p90_spike.std(axis=0)
    else:
        # Degenerate case — no spikes in the evaluated period
        mean_p90_spike = np.full(H, np.nan)
        std_p90_spike = np.full(H, np.nan)
        print("[spike_anticipation] WARNING: no spike windows found.")

    if n_base > 0:
        p90_base = p90_all[base_mask]  # (n_base, 24)
        mean_p90_base = p90_base.mean(axis=0)
        std_p90_base = p90_base.std(axis=0)
    else:
        mean_p90_base = np.full(H, np.nan)
        std_p90_base = np.full(H, np.nan)

    # ------------------------------------------------------------------
    # Identify where spikes typically occur (most common spike hour)
    # ------------------------------------------------------------------
    if n_spike > 0:
        spike_hours = y_true[spike_mask] > 200.0  # (n_spike, 24)
        spike_hour_counts = spike_hours.sum(axis=0)  # (24,)
        peak_spike_hour = int(np.argmax(spike_hour_counts))
    else:
        peak_spike_hour = None

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    hours = np.arange(H)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Spike windows
    ax.plot(
        hours,
        mean_p90_spike,
        color=_PALETTE[3],
        lw=2.2,
        label=f"Spike windows (n={n_spike})",
        zorder=4,
    )
    ax.fill_between(
        hours,
        mean_p90_spike - std_p90_spike,
        mean_p90_spike + std_p90_spike,
        color=_PALETTE[3],
        alpha=0.20,
        label="±1 std (spike)",
    )

    # Base windows
    ax.plot(
        hours,
        mean_p90_base,
        color=_PALETTE[0],
        lw=2.2,
        label=f"Base windows (n={n_base})",
        zorder=4,
    )
    ax.fill_between(
        hours,
        mean_p90_base - std_p90_base,
        mean_p90_base + std_p90_base,
        color=_PALETTE[0],
        alpha=0.20,
        label="±1 std (base)",
    )

    # Annotate peak spike hour
    if peak_spike_hour is not None:
        ax.axvline(
            peak_spike_hour,
            color="k",
            ls="--",
            lw=1.3,
            alpha=0.6,
            label=f"Peak spike hour = {peak_spike_hour}",
        )

    ax.set_xlabel("Hour within window (0 = midnight)", fontsize=11)
    ax.set_ylabel("90th-percentile forecast (EUR/MWh)", fontsize=11)
    ax.set_title(
        "LF-GP-NRF — Spike Anticipation: 90th-Percentile Profile",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(hours)
    ax.set_xticklabels([str(h) for h in hours], fontsize=8)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "spike_anticipation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[spike_anticipation] Saved → {out_path}")

    return {
        "n_spike_windows": n_spike,
        "mean_p90_spike": mean_p90_spike,
        "mean_p90_base": mean_p90_base,
    }


# =============================================================================
# 3.  Targeted test — regime transition smoothness
# =============================================================================


def run_targeted_test_regime_smoothness(
    results: dict,
    save_dir: str,
) -> dict:
    """Regime transition smoothness test.

    For every test window (day) and every consecutive-hour pair (t → t+1),
    compute the Wasserstein-1 distance between the predictive distributions
    at hour *t* and hour *t+1* using the sorted-quantile (Monge) trick::

        W1(F_t, F_{t+1}) ≈ mean(|sorted_samples_t - sorted_samples_{t+1}|)

    This gives a (N, 23) matrix of W1 distances.  We then report:
    - mean W1 by hour-transition index (0→1, 1→2, …, 22→23)
    - std  W1 by hour-transition index (lower std = smoother across days)

    Parameters
    ----------
    results :
        Dict from ``EPFEvaluator.run()``.  Must contain ``'samples'`` (N, S, 24).
    save_dir :
        Directory for ``regime_smoothness.png``.

    Returns
    -------
    dict with keys:
        ``'mean_w1_by_hour'`` : np.ndarray (23,)
        ``'std_w1_by_hour'``  : np.ndarray (23,)
    """
    os.makedirs(save_dir, exist_ok=True)

    samples = np.asarray(results["samples"], dtype=np.float64)  # (N, S, 24)
    N, S, H = samples.shape

    # Sort samples along the S axis once for efficiency
    # sorted_samples: (N, S, 24)
    sorted_samples = np.sort(samples, axis=1)

    # ------------------------------------------------------------------
    # W1 distances — shape (N, 23)
    # Using the exact formula for 1D Wasserstein-1 between two empirical
    # distributions with equal weight on each sample:
    #   W1 = (1/S) * sum_k |X_{(k),t} - X_{(k),t+1}|
    # ------------------------------------------------------------------
    w1 = np.mean(
        np.abs(sorted_samples[:, :, :-1] - sorted_samples[:, :, 1:]),
        axis=1,
    )  # (N, 23)

    mean_w1 = w1.mean(axis=0)  # (23,)
    std_w1 = w1.std(axis=0)  # (23,)

    # ------------------------------------------------------------------
    # Plot: mean ± std W1 distance by hour-of-day transition
    # ------------------------------------------------------------------
    transitions = np.arange(H - 1)  # 0 … 22  representing h→h+1
    tick_labels = [f"{h}→{h + 1}" for h in range(H - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart of mean W1 with std error bars
    ax0 = axes[0]
    ax0.bar(
        transitions,
        mean_w1,
        color=_PALETTE[0],
        edgecolor="white",
        linewidth=0.5,
        label="Mean W1",
        zorder=2,
    )
    ax0.errorbar(
        transitions,
        mean_w1,
        yerr=std_w1,
        fmt="none",
        color="k",
        elinewidth=1.0,
        capsize=3,
        alpha=0.6,
        label="±1 std",
        zorder=3,
    )
    overall_mean = float(mean_w1.mean())
    ax0.axhline(
        overall_mean,
        color=_PALETTE[3],
        ls="--",
        lw=1.4,
        label=f"Overall mean = {overall_mean:.2f}",
    )
    ax0.set_xticks(transitions)
    ax0.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax0.set_xlabel("Hour-of-day transition", fontsize=10)
    ax0.set_ylabel("Wasserstein-1 distance (EUR/MWh)", fontsize=10)
    ax0.set_title("Mean W1 by Transition (± std)", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=9)

    # Right: std W1 — lower = smoother across days
    ax1 = axes[1]
    ax1.bar(
        transitions,
        std_w1,
        color=_PALETTE[1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xticks(transitions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_xlabel("Hour-of-day transition", fontsize=10)
    ax1.set_ylabel("Std of W1 across days (EUR/MWh)", fontsize=10)
    ax1.set_title(
        "Distribution Smoothness (lower = smoother)", fontsize=11, fontweight="bold"
    )

    fig.suptitle(
        "LF-GP-NRF — Regime Transition Smoothness",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(save_dir, "regime_smoothness.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regime_smoothness] Saved → {out_path}")

    return {
        "mean_w1_by_hour": mean_w1,
        "std_w1_by_hour": std_w1,
    }


# =============================================================================
# 4.  Targeted test — negative price CRPS
# =============================================================================


def run_targeted_test_negative_prices(
    results: dict,
    save_dir: str,
) -> dict:
    """Negative price CRPS test — does the flow handle negative prices well?

    Segments all (window, hour) pairs by price regime:
    - Negative:  y_true < 0
    - Base:      0 ≤ y_true ≤ 50
    - Spike:     y_true > 200

    Computes per-observation CRPS in each regime and visualises the CRPS
    distributions as violin plots.

    Parameters
    ----------
    results :
        Dict from ``EPFEvaluator.run()``.  Must contain:
        ``'samples'`` (N, S, 24) and ``'y_true'`` (N, 24).
    save_dir :
        Directory for ``negative_price_crps.png``.

    Returns
    -------
    dict with keys:
        ``'crps_negative'`` : float  (mean CRPS for y < 0)
        ``'crps_base'``     : float  (mean CRPS for 0 ≤ y ≤ 50)
        ``'crps_spike'``    : float  (mean CRPS for y > 200)
        ``'n_negative'``    : int
        ``'n_spike'``       : int
    """
    os.makedirs(save_dir, exist_ok=True)

    samples = np.asarray(results["samples"], dtype=np.float64)  # (N, S, 24)
    y_true = np.asarray(results["y_true"], dtype=np.float64)  # (N, 24)

    N, S, H = samples.shape

    # ------------------------------------------------------------------
    # Flatten to per-observation arrays
    # y_flat:       (N*H,)
    # samples_flat: (N*H, S)
    # ------------------------------------------------------------------
    y_flat = y_true.reshape(-1)  # (N*H,)
    samples_flat = samples.transpose(0, 2, 1).reshape(N * H, S)
    # After transpose: (N, H, S) → reshape → (N*H, S)

    # Per-observation CRPS
    crps_all = compute_crps_samples(y_flat, samples_flat)  # (N*H,)

    # ------------------------------------------------------------------
    # Regime masks
    # ------------------------------------------------------------------
    mask_neg = y_flat < 0.0
    mask_base = (y_flat >= 0.0) & (y_flat <= 50.0)
    mask_spike = y_flat > 200.0

    crps_neg = crps_all[mask_neg]
    crps_base = crps_all[mask_base]
    crps_spike = crps_all[mask_spike]

    mean_crps_neg = float(np.mean(crps_neg)) if mask_neg.any() else float("nan")
    mean_crps_base = float(np.mean(crps_base)) if mask_base.any() else float("nan")
    mean_crps_spike = float(np.mean(crps_spike)) if mask_spike.any() else float("nan")

    n_neg = int(mask_neg.sum())
    n_spike = int(mask_spike.sum())

    print(
        f"[negative_prices] n_negative={n_neg}, n_base={int(mask_base.sum())}, "
        f"n_spike={n_spike}"
    )

    # ------------------------------------------------------------------
    # Build a tidy DataFrame for seaborn violin plot
    # Clip extreme outlier CRPS values at the 99th percentile for
    # readability — they arise from gross misses on extreme spikes.
    # ------------------------------------------------------------------
    clip_p99 = np.nanpercentile(crps_all, 99)

    regime_data = []
    for regime_label, crps_vals in [
        ("Negative\n(y < 0)", crps_neg),
        ("Base\n(0 ≤ y ≤ 50)", crps_base),
        ("Spike\n(y > 200)", crps_spike),
    ]:
        if len(crps_vals) > 0:
            clipped = np.clip(crps_vals, 0.0, clip_p99)
            for v in clipped:
                regime_data.append({"Regime": regime_label, "CRPS (EUR/MWh)": float(v)})

    df_violin = pd.DataFrame(regime_data)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(df_violin) > 0:
        palette_violin = {
            "Negative\n(y < 0)": _PALETTE[1],
            "Base\n(0 ≤ y ≤ 50)": _PALETTE[0],
            "Spike\n(y > 200)": _PALETTE[3],
        }
        # Use only the regimes that are present
        present_regimes = df_violin["Regime"].unique().tolist()
        order = [
            r
            for r in ["Negative\n(y < 0)", "Base\n(0 ≤ y ≤ 50)", "Spike\n(y > 200)"]
            if r in present_regimes
        ]

        sns.violinplot(
            data=df_violin,
            x="Regime",
            y="CRPS (EUR/MWh)",
            order=order,
            palette=palette_violin,
            inner="box",
            cut=0,
            linewidth=1.2,
            ax=ax,
        )

        # Overlay mean annotation
        regime_means = {
            "Negative\n(y < 0)": mean_crps_neg,
            "Base\n(0 ≤ y ≤ 50)": mean_crps_base,
            "Spike\n(y > 200)": mean_crps_spike,
        }
        for i, regime in enumerate(order):
            mean_val = regime_means[regime]
            if not np.isnan(mean_val):
                ax.text(
                    i,
                    min(mean_val * 1.05, clip_p99 * 0.95),
                    f"μ={mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="k",
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No observations in any regime",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_title(
        "LF-GP-NRF — CRPS by Price Regime",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Price Regime", fontsize=11)
    ax.set_ylabel("CRPS (EUR/MWh, clipped at 99th pct)", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "negative_price_crps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[negative_prices] Saved → {out_path}")

    return {
        "crps_negative": mean_crps_neg,
        "crps_base": mean_crps_base,
        "crps_spike": mean_crps_spike,
        "n_negative": n_neg,
        "n_spike": n_spike,
    }


# =============================================================================
# 5.  main
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """Entry point — orchestrates all evaluation steps."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "data/raw"
    OUT_DIR = "outputs"
    FIGS_DIR = os.path.join(OUT_DIR, "figures")
    TABLES_DIR = os.path.join(OUT_DIR, "tables")

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}")

    # ------------------------------------------------------------------
    # 1.  Build dataloaders
    # ------------------------------------------------------------------
    print("\n[Step 1] Building dataloaders …")
    train_loader, val_loader, test_loader, scalers = build_dataloaders(
        data_path=os.path.join(DATA_DIR, "Germany_master_entsoe_2015_2026.csv"),
        comm_path=os.path.join(DATA_DIR, "commodities.csv"),
        flow_path=os.path.join(DATA_DIR, "cross_border_flows.csv"),
        train_end="2022-12-31",
        val_end="2023-12-31",
        test_end="2024-12-31",
        batch_size=16,  # smaller for inference — need room for n_paths_infer=64
        num_workers=0,
    )
    print(
        f"  train={len(train_loader.dataset)} windows, "
        f"val={len(val_loader.dataset)} windows, "
        f"test={len(test_loader.dataset)} windows"
    )

    # ------------------------------------------------------------------
    # 2.  Load model
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading model …")
    ckpt = args.checkpoint or os.path.join(
        OUT_DIR, "checkpoints", "lfgpnrf_phase3_final.pt"
    )
    feat_dims = os.path.join(OUT_DIR, "feature_dims.json")
    model = load_model(ckpt, feat_dims, DEVICE)
    print(f"  Loaded model from {ckpt}")
    print(
        f"  n_paths_infer={model.n_paths_infer}, n_samples_infer={model.n_samples_infer}"
    )

    # ------------------------------------------------------------------
    # 3.  Choose evaluation split
    # ------------------------------------------------------------------
    loader = test_loader if args.split == "test" else val_loader
    print(
        f"\n[Step 3] Evaluating on '{args.split}' split ({len(loader.dataset)} windows)"
    )

    # ------------------------------------------------------------------
    # 4.  Run inference
    # ------------------------------------------------------------------
    print("\n[Step 4] Running inference …")
    evaluator = EPFEvaluator(model, loader, device=DEVICE)
    results = evaluator.run()
    print(f"  samples shape: {results['samples'].shape}")
    print(f"  y_true  shape: {results['y_true'].shape}")

    # ------------------------------------------------------------------
    # 5.  Save raw results for re-use (avoid re-running inference)
    # ------------------------------------------------------------------
    np.save(
        os.path.join(OUT_DIR, f"results_{args.split}_samples.npy"),
        results["samples"],
    )
    np.save(
        os.path.join(OUT_DIR, f"results_{args.split}_ytrue.npy"),
        results["y_true"],
    )
    print("  Raw results saved.")

    # ------------------------------------------------------------------
    # 6.  Compute all metrics
    # ------------------------------------------------------------------
    print("\n[Step 5] Computing metrics …")
    metrics = evaluator.compute_all_metrics(results["samples"], results["y_true"])

    print("\n=== LF-GP-NRF Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<25s} {v:.4f}")

    # ------------------------------------------------------------------
    # 7.  Save metrics table (CSV + LaTeX via EPFEvaluator)
    # ------------------------------------------------------------------
    evaluator.save_results_table(metrics, OUT_DIR, model_name="LF-GP-NRF")

    # ------------------------------------------------------------------
    # 8.  Diagnostic plots
    # ------------------------------------------------------------------
    print("\n[Step 6] Generating diagnostic plots …")

    evaluator.plot_forecast_examples(
        results["samples"], results["y_true"], FIGS_DIR, n_examples=6
    )
    evaluator.plot_crps_by_hour(results["samples"], results["y_true"], FIGS_DIR)

    # Latent force plot — plot_latent_forces expects a raw DataLoader batch
    # (dict with 'history', 'future_exog', 'target_raw' as tensors), not the
    # aggregated results dict.  Fetch a single batch from the loader here.
    if "z_paths" in results and results["z_paths"] is not None:
        try:
            sample_batch = next(iter(loader))
            evaluator.plot_latent_forces(model, sample_batch, FIGS_DIR)
        except Exception as exc:  # pragma: no cover
            print(f"  [plots] plot_latent_forces failed ({exc}); skipping.")
    else:
        print("  [plots] z_paths not available — skipping latent force plot.")

    # ------------------------------------------------------------------
    # 9.  Three targeted tests
    # ------------------------------------------------------------------
    print("\n[Step 7] Running targeted tests …")

    # --- 9a.  Spike anticipation ---
    print("\n--- Spike Anticipation Test ---")
    spike_res = run_targeted_test_spike_anticipation(results, FIGS_DIR)
    print(f"  Spike windows: {spike_res['n_spike_windows']}")
    print(f"  Mean P90 at hour 0 (spike windows): {spike_res['mean_p90_spike'][0]:.2f}")
    print(f"  Mean P90 at hour 0 (base windows):  {spike_res['mean_p90_base'][0]:.2f}")

    # --- 9b.  Regime transition smoothness ---
    print("\n--- Regime Smoothness Test ---")
    smooth_res = run_targeted_test_regime_smoothness(results, FIGS_DIR)
    print(f"  Mean W1 across transitions: {smooth_res['mean_w1_by_hour'].mean():.2f}")

    # --- 9c.  Negative price CRPS ---
    print("\n--- Negative Price CRPS Test ---")
    neg_res = run_targeted_test_negative_prices(results, FIGS_DIR)
    print(
        f"  CRPS (negative prices): "
        f"{neg_res['crps_negative']:.4f}  (n={neg_res['n_negative']})"
    )
    print(f"  CRPS (base prices):     {neg_res['crps_base']:.4f}")
    print(
        f"  CRPS (spike prices):    "
        f"{neg_res['crps_spike']:.4f}  (n={neg_res['n_spike']})"
    )

    # ------------------------------------------------------------------
    # 10.  Persist targeted test results as JSON
    # ------------------------------------------------------------------
    targeted = {
        "spike_anticipation": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in spike_res.items()
        },
        "regime_smoothness": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in smooth_res.items()
        },
        "negative_price": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in neg_res.items()
        },
    }

    targeted_path = os.path.join(OUT_DIR, f"targeted_tests_{args.split}.json")
    with open(targeted_path, "w", encoding="utf-8") as fh:
        json.dump(targeted, fh, indent=2)
    print(f"\nTargeted test results saved → {targeted_path}")

    print(f"\nAll results saved to {OUT_DIR}/")
    print("Evaluation complete.")


# =============================================================================
# 6.  CLI entry point
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_evaluation.py",
        description=(
            "LF-GP-NRF evaluation runner: load checkpoint, run inference on "
            "the test / validation split, compute metrics, generate plots and "
            "run three targeted tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the Phase-3 checkpoint (.pt file).  "
            "Defaults to outputs/checkpoints/lfgpnrf_phase3_final.pt"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which data split to evaluate: 'test' (2024) or 'val' (2023).",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()
    main(_args)
