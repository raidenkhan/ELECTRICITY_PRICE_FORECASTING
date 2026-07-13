"""
Stage 2 – Part B: Rolling MFDFA Hurst Exponent
================================================
Implements Multifractal Detrended Fluctuation Analysis (MFDFA) with a
causal rolling window and produces Figure 1a.

Algorithm reference:
  Kantelhardt et al. (2002), Physica A 316, 87-114.

CRITICAL: Only uses data that was AVAILABLE at time t (causal).
          The H(t) value at position t uses prices from [t-719, t].

Output:
  data/regimes/hurst_series.parquet
  results/figures/fig_01a_hurst_series.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# MFDFA core (vectorised)
# ---------------------------------------------------------------------------

def _mfdfa_single(x: np.ndarray,
                  scales: list[int],
                  q_orders: list[float],
                  poly_order: int = 2) -> dict[float, float]:
    """
    Compute generalised Hurst exponents H(q) for a single price window.

    Parameters
    ----------
    x          : 1-D price array (length N, typically 720)
    scales     : segment lengths  [8, 16, 32, 64, 128, 256]
    q_orders   : generalised orders [-4, -2, 0, 2, 4]
    poly_order : polynomial degree for local detrending (2)

    Returns
    -------
    dict mapping q → H(q)
    """
    N = len(x)
    x_mean = x.mean()
    profile = np.cumsum(x - x_mean)           # integrated profile Y(i)

    log_scales = np.log(np.asarray(scales, dtype=float))
    Fq = np.full((len(q_orders), len(scales)), np.nan)

    for js, s in enumerate(scales):
        Ns = N // s
        if Ns < 2:
            continue

        # Collect forward and backward non-overlapping segments
        fwd = [profile[v * s:(v + 1) * s] for v in range(Ns)]
        bwd = [profile[N - (v + 1) * s: N - v * s] for v in range(Ns)]
        segs = np.array(fwd + bwd, dtype=float)   # shape (2*Ns, s)

        # Design matrix for polynomial fitting (shared across segments)
        x_seg = np.arange(s, dtype=float)
        V = np.vander(x_seg, poly_order + 1, increasing=True)   # (s, p+1)
        VtV_inv = np.linalg.pinv(V.T @ V)                        # (p+1, p+1)

        # Batch polynomial fit: coeffs (2Ns, p+1), trends (2Ns, s)
        VtY   = segs @ V                             # (2Ns, p+1)
        coeffs = VtY @ VtV_inv                       # (2Ns, p+1)
        trends = coeffs @ V.T                        # (2Ns, s)

        residuals = segs - trends                    # (2Ns, s)
        F2 = np.mean(residuals ** 2, axis=1)         # variance per segment

        # q-th order fluctuation function
        for iq, q in enumerate(q_orders):
            F2_pos = np.maximum(F2, 1e-30)
            if q == 0:
                # Geometric mean
                Fq[iq, js] = np.exp(0.5 * np.mean(np.log(F2_pos)))
            else:
                Fq[iq, js] = np.mean(F2_pos ** (q / 2)) ** (1.0 / q)

    # Log-log regression → slope = H(q)
    H_dict: dict[float, float] = {}
    for iq, q in enumerate(q_orders):
        fq = Fq[iq, :]
        valid = np.isfinite(fq) & (fq > 0)
        if valid.sum() >= 2:
            slope = float(np.polyfit(log_scales[valid], np.log(fq[valid]), 1)[0])
            H_dict[q] = slope
        else:
            H_dict[q] = np.nan

    return H_dict


# ---------------------------------------------------------------------------
# Rolling computation (joblib parallel)
# ---------------------------------------------------------------------------

def _process_one(prices_arr: np.ndarray,
                 start: int, window: int,
                 scales: list[int], q_orders: list[float]) -> float:
    """Process one window and return H(q=2)."""
    x = prices_arr[start: start + window]
    h = _mfdfa_single(x, scales, q_orders, poly_order=2)
    return h.get(2, np.nan)


def compute_hurst_series(prices: pd.Series,
                         window: int = 720,
                         step: int = 24,
                         scales: list[int] | None = None,
                         q_orders: list[float] | None = None,
                         n_jobs: int = -1) -> pd.Series:
    """
    Rolling causal MFDFA.

    H(t) at position ``end - 1`` uses prices in [start, end).
    Values before the first full window are NaN.

    Parameters
    ----------
    prices   : hourly price Series with DatetimeIndex
    window   : rolling window length in hours (720)
    step     : step between successive windows in hours (24)
    n_jobs   : parallel jobs (joblib); -1 = all cores

    Returns
    -------
    pd.Series of H(q=2), same index as `prices`, NaN where unavailable.
    """
    if scales is None:
        scales = [8, 16, 32, 64, 128, 256]
    if q_orders is None:
        q_orders = [-4, -2, 0, 2, 4]

    prices_arr = prices.values.astype(float)
    N = len(prices_arr)

    starts = list(range(0, N - window + 1, step))
    print(f"[hurst] Processing {len(starts)} windows "
          f"(window={window}h, step={step}h) …")

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_one)(prices_arr, s, window, scales, q_orders)
            for s in starts
        )
    except ImportError:
        results = [_process_one(prices_arr, s, window, scales, q_orders)
                   for s in starts]

    # Assign each H value to the LAST index of its window (causal)
    H_arr = np.full(N, np.nan)
    for i, start in enumerate(starts):
        end_idx = start + window - 1        # last index of this window
        H_arr[end_idx] = results[i]

    # Forward-fill the 24-h gaps between daily estimates
    H_series = pd.Series(H_arr, index=prices.index, name="H")
    H_series = H_series.ffill()             # causal: propagate last known H

    print(f"[hurst] Done. H range: [{np.nanmin(H_arr):.3f}, "
          f"{np.nanmax(H_arr):.3f}]  "
          f"NaN fraction: {np.isnan(H_series.values).mean():.2%}")
    return H_series


# ---------------------------------------------------------------------------
# Figure 1a: H(t) time series
# ---------------------------------------------------------------------------

def plot_fig01a(H_series: pd.Series, out_path: Path) -> None:
    """
    Figure 1a – H(t) 2018-2026 with threshold lines and shaded regions.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Drop leading NaN
    H = H_series.dropna()

    # Shaded regions
    ax.fill_between(H.index, 0.6, 1.0,
                    alpha=0.12, color="#2196F3", label="Persistent (H > 0.6)")
    ax.fill_between(H.index, 0.0, 0.4,
                    alpha=0.12, color="#F44336", label="Anti-persistent (H < 0.4)")

    # H(t) trace
    ax.plot(H.index, H.values, color="#37474F", linewidth=0.6, alpha=0.85)

    # Reference lines
    ax.axhline(0.5, color="#FF9800", linewidth=1.5, linestyle="--",
               label="Random walk (H = 0.5)")
    ax.axhline(0.6, color="#2196F3", linewidth=0.8, linestyle=":")
    ax.axhline(0.4, color="#F44336", linewidth=0.8, linestyle=":")

    # Axis limits / labels
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Generalised Hurst Exponent H(q=2)")
    ax.set_title("Figure 1a – Rolling MFDFA Generalised Hurst Exponent H(t), 2018–2026",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[hurst] Figure 1a saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg_path: Path | None = None) -> pd.Series:
    with open(cfg_path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    feat_cfg = cfg["features"]
    window    = feat_cfg["hurst_window"]
    step      = feat_cfg["hurst_step"]
    scales    = feat_cfg["hurst_scales"]
    q_orders  = feat_cfg["hurst_q_orders"]

    # Load feature matrix to get the price series
    fm_path = ROOT / cfg["data"]["processed_path"]
    if not fm_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {fm_path}. "
            "Run build_features.py first."
        )
    fm = pd.read_parquet(fm_path)
    prices = fm["price"].sort_index()

    # Compute rolling MFDFA
    H_series = compute_hurst_series(prices, window=window, step=step,
                                    scales=scales, q_orders=q_orders)

    # Save
    regimes_dir = ROOT / cfg["data"]["regimes_dir"]
    regimes_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = regimes_dir / "hurst_series.parquet"
    h_df = H_series.to_frame("H")
    h_df["split"] = fm["split"]
    h_df.to_parquet(out_parquet)
    print(f"[hurst] Saved hurst_series -> {out_parquet}  shape={h_df.shape}")

    # Figure 1a
    fig_dir = ROOT / cfg["data"]["figures_dir"]
    plot_fig01a(H_series, fig_dir / "fig_01a_hurst_series.pdf")

    return H_series


if __name__ == "__main__":
    run()
