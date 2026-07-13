"""
Stage 2 – Part C: Regime Classification
=========================================
Fits a k-means (k=4) regime classifier on training data and applies it
to all splits. Falls back to GMM if silhouette score is higher.

Produces:
  data/regimes/regime_labels.parquet
  data/regimes/regime_summary_stats.csv
  results/figures/fig_01b_regime_scatter.pdf
  results/figures/fig_01c_regime_timeshare.pdf
  results/figures/fig_01d_regime_price_distributions.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Semantic labels for ordered regimes (by mean price)
REGIME_NAMES = {
    0: "Low-vol Off-peak",
    1: "Thermal Peak",
    2: "Renewable Surplus",
    3: "Extreme Spike",
}

REGIME_COLORS = {
    0: "#607D8B",   # blue-grey: off-peak
    1: "#FF6F00",   # amber: thermal peak
    2: "#43A047",   # green: surplus
    3: "#E53935",   # red: spike
}

CLUSTER_FEATURES = [
    "W_CF", "S_CF", "G_price_zscore",
    "T_demand", "hour_sin", "hour_cos", "P_lag_24",
]


# ---------------------------------------------------------------------------
# Regime labelling helpers
# ---------------------------------------------------------------------------

def _label_regimes(df_train: pd.DataFrame,
                   raw_labels: np.ndarray,
                   n_clusters: int = 4) -> dict[int, int]:
    """
    Map raw cluster IDs → semantic regime IDs by interpreting clusters:
      Regime 0 = lowest mean price        (low-vol off-peak)
      Regime 1 = highest mean price,      (thermal peak)   ← unless high RE
      Regime 2 = negative/near-zero price with high RE     (surplus)
      Regime 3 = extreme |price| > 150    (spike)
    Returns a mapping {raw_label → semantic_label}.
    """
    stats = {}
    for c in range(n_clusters):
        mask = raw_labels == c
        subset = df_train.loc[mask, "price"]
        wc = df_train.loc[mask, "W_CF"].mean()
        sc = df_train.loc[mask, "S_CF"].mean()
        re_share = wc + sc
        mean_price = float(subset.mean())
        pct_neg = float((subset < 0).mean())
        pct_extreme = float((subset.abs() > 150).mean())
        stats[c] = {
            "mean_price": mean_price,
            "pct_neg": pct_neg,
            "pct_extreme": pct_extreme,
            "re_share": re_share,
            "n": int(mask.sum()),
        }

    # Sort clusters by mean price ascending
    sorted_by_price = sorted(stats.keys(), key=lambda c: stats[c]["mean_price"])

    # First pass: assign 0 → lowest, 3 → highest
    raw_to_sem = {}
    raw_to_sem[sorted_by_price[0]] = 0   # cheapest → off-peak
    raw_to_sem[sorted_by_price[-1]] = 1  # most expensive → thermal peak candidate
    raw_to_sem[sorted_by_price[1]] = 2   # second cheapest
    raw_to_sem[sorted_by_price[2]] = 3   # third

    # Refine: extreme spike cluster = highest pct_extreme or |mean| > 150
    spike_candidate = max(stats.keys(),
                          key=lambda c: stats[c]["pct_extreme"])
    surplus_candidate = max(stats.keys(),
                            key=lambda c: stats[c]["pct_neg"] + stats[c]["re_share"] * 0.3)

    # Rebuild mapping with semantics:
    # 3 = spike (extreme pct or |mean| > 150)
    # 2 = surplus (high neg + high RE)
    # 0 = lowest mean price among remaining
    # 1 = highest mean price among remaining (thermal peak)
    remaining = set(range(n_clusters))
    final = {}

    # Assign spike
    final[spike_candidate] = 3
    remaining.discard(spike_candidate)

    # Assign surplus (if different from spike)
    if surplus_candidate in remaining:
        final[surplus_candidate] = 2
        remaining.discard(surplus_candidate)
    else:
        # Fall back to second-cheapest
        sc_candidate = sorted_by_price[1]
        if sc_candidate in remaining:
            final[sc_candidate] = 2
            remaining.discard(sc_candidate)

    # Sort remaining by mean price: lowest → 0, highest → 1
    remaining_sorted = sorted(remaining, key=lambda c: stats[c]["mean_price"])
    if len(remaining_sorted) >= 1:
        final[remaining_sorted[0]] = 0
    if len(remaining_sorted) >= 2:
        final[remaining_sorted[1]] = 1

    return final


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_and_label(fm: pd.DataFrame) -> tuple[pd.Series, str, float, float]:
    """
    Fix 5: Force price < 0 hours to Regime 2 (Renewable Surplus) before clustering.
    Then run KMeans(k=3) / GMM(k=3) on the remaining hours.
    """
    train_mask = fm["split"] == "train"
    
    # --- Step 1: Rule-based pre-labelling (Fix 5) ---
    # Any hour with price < 0 is definitively Regime 2 (surplus)
    neg_price_mask = fm["price"] < 0
    n_forced = int(neg_price_mask.sum())
    print(f"[regime] Fix 5: forcing {n_forced} negative-price hours -> Regime 2 (surplus)")
    
    # Remaining rows to cluster
    residual_mask = ~neg_price_mask
    fm_residual = fm.loc[residual_mask]
    
    # Training rows among the residual
    X_train = fm_residual.loc[fm_residual["split"] == "train", CLUSTER_FEATURES].values
    X_all   = fm_residual[CLUSTER_FEATURES].values
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_all_sc   = scaler.transform(X_all)
    
    # --- Step 2: k=3 clustering on non-negative hours ---
    print("[regime] Fitting KMeans (k=3) on non-negative-price hours ...")
    km = KMeans(n_clusters=3, n_init=20, random_state=42, max_iter=500)
    km.fit(X_train_sc)
    km_train_labels = km.labels_
    km_all_labels   = km.predict(X_all_sc)
    sil_km = float(silhouette_score(X_train_sc, km_train_labels, sample_size=5000,
                                     random_state=42))
    print(f"[regime]   KMeans silhouette = {sil_km:.4f}")
    
    print("[regime] Fitting GMM (k=3, full covariance) ...")
    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          n_init=5, random_state=42, max_iter=300)
    gmm.fit(X_train_sc)
    gmm_train_labels = gmm.predict(X_train_sc)
    gmm_all_labels   = gmm.predict(X_all_sc)
    sil_gmm = float(silhouette_score(X_train_sc, gmm_train_labels, sample_size=5000,
                                      random_state=42))
    print(f"[regime]   GMM silhouette    = {sil_gmm:.4f}")
    
    if sil_km >= sil_gmm:
        method = "kmeans"
        raw_labels_train = km_train_labels
        raw_labels_all   = km_all_labels
        print("[regime] -> KMeans selected")
    else:
        method = "gmm"
        raw_labels_train = gmm_train_labels
        raw_labels_all   = gmm_all_labels
        print("[regime] -> GMM selected")
    
    # --- Step 3: Label the 3 residual clusters semantically (exclude Regime 2 = 0,1,3 only) ---
    # Map the 3 raw cluster IDs to {0 (off-peak), 1 (thermal peak), 3 (spike)}
    df_train_res = fm_residual.loc[fm_residual["split"] == "train"].copy()
    df_train_res["_raw"] = raw_labels_train
    
    stats = {}
    for c in range(3):
        mask = raw_labels_train == c
        subset = df_train_res.loc[mask, "price"]
        pct_extreme = float((subset.abs() > 150).mean())
        stats[c] = {"mean_price": float(subset.mean()), "pct_extreme": pct_extreme}
    
    sorted_by_price = sorted(stats.keys(), key=lambda c: stats[c]["mean_price"])
    spike_candidate = max(stats.keys(), key=lambda c: stats[c]["pct_extreme"])
    
    raw_to_sem = {}
    raw_to_sem[spike_candidate] = 3  # Extreme spike
    remaining = [c for c in sorted_by_price if c != spike_candidate]
    raw_to_sem[remaining[0]] = 0   # Lowest price -> off-peak
    raw_to_sem[remaining[1]] = 1   # Highest among remaining -> thermal peak
    print(f"[regime] Residual regime mapping (raw->semantic): {raw_to_sem}")
    
    # --- Step 4: Assemble full regime series ---
    regime_residual = np.vectorize(raw_to_sem.get)(raw_labels_all)
    regime_all = pd.Series(index=fm.index, dtype=np.int8, name="regime")
    regime_all[residual_mask] = regime_residual
    regime_all[neg_price_mask] = 2  # Force surplus
    
    return regime_all, method, sil_km, sil_gmm


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def regime_summary(fm: pd.DataFrame,
                   regimes: pd.Series,
                   method: str,
                   sil_km: float,
                   sil_gmm: float) -> pd.DataFrame:
    joint = fm.copy()
    joint["regime"] = regimes

    rows = []
    for r in sorted(joint["regime"].unique()):
        subset = joint[joint["regime"] == r]
        rows.append({
            "regime": int(r),
            "regime_name": REGIME_NAMES.get(r, f"Regime {r}"),
            "n_hours": len(subset),
            "mean_price": float(subset["price"].mean()),
            "std_price": float(subset["price"].std()),
            "min_price": float(subset["price"].min()),
            "max_price": float(subset["price"].max()),
            "pct_negative": float((subset["price"] < 0).mean()),
            "pct_extreme": float((subset["price"].abs() > 150).mean()),
            "mean_W_CF": float(subset["W_CF"].mean()),
            "mean_S_CF": float(subset["S_CF"].mean()),
            "mean_re_penetration": float(subset["renewable_penetration"].mean()),
            "method": method,
            "silhouette_kmeans": round(sil_km, 4),
            "silhouette_gmm": round(sil_gmm, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1b: 2x2 scatter W_CF vs G_price_zscore coloured by regime
# ---------------------------------------------------------------------------

def plot_fig01b(fm: pd.DataFrame, regimes: pd.Series, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})

    train_mask = fm["split"] == "train"
    df_tr = fm.loc[train_mask].copy()
    df_tr["regime"] = regimes.loc[train_mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Figure 1b - Regime Clusters: W_CF vs G_price_zscore (Training data only)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, r in zip(axes.flat, range(4)):
        is_r = df_tr["regime"] == r
        rest = df_tr[~is_r]
        fore = df_tr[is_r]

        # Background (other regimes)
        ax.scatter(rest["W_CF"], rest["G_price_zscore"],
                   c="#BDBDBD", s=2, alpha=0.3, rasterized=True)
        # Focal regime
        ax.scatter(fore["W_CF"], fore["G_price_zscore"],
                   c=REGIME_COLORS[r], s=4, alpha=0.5, rasterized=True,
                   label=REGIME_NAMES.get(r))

        ax.set_xlabel("Wind Capacity Factor (W_CF)")
        ax.set_ylabel("Gas Price z-score")
        ax.set_xlim(-0.05, 1.05)
        title = f"Regime {r}: {REGIME_NAMES.get(r, '')}"
        title += f"\n(n={is_r.sum():,}, mean price={df_tr.loc[is_r,'price'].mean():.1f} EUR/MWh)"
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", markerscale=4, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[regime] Figure 1b saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 1c: Stacked area chart of regime share per month
# ---------------------------------------------------------------------------

def plot_fig01c(fm: pd.DataFrame, regimes: pd.Series, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})

    df = fm.copy()
    df["regime"] = regimes
    df["ym"] = df.index.to_period("M")

    # Share per month
    monthly = (df.groupby(["ym", "regime"]).size()
                 .unstack(fill_value=0))
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index()

    fig, ax = plt.subplots(figsize=(15, 5))
    colors = [REGIME_COLORS[c] for c in sorted(monthly.columns)]
    labels = [REGIME_NAMES.get(c, f"Regime {c}") for c in sorted(monthly.columns)]

    ax.stackplot(monthly.index,
                 [monthly[c] for c in sorted(monthly.columns)],
                 colors=colors, labels=labels, alpha=0.85)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Month")
    ax.set_ylabel("Regime share")
    ax.set_title("Figure 1c - Monthly Regime Share 2018-2026",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[regime] Figure 1c saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 1d: Box plots of price per regime (symlog y-axis for negatives)
# ---------------------------------------------------------------------------

def plot_fig01d(fm: pd.DataFrame, regimes: pd.Series, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                          "axes.spines.top": False, "axes.spines.right": False})

    df = fm.copy()
    df["regime"] = regimes

    data_by_regime = [df.loc[df["regime"] == r, "price"].values for r in range(4)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        data_by_regime,
        positions=range(4),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=1.5, alpha=0.3),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        notch=False,
    )

    for patch, r in zip(bp["boxes"], range(4)):
        patch.set_facecolor(REGIME_COLORS[r])
        patch.set_alpha(0.80)

    # Symmetric log scale handles both positive AND negative prices correctly
    # linthresh = linear region around zero  (±1 EUR/MWh)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.axhline(0, color="#757575", linewidth=0.8, linestyle="--")
    ax.axhline(150, color="#B71C1C", linewidth=0.8, linestyle=":",
               label="|P| = 150 EUR/MWh (spike threshold)")
    ax.axhline(-150, color="#B71C1C", linewidth=0.8, linestyle=":")

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Regime {r}\n{REGIME_NAMES.get(r,'')}" for r in range(4)],
                        fontsize=9)
    ax.set_ylabel("Price (EUR/MWh) - symlog scale")
    ax.set_title("Figure 1d - Price Distribution by Regime (symlog y-axis)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[regime] Figure 1d saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg_path: Path | None = None) -> None:
    with open(cfg_path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load feature matrix
    fm_path = ROOT / cfg["data"]["processed_path"]
    if not fm_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {fm_path}. "
            "Run build_features.py first."
        )
    print("[regime] Loading feature matrix ...")
    fm = pd.read_parquet(fm_path)

    # Drop any rows with NaNs in clustering features
    fm_clean = fm.dropna(subset=CLUSTER_FEATURES).copy()
    n_dropped = len(fm) - len(fm_clean)
    if n_dropped:
        print(f"[regime] Dropped {n_dropped} rows with NaN in cluster features")

    # ---- Fit and label ----
    regimes, method, sil_km, sil_gmm = fit_and_label(fm_clean)

    # ---- Regime labels parquet ----
    regimes_dir = ROOT / cfg["data"]["regimes_dir"]
    regimes_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.DataFrame({
        "regime": regimes,
        "split": fm_clean["split"],
        "price": fm_clean["price"],
    })
    labels_path = regimes_dir / "regime_labels.parquet"
    labels_df.to_parquet(labels_path)
    print(f"[regime] Saved regime labels -> {labels_path}")

    # ---- Summary CSV ----
    summary = regime_summary(fm_clean, regimes, method, sil_km, sil_gmm)
    summary_path = regimes_dir / "regime_summary_stats.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[regime] Saved regime summary ->\n{summary.to_string(index=False)}\n"
          f"  -> {summary_path}")

    # ---- Figures ----
    fig_dir = ROOT / cfg["data"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig01b(fm_clean, regimes, fig_dir / "fig_01b_regime_scatter.pdf")
    plot_fig01c(fm_clean, regimes, fig_dir / "fig_01c_regime_timeshare.pdf")
    plot_fig01d(fm_clean, regimes, fig_dir / "fig_01d_regime_price_distributions.pdf")

    print("\n[regime] All outputs complete.")


if __name__ == "__main__":
    run()
