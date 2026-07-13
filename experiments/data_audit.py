#!/usr/bin/env python
# data_audit.py — Section 2: Empirical Stylised Facts for LF-GP-NRF
# Usage: python src/experiments/data_audit.py
# Saves all figures to outputs/figures/ and summary to outputs/tables/data_audit.json

import json
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis, skew
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, grangercausalitytests

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.preprocess import EPFPreprocessor

sns.set_style("whitegrid")
FIGDPI = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_path: str, comm_path: str | None = None) -> pd.DataFrame:
    """Load CSV, parse datetime, set UTC index, join commodities, run preprocessor."""
    df = pd.read_csv(data_path)

    # Identify and parse the datetime column
    date_col = "date" if "date" in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.set_index(date_col).sort_index()

    # Join commodities if provided
    if comm_path is not None and os.path.isfile(comm_path):
        try:
            comm = pd.read_csv(comm_path)
            # Find datetime column
            dt_col = comm.columns[0]
            comm[dt_col] = pd.to_datetime(comm[dt_col], utc=True)
            comm = comm.set_index(dt_col).sort_index()
            # Forward-fill daily commodity data to hourly
            comm = comm.resample("h").ffill()
            df = df.join(comm, how="left")
            print(f"  Joined commodities: {list(comm.columns)}")
        except Exception as exc:
            print(f"  [WARN] Could not join commodities: {exc}")

    # Run full preprocessing pipeline
    preprocessor = EPFPreprocessor()
    df = preprocessor.process(df)
    return df


# ---------------------------------------------------------------------------
# Analysis 1 — Price distribution
# ---------------------------------------------------------------------------


def analysis_1_price_distribution(df: pd.DataFrame, save_dir: str) -> dict:
    """Histogram + KDE + yearly box plots; GPD tail fit; distribution statistics."""
    prices = df["price"].dropna()

    neg_pct = float((prices < 0).mean() * 100)
    spike_pct = float((prices > 200).mean() * 100)
    p_mean = float(prices.mean())
    p_median = float(prices.median())
    p1 = float(np.percentile(prices, 1))
    p99 = float(np.percentile(prices, 99))

    sk = float(skew(prices))
    ku = float(kurtosis(prices, fisher=True))  # excess kurtosis
    jb_stat, jb_p, _, _ = jarque_bera(prices.values)

    # GPD fit to spike tail (prices > 200)
    tail = prices[prices > 200] - 200.0
    gpd_xi = np.nan
    if len(tail) >= 30:
        try:
            xi, loc, scale = stats.genpareto.fit(tail, floc=0)
            gpd_xi = float(xi)
        except Exception:
            pass

    # ---- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(14, 10), dpi=FIGDPI)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    # -- Panel 1: histogram + KDE
    ax1 = fig.add_subplot(gs[0])
    clip_prices = prices.clip(-300, 600)
    ax1.hist(
        clip_prices,
        bins=200,
        density=True,
        color="steelblue",
        alpha=0.55,
        label="Hourly prices (clipped ±300/600)",
    )
    kde = stats.gaussian_kde(clip_prices, bw_method="silverman")
    xg = np.linspace(clip_prices.min(), clip_prices.max(), 800)
    ax1.plot(xg, kde(xg), color="navy", lw=1.8, label="KDE")

    ax1.axvline(
        0, color="crimson", lw=1.4, ls="--", label="Negative threshold (0 €/MWh)"
    )
    ax1.axvline(
        200, color="darkorange", lw=1.4, ls="--", label="Spike threshold (200 €/MWh)"
    )
    ax1.axvline(p_mean, color="green", lw=1.2, ls=":", label=f"Mean={p_mean:.1f}")
    ax1.axvline(
        p_median, color="purple", lw=1.2, ls=":", label=f"Median={p_median:.1f}"
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Price (€/MWh)", fontsize=11)
    ax1.set_ylabel("Density (log)", fontsize=11)
    ax1.set_title(
        "Marginal Distribution of Hourly Electricity Prices — Germany 2015–2024",
        fontsize=12,
    )
    # Annotation box
    ann = (
        f"Neg%={neg_pct:.2f}%  Spike%={spike_pct:.2f}%\n"
        f"P1={p1:.1f}  P99={p99:.1f}\n"
        f"Skew={sk:.2f}  ExKurt={ku:.2f}\n"
        f"JB p={jb_p:.2e}  GPD ξ={gpd_xi:.3f}"
    )
    ax1.text(
        0.98,
        0.97,
        ann,
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8),
    )
    ax1.legend(fontsize=8, loc="upper left")

    # -- Panel 2: box plots by year
    ax2 = fig.add_subplot(gs[1])
    df_y = df[["price"]].copy()
    df_y["year"] = df_y.index.year
    years = sorted(df_y["year"].unique())
    data_by_year = [df_y.loc[df_y["year"] == y, "price"].dropna().values for y in years]
    bp = ax2.boxplot(
        data_by_year,
        labels=years,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="red", lw=2),
    )
    cmap = plt.cm.viridis
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(len(years) - 1, 1)))
        patch.set_alpha(0.7)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Price (€/MWh)", fontsize=11)
    ax2.set_title(
        "Annual Price Distribution — Box Plots (no outliers shown)", fontsize=12
    )

    plt.savefig(
        os.path.join(save_dir, "price_distribution.png"),
        dpi=FIGDPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "mean": p_mean,
        "median": p_median,
        "p1": p1,
        "p99": p99,
        "negative_pct": neg_pct,
        "spike_pct": spike_pct,
        "skewness": sk,
        "excess_kurtosis": ku,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "gpd_tail_index_xi": gpd_xi,
    }


# ---------------------------------------------------------------------------
# Analysis 2 — Negative price trends by year
# ---------------------------------------------------------------------------


def analysis_2_negative_prices_by_year(df: pd.DataFrame, save_dir: str) -> dict:
    """Bar chart: neg% and spike% by year; RE penetration overlay; Spearman correlation."""
    df2 = df.copy()
    df2["year"] = df2.index.year
    years = list(range(2015, 2025))
    df2 = df2[df2["year"].isin(years)]

    neg_pct_yr = df2.groupby("year")["price"].apply(lambda x: (x < 0).mean() * 100)
    spike_pct_yr = df2.groupby("year")["price"].apply(lambda x: (x > 200).mean() * 100)

    # RE penetration columns — use whichever is available
    solar_col = "solar_penetration" if "solar_penetration" in df2.columns else None
    wind_col = (
        "wind_on_penetration"
        if "wind_on_penetration" in df2.columns
        else "total_re_penetration"
        if "total_re_penetration" in df2.columns
        else None
    )

    mean_solar = (
        df2.groupby("year")[solar_col].mean()
        if solar_col
        else pd.Series(np.nan, index=years)
    )
    mean_wind = (
        df2.groupby("year")[wind_col].mean()
        if wind_col
        else pd.Series(np.nan, index=years)
    )

    # Spearman correlation: neg% ~ mean solar penetration
    spearman_r, spearman_p = np.nan, np.nan
    if solar_col:
        common = neg_pct_yr.index.intersection(mean_solar.index)
        if len(common) >= 4:
            spearman_r, spearman_p = stats.spearmanr(
                neg_pct_yr.loc[common], mean_solar.loc[common]
            )

    # ---- Figure --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=FIGDPI)
    x = np.arange(len(years))
    w = 0.38

    ax1.bar(
        x - w / 2,
        [neg_pct_yr.get(y, 0) for y in years],
        width=w,
        color="steelblue",
        alpha=0.8,
        label="Negative price %",
    )
    ax1.bar(
        x + w / 2,
        [spike_pct_yr.get(y, 0) for y in years],
        width=w,
        color="tomato",
        alpha=0.8,
        label="Spike (>200) %",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, fontsize=10)
    ax1.set_ylabel("Share of hours (%)", fontsize=11)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_title(
        "Negative Price & Spike Frequency by Year — Germany 2015–2024", fontsize=12
    )

    ax2 = ax1.twinx()
    if solar_col:
        ax2.plot(
            x,
            [mean_solar.get(y, np.nan) for y in years],
            "o--",
            color="gold",
            lw=1.8,
            markersize=6,
            label="Mean solar penetration",
        )
    if wind_col:
        ax2.plot(
            x,
            [mean_wind.get(y, np.nan) for y in years],
            "s--",
            color="seagreen",
            lw=1.8,
            markersize=6,
            label="Mean wind penetration",
        )
    ax2.set_ylabel("Mean RE penetration (capacity-norm.)", fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ann = (
        f"Spearman(neg%, solar): r={spearman_r:.3f},  p={spearman_p:.3f}"
        if not np.isnan(spearman_r)
        else "Spearman: n/a"
    )
    fig.text(0.5, -0.02, ann, ha="center", fontsize=9, style="italic")

    plt.savefig(
        os.path.join(save_dir, "negative_price_trends.png"),
        dpi=FIGDPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "neg_pct_by_year": {y: float(neg_pct_yr.get(y, np.nan)) for y in years},
        "spike_pct_by_year": {y: float(spike_pct_yr.get(y, np.nan)) for y in years},
        "spearman_neg_solar_r": float(spearman_r) if not np.isnan(spearman_r) else None,
        "spearman_neg_solar_p": float(spearman_p) if not np.isnan(spearman_p) else None,
    }


# ---------------------------------------------------------------------------
# Analysis 3 — Seasonal patterns
# ---------------------------------------------------------------------------


def analysis_3_seasonal_patterns(df: pd.DataFrame, save_dir: str) -> dict:
    """4-panel: hourly profile (winter/summer), day-of-week, monthly, hour×month heatmap."""
    df3 = df.copy()
    df3["hour"] = df3.index.hour
    df3["month"] = df3.index.month
    df3["dow"] = df3.index.dayofweek
    # Winter = Dec/Jan/Feb, Summer = Jun/Jul/Aug
    df3["season"] = "other"
    df3.loc[df3["month"].isin([12, 1, 2]), "season"] = "winter"
    df3.loc[df3["month"].isin([6, 7, 8]), "season"] = "summer"

    # Hour profiles
    hour_stats = (
        df3.groupby(["season", "hour"])["price"].agg(["mean", "std"]).reset_index()
    )
    hour_all = df3.groupby("hour")["price"].mean()

    # Identify peak / off-peak hours from all-year profile
    peak_hours = sorted(hour_all.nlargest(4).index.tolist())
    offpeak_hours = sorted(hour_all.nsmallest(4).index.tolist())

    # Day-of-week and monthly averages
    dow_mean = df3.groupby("dow")["price"].mean()
    month_mean = df3.groupby("month")["price"].mean()

    # Winter premium = winter mean − summer mean
    winter_mean = df3.loc[df3["season"] == "winter", "price"].mean()
    summer_mean = df3.loc[df3["season"] == "summer", "price"].mean()
    winter_premium = float(winter_mean - summer_mean)

    # Hour × month pivot for heatmap
    pivot = df3.groupby(["hour", "month"])["price"].mean().unstack("month")

    # ---- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(16, 14), dpi=FIGDPI)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Panel 1: hourly profile
    ax1 = fig.add_subplot(gs[0, 0])
    palette = {"winter": "steelblue", "summer": "goldenrod", "other": "grey"}
    for season in ["winter", "summer"]:
        sub = hour_stats[hour_stats["season"] == season]
        ax1.plot(
            sub["hour"],
            sub["mean"],
            lw=2,
            label=season.capitalize(),
            color=palette[season],
        )
        ax1.fill_between(
            sub["hour"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.2,
            color=palette[season],
        )
    ax1.set_xlabel("Hour of day (UTC)", fontsize=10)
    ax1.set_ylabel("Mean price (€/MWh)", fontsize=10)
    ax1.set_title("Mean Hourly Price Profile by Season (±1σ)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(0, 24, 2))

    # Panel 2: day-of-week
    ax2 = fig.add_subplot(gs[0, 1])
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax2.bar(
        range(7),
        [dow_mean.get(i, np.nan) for i in range(7)],
        color="mediumseagreen",
        alpha=0.8,
        edgecolor="white",
    )
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dow_labels, fontsize=10)
    ax2.set_ylabel("Mean price (€/MWh)", fontsize=10)
    ax2.set_title("Mean Price by Day of Week", fontsize=11)

    # Panel 3: monthly
    ax3 = fig.add_subplot(gs[1, 0])
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax3.bar(
        range(1, 13),
        [month_mean.get(m, np.nan) for m in range(1, 13)],
        color="coral",
        alpha=0.8,
        edgecolor="white",
    )
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_labels, fontsize=9, rotation=30)
    ax3.set_ylabel("Mean price (€/MWh)", fontsize=10)
    ax3.set_title("Mean Price by Month", fontsize=11)

    # Panel 4: hour × month heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    pivot_vals = pivot.reindex(columns=range(1, 13))
    sns.heatmap(
        pivot_vals,
        ax=ax4,
        cmap="RdYlGn_r",
        xticklabels=month_labels,
        yticklabels=[str(h) for h in range(0, 24)],
        cbar_kws={"label": "€/MWh"},
        linewidths=0,
    )
    ax4.set_xlabel("Month", fontsize=10)
    ax4.set_ylabel("Hour of day (UTC)", fontsize=10)
    ax4.set_title("Mean Price Heat Map: Hour × Month", fontsize=11)
    ax4.set_yticks(range(0, 24, 2))
    ax4.set_yticklabels(range(0, 24, 2), fontsize=7)
    ax4.set_xticklabels(month_labels, fontsize=8, rotation=30)

    fig.suptitle("Seasonal Patterns — German Electricity Prices", fontsize=13, y=1.01)
    plt.savefig(
        os.path.join(save_dir, "seasonal_patterns.png"), dpi=FIGDPI, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "peak_hours": peak_hours,
        "offpeak_hours": offpeak_hours,
        "winter_premium_eur_mwh": winter_premium,
        "monthly_mean": {m: float(month_mean.get(m, np.nan)) for m in range(1, 13)},
    }


# ---------------------------------------------------------------------------
# Analysis 4 — Residual load vs price (merit order)
# ---------------------------------------------------------------------------


def analysis_4_residual_load_vs_price(df: pd.DataFrame, save_dir: str) -> dict:
    """Scatter: residual load vs price coloured by RE penetration; merit order curve."""
    rl_col = "Residual_Load" if "Residual_Load" in df.columns else "residual_load"
    re_col = (
        "total_re_penetration"
        if "total_re_penetration" in df.columns
        else "re_penetration"
    )

    sub = df[[rl_col, "price", re_col]].dropna()
    # Cap for visual clarity
    sub = sub[(sub["price"] > -200) & (sub["price"] < 500)]

    pearson_r, pearson_p = stats.pearsonr(sub[rl_col], sub["price"])
    spearman_r, spearman_p = stats.spearmanr(sub[rl_col], sub["price"])

    # Merit order curve: 100 percentile bins of residual load
    sub = sub.copy()
    sub["rl_bin"] = pd.qcut(sub[rl_col], q=100, labels=False, duplicates="drop")
    merit_curve = (
        sub.groupby("rl_bin")
        .agg(mean_rl=(rl_col, "mean"), mean_price=("price", "mean"))
        .dropna()
    )

    # ---- Figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=FIGDPI)
    sc = ax.scatter(
        sub[rl_col],
        sub["price"],
        c=sub[re_col],
        cmap="RdYlGn_r",
        alpha=0.18,
        s=3,
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("RE penetration", fontsize=9)

    ax.plot(
        merit_curve["mean_rl"],
        merit_curve["mean_price"],
        "o-",
        color="navy",
        lw=2,
        markersize=4,
        label="Merit order curve (100-bin mean)",
    )

    ax.axhline(0, color="crimson", lw=1.2, ls="--", alpha=0.7)
    ax.axhline(200, color="darkorange", lw=1.2, ls="--", alpha=0.7)

    ax.set_xlabel("Residual Load (MW)", fontsize=11)
    ax.set_ylabel("Price (€/MWh)", fontsize=11)
    ax.set_title("Merit Order Curve — Residual Load vs. Electricity Price", fontsize=12)
    ax.legend(fontsize=9)

    ann = (
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e})\n"
        f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.2e})"
    )
    ax.text(
        0.02,
        0.97,
        ann,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", alpha=0.85),
    )

    plt.savefig(
        os.path.join(save_dir, "merit_order.png"), dpi=FIGDPI, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


# ---------------------------------------------------------------------------
# Analysis 5 — Granger causality
# ---------------------------------------------------------------------------


def analysis_5_granger_causality(df: pd.DataFrame, save_dir: str) -> dict:
    """Granger causality: surprise signals → price; re_penetration → |price|."""
    MAXLAG = 6
    N_ROWS = 5000  # keep fast — O(n) per lag
    LAGS_OF_INTEREST = [1, 3, 6]

    sub = (
        df[
            [
                "price",
                "wind_onshore_surprise",
                "solar_surprise",
                "load_surprise",
                "re_penetration",
                "total_re_penetration",
            ]
        ]
        .dropna()
        .iloc[:N_ROWS]
        .copy()
    )

    sub["abs_price"] = sub["price"].abs()

    # Map nice names → (cause_col, effect_col)
    tests_spec = {
        "wind_onshore_surprise→price": ("wind_onshore_surprise", "price"),
        "solar_surprise→price": ("solar_surprise", "price"),
        "load_surprise→price": ("load_surprise", "price"),
        "re_penetration→abs_price": (
            "total_re_penetration"
            if "total_re_penetration" in sub.columns
            else "re_penetration",
            "abs_price",
        ),
    }

    results = {}
    rows_csv = []

    for test_name, (cause, effect) in tests_spec.items():
        if cause not in sub.columns or effect not in sub.columns:
            results[test_name] = {"error": "column_missing"}
            continue
        data_pair = sub[[effect, cause]].dropna()
        if len(data_pair) < MAXLAG + 10:
            results[test_name] = {"error": "insufficient_data"}
            continue
        try:
            gc_out = grangercausalitytests(data_pair, maxlag=MAXLAG, verbose=False)
            test_result = {}
            for lag in LAGS_OF_INTEREST:
                if lag in gc_out:
                    f_stat = gc_out[lag][0]["ssr_ftest"][0]
                    p_val = gc_out[lag][0]["ssr_ftest"][1]
                    test_result[f"lag{lag}_f"] = float(f_stat)
                    test_result[f"lag{lag}_p"] = float(p_val)
                    rows_csv.append(
                        {
                            "test": test_name,
                            "lag": lag,
                            "f_stat": float(f_stat),
                            "p_value": float(p_val),
                            "significant": p_val < 0.05,
                        }
                    )
            results[test_name] = test_result
        except Exception as exc:
            results[test_name] = {"error": str(exc)}

    # Save CSV table
    tables_dir = os.path.join(os.path.dirname(save_dir), "tables")
    os.makedirs(tables_dir, exist_ok=True)
    if rows_csv:
        pd.DataFrame(rows_csv).to_csv(
            os.path.join(tables_dir, "granger_causality.csv"), index=False
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 6 — Structural breaks
# ---------------------------------------------------------------------------


def analysis_6_structural_breaks(df: pd.DataFrame, save_dir: str) -> dict:
    """30-day rolling mean/std of daily median price; annotate gas-crisis breaks."""
    daily_median = df["price"].resample("D").median().dropna()

    roll_mean = daily_median.rolling(window=30, min_periods=15).mean()
    roll_std = daily_median.rolling(window=30, min_periods=15).std()

    # Coefficient of variation by year (regime instability)
    df_yr = df[["price"]].copy()
    df_yr["year"] = df_yr.index.year
    cv_by_year = df_yr.groupby("year")["price"].agg(
        lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan
    )

    # Break dates
    BREAKS = {
        "2021-06-01": ("Gas crisis onset", "crimson"),
        "2022-03-01": ("Peak gas crisis", "darkorange"),
        "2023-04-01": ("Nuclear shutdown / abatement", "purple"),
    }

    # ---- Figure --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), dpi=FIGDPI, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Structural Breaks in German Electricity Prices", fontsize=13)

    ax = axes[0]
    ax.plot(
        daily_median.index,
        daily_median.values,
        color="lightsteelblue",
        lw=0.8,
        alpha=0.7,
        label="Daily median price",
    )
    ax.plot(
        roll_mean.index,
        roll_mean.values,
        color="navy",
        lw=1.8,
        label="30-day rolling mean",
    )
    ax.fill_between(
        roll_mean.index,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color="steelblue",
        alpha=0.2,
        label="±1σ band",
    )

    for date_str, (label, color) in BREAKS.items():
        try:
            bdate = pd.Timestamp(date_str, tz="UTC")
            ax.axvline(bdate, color=color, lw=1.8, ls="--", label=label)
        except Exception:
            pass

    ax.set_ylabel("Price (€/MWh)", fontsize=11)
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # CV subplot
    ax2 = axes[1]
    years = sorted(cv_by_year.index)
    ax2.bar(
        years,
        [cv_by_year.get(y, np.nan) for y in years],
        color="slateblue",
        alpha=0.75,
        edgecolor="white",
    )
    ax2.set_ylabel("CV (σ/μ)", fontsize=10)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_title(
        "Annual Coefficient of Variation (regime instability proxy)", fontsize=10
    )
    ax2.set_xticks(years)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "structural_breaks.png"), dpi=FIGDPI, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "cv_by_year": {int(y): float(cv_by_year.get(y, np.nan)) for y in years},
        "annotated_break_dates": list(BREAKS.keys()),
    }


# ---------------------------------------------------------------------------
# Analysis 7 — Autocorrelation & ARCH effects
# ---------------------------------------------------------------------------


def analysis_7_autocorrelation(df: pd.DataFrame, save_dir: str) -> dict:
    """ACF to lag 200; mark 24h/168h seasonalities; ARCH test on squared residuals."""
    prices = df["price"].dropna()

    N_LAGS = 200
    acf_vals = acf(prices.values, nlags=N_LAGS, fft=True)

    acf_lag24 = float(acf_vals[24]) if len(acf_vals) > 24 else np.nan
    acf_lag48 = float(acf_vals[48]) if len(acf_vals) > 48 else np.nan
    acf_lag168 = float(acf_vals[168]) if len(acf_vals) > 168 else np.nan

    # Partial autocorrelation at lag 24 and 168 (manual: correlation after removing lower lags)
    try:
        from statsmodels.tsa.stattools import pacf

        pacf_vals = pacf(prices.values, nlags=min(168, len(prices) // 2 - 1))
        pacf_lag24 = float(pacf_vals[24]) if len(pacf_vals) > 24 else np.nan
        pacf_lag168 = float(pacf_vals[min(168, len(pacf_vals) - 1)])
    except Exception:
        pacf_lag24, pacf_lag168 = np.nan, np.nan

    # ARCH effects: remove 24h seasonal mean, compute ACF of squared residuals
    seasonal_mean = prices.groupby(prices.index.hour).transform("mean")
    residuals = (prices - seasonal_mean).fillna(0)
    sq_resid = residuals**2
    arch_acf = acf(sq_resid.values, nlags=10, fft=True)
    arch_lag1 = float(arch_acf[1])

    # Confidence band
    ci = 1.96 / np.sqrt(len(prices))

    # ---- Figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=FIGDPI, sharex=False)
    fig.suptitle(
        "Autocorrelation Structure — German Hourly Electricity Prices", fontsize=13
    )

    lags_range = np.arange(N_LAGS + 1)

    ax1 = axes[0]
    ax1.bar(lags_range, acf_vals, color="steelblue", alpha=0.6, width=0.8, label="ACF")
    ax1.axhline(ci, color="red", ls="--", lw=1.0, label=f"95% CI (±{ci:.3f})")
    ax1.axhline(-ci, color="red", ls="--", lw=1.0)
    ax1.axhline(0, color="black", lw=0.6)
    for lag, color, label in [
        (24, "darkorange", "Lag 24h"),
        (48, "purple", "Lag 48h"),
        (168, "crimson", "Lag 168h (1 week)"),
    ]:
        if lag <= N_LAGS:
            ax1.axvline(lag, color=color, lw=1.5, ls=":", label=label)
    ax1.set_ylabel("ACF", fontsize=10)
    ax1.set_xlabel("Lag (hours)", fontsize=10)
    ax1.set_xlim(0, N_LAGS)
    ax1.legend(fontsize=8, ncol=3)
    ann1 = (
        f"ACF(24h)={acf_lag24:.3f}  ACF(48h)={acf_lag48:.3f}  "
        f"ACF(168h)={acf_lag168:.3f}"
    )
    ax1.text(
        0.98,
        0.97,
        ann1,
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
    )

    # Squared residuals ACF (ARCH evidence)
    ax2 = axes[1]
    lags10 = np.arange(len(arch_acf))
    ax2.bar(
        lags10,
        arch_acf,
        color="tomato",
        alpha=0.7,
        width=0.6,
        label="ACF(ε²) — ARCH signal",
    )
    ax2.axhline(ci, color="red", ls="--", lw=1.0)
    ax2.axhline(-ci, color="red", ls="--", lw=1.0)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_ylabel("ACF of squared residuals", fontsize=10)
    ax2.set_xlabel("Lag (hours)", fontsize=10)
    ax2.set_title(
        "ARCH Effects — ACF of Squared Price Residuals (after 24h seasonal removal)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ann2 = f"ACF(ε², lag=1) = {arch_lag1:.3f}"
    ax2.text(
        0.98,
        0.97,
        ann2,
        transform=ax2.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "autocorrelation.png"), dpi=FIGDPI, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "acf_lag24": acf_lag24,
        "acf_lag48": acf_lag48,
        "acf_lag168": acf_lag168,
        "pacf_lag24": pacf_lag24,
        "pacf_lag168": pacf_lag168,
        "arch_lag1": arch_lag1,
    }


# ---------------------------------------------------------------------------
# Analysis 8 — RE penetration distribution
# ---------------------------------------------------------------------------


def analysis_8_re_penetration_distribution(df: pd.DataFrame, save_dir: str) -> dict:
    """Histogram of RE penetration; scatter vs price; neg-price prob by quintile."""
    re_col = (
        "total_re_penetration"
        if "total_re_penetration" in df.columns
        else "re_penetration"
    )
    sub = df[[re_col, "price"]].dropna()
    sub = sub[(sub["price"] > -400) & (sub["price"] < 600)]
    sub = sub.copy()
    sub["hour"] = sub.index.hour

    HIGH_RE = 0.80
    high_re_frac = float((sub[re_col] > HIGH_RE).mean() * 100)

    # Negative price probability per RE quintile
    sub["re_quintile"] = pd.qcut(
        sub[re_col], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
    )
    neg_prob_quintile = (
        sub.groupby("re_quintile", observed=True)["price"]
        .apply(lambda x: (x < 0).mean() * 100)
        .to_dict()
    )

    # ---- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(14, 10), dpi=FIGDPI)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Panel 1: histogram of RE penetration
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        sub[re_col].clip(0, 1),
        bins=80,
        color="seagreen",
        alpha=0.75,
        edgecolor="white",
        density=True,
    )
    ax1.axvline(
        HIGH_RE,
        color="crimson",
        lw=1.8,
        ls="--",
        label=f"RE > {HIGH_RE:.0%}  ({high_re_frac:.1f}% of hours)",
    )
    ax1.set_xlabel("RE penetration (capacity-normalised)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("Distribution of Hourly RE Penetration", fontsize=11)
    ax1.legend(fontsize=9)

    # Panel 2: scatter RE penetration vs price, coloured by hour-of-day
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(
        sub[re_col],
        sub["price"],
        c=sub["hour"],
        cmap="twilight",
        alpha=0.15,
        s=3,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax2, label="Hour of day")
    ax2.axhline(0, color="crimson", lw=1.2, ls="--", alpha=0.8)
    ax2.axhline(200, color="darkorange", lw=1.2, ls="--", alpha=0.8)
    ax2.axvline(
        HIGH_RE, color="purple", lw=1.4, ls=":", alpha=0.9, label=f"RE={HIGH_RE:.0%}"
    )
    ax2.set_xlabel("RE penetration", fontsize=10)
    ax2.set_ylabel("Price (€/MWh)", fontsize=10)
    ax2.set_title("RE Penetration vs. Price (coloured by hour-of-day)", fontsize=11)
    ax2.legend(fontsize=8)

    # Panel 3: highlight hours with RE > 0.8
    ax3 = fig.add_subplot(gs[1, 0])
    normal = sub[sub[re_col] <= HIGH_RE]
    high = sub[sub[re_col] > HIGH_RE]
    ax3.scatter(
        normal[re_col],
        normal["price"],
        color="steelblue",
        alpha=0.10,
        s=3,
        rasterized=True,
        label=f"RE ≤ {HIGH_RE:.0%}",
    )
    ax3.scatter(
        high[re_col],
        high["price"],
        color="crimson",
        alpha=0.30,
        s=5,
        rasterized=True,
        label=f"RE > {HIGH_RE:.0%}",
    )
    ax3.axhline(0, color="black", lw=1.0, ls="--")
    ax3.set_xlabel("RE penetration", fontsize=10)
    ax3.set_ylabel("Price (€/MWh)", fontsize=10)
    ax3.set_title(f"High RE Hours (>{HIGH_RE:.0%}) — Price Pressure", fontsize=11)
    ax3.legend(fontsize=8)

    # Panel 4: negative price probability by RE quintile
    ax4 = fig.add_subplot(gs[1, 1])
    quintile_labels = list(neg_prob_quintile.keys())
    quintile_vals = [neg_prob_quintile[q] for q in quintile_labels]
    cmap_q = plt.cm.YlOrRd
    colors_q = [
        cmap_q(i / max(len(quintile_labels) - 1, 1))
        for i in range(len(quintile_labels))
    ]
    ax4.bar(
        quintile_labels, quintile_vals, color=colors_q, alpha=0.85, edgecolor="white"
    )
    ax4.set_xlabel("RE penetration quintile", fontsize=10)
    ax4.set_ylabel("Negative price probability (%)", fontsize=10)
    ax4.set_title("P(price < 0) Conditioned on RE Quintile", fontsize=11)
    for i, (q, v) in enumerate(zip(quintile_labels, quintile_vals)):
        ax4.text(i, v + 0.05, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle(
        "Renewable Penetration Analysis — German Electricity Market", fontsize=13
    )
    plt.savefig(
        os.path.join(save_dir, "re_penetration.png"), dpi=FIGDPI, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "high_re_pct_of_hours": high_re_frac,
        "neg_price_prob_by_re_quintile": {
            k: float(v) for k, v in neg_prob_quintile.items()
        },
        "re_col_used": re_col,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    DATA_DIR = "data/raw"
    OUT_DIR = "outputs"
    FIGS_DIR = os.path.join(OUT_DIR, "figures")
    TABLES_DIR = os.path.join(OUT_DIR, "tables")
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data(
        os.path.join(DATA_DIR, "Germany_master_entsoe_2015_2026.csv"),
        os.path.join(DATA_DIR, "commodities.csv"),
    )
    print(f"  Shape: {df.shape}  |  {df.index[0]} → {df.index[-1]}")

    results = {}

    print("Analysis 1: Price distribution...")
    results["price_distribution"] = analysis_1_price_distribution(df, FIGS_DIR)

    print("Analysis 2: Negative price trends...")
    results["negative_price_trends"] = analysis_2_negative_prices_by_year(df, FIGS_DIR)

    print("Analysis 3: Seasonal patterns...")
    results["seasonal_patterns"] = analysis_3_seasonal_patterns(df, FIGS_DIR)

    print("Analysis 4: Merit order / residual load...")
    results["merit_order"] = analysis_4_residual_load_vs_price(df, FIGS_DIR)

    print("Analysis 5: Granger causality...")
    results["granger"] = analysis_5_granger_causality(df, FIGS_DIR)

    print("Analysis 6: Structural breaks...")
    results["structural_breaks"] = analysis_6_structural_breaks(df, FIGS_DIR)

    print("Analysis 7: Autocorrelation / ARCH...")
    results["autocorrelation"] = analysis_7_autocorrelation(df, FIGS_DIR)

    print("Analysis 8: RE penetration...")
    results["re_penetration"] = analysis_8_re_penetration_distribution(df, FIGS_DIR)

    # ------------------------------------------------------------------
    # Save JSON summary — convert all numpy types for serialisation
    # ------------------------------------------------------------------
    def _to_json_safe(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_safe(x) for x in obj]
        return obj

    with open(os.path.join(TABLES_DIR, "data_audit.json"), "w") as f:
        json.dump(_to_json_safe(results), f, indent=2)

    print("\nAll analyses complete.")
    print(f"  Figures  → {FIGS_DIR}/")
    print(f"  Summary  → {TABLES_DIR}/data_audit.json")


if __name__ == "__main__":
    main()
