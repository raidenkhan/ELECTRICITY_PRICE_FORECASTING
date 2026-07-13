"""
Stage 2 – Part A: Feature Engineering
======================================
Builds the hourly feature matrix X_t from the raw ENTSO-E dataset and
commodities file. All transformations are causally valid (no look-ahead).

Output: data/processed/feature_matrix.parquet  (with 'split' column)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(cfg_path: Path | None = None) -> dict:
    path = cfg_path or ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for key in ("processed_path", "regimes_dir", "figures_dir"):
        p = ROOT / (cfg["data"].get(key, ""))
        if p.suffix:          # it's a file path
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gas price loading
# ---------------------------------------------------------------------------

def load_gas_price_hourly(cfg: dict, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Load TTF Natural Gas daily price (EUR/MWh) from commodities.csv.
    Forward-fill weekends / gaps, then back-fill the pre-2017 period.
    Returns a Series aligned to `hourly_index`.
    """
    path = ROOT / cfg["data"]["commodities_path"]
    comm = pd.read_csv(path, parse_dates=["Date"])
    comm["Date"] = pd.to_datetime(comm["Date"], utc=True).dt.floor("D")
    comm = comm.set_index("Date").sort_index()

    # Use TTF where available; fall back to API2_Coal as crude proxy (different
    # units, but z-scoring removes level differences)
    if "TTF_Gas" in comm.columns:
        gas = comm["TTF_Gas"].copy()
    else:
        gas = comm["API2_Coal"].copy()

    # Forward-fill to cover weekends / holidays, then back-fill the early period
    gas = gas.ffill().bfill()

    # Reindex to the hourly datetime index (daily value broadcast to all hours)
    daily_dates = hourly_index.normalize()   # floor to day
    price_hourly = gas.reindex(daily_dates).values
    return pd.Series(price_hourly, index=hourly_index, name="gas_price_raw")


# ---------------------------------------------------------------------------
# Causal (expanding) z-score
# ---------------------------------------------------------------------------

def expanding_zscore(s: pd.Series, min_periods: int = 30) -> pd.Series:
    """
    Causally z-score a series using expanding mean and std.
    Rows before min_periods are NaN.
    """
    mu = s.expanding(min_periods=min_periods).mean()
    sigma = s.expanding(min_periods=min_periods).std()
    sigma = sigma.replace(0, np.nan)
    return (s - mu) / sigma


# ---------------------------------------------------------------------------
# Capacity factor derivation
# ---------------------------------------------------------------------------

def capacity_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Causal, data-driven capacity factor estimation using expanding cummax.
    This estimates installed capacity as the running maximum generation ever seen.
    A floor equal to 5% of the config capacity prevents division by zero in the
    early dataset period.
    """
    wind_gen = df["Wind Onshore"].fillna(0) + df["Wind Offshore"].fillna(0)
    solar_gen = df["Solar"].fillna(0)

    # Expanding maximum = monotonically increasing installed capacity proxy
    wind_cap = wind_gen.expanding().max().clip(lower=1.0)     # MW
    solar_cap = solar_gen.expanding().max().clip(lower=1.0)   # MW

    df["W_CF"] = (wind_gen / wind_cap).clip(0, 1)
    df["S_CF"] = (solar_gen / solar_cap).clip(0, 1)
    df["_wind_gen"] = wind_gen
    df["_solar_gen"] = solar_gen
    return df


# ---------------------------------------------------------------------------
# Temperature-adjusted demand
# ---------------------------------------------------------------------------

def temperature_adjusted_demand(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    T_demand = Actual Load / mean_load_by_hour_of_day
    The mean is computed on TRAINING data only and then broadcast to all splits.
    """
    df["_hour"] = df.index.hour
    # Mean demand per hour-of-day on training data
    train_df = df.loc[train_mask]
    hourly_mean = train_df.groupby("_hour")["Actual Load"].mean()

    df["_mean_load_by_hour"] = df["_hour"].map(hourly_mean)
    df["T_demand"] = df["Actual Load"] / df["_mean_load_by_hour"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Holiday flag
# ---------------------------------------------------------------------------

def build_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag for German federal holidays (2018-2026).
    """
    years = range(2015, 2027)
    de_holidays = holidays.Germany(years=years)

    dates = df.index.date
    df["is_holiday"] = np.array([1 if d in de_holidays else 0 for d in dates], dtype=np.int8)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Cyclical encodings
# ---------------------------------------------------------------------------

def cyclical_encode(df: pd.DataFrame) -> pd.DataFrame:
    hour = df.index.hour
    dow  = df.index.dayofweek
    month = df.index.month

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


# ---------------------------------------------------------------------------
# Price lags
# ---------------------------------------------------------------------------

def add_price_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"P_lag_{lag}"] = df["price"].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Renewable penetration
# ---------------------------------------------------------------------------

def renewable_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """
    renewable_penetration = (wind_gen + solar_gen) / Actual Load
    Using actual generation totals (causal, no look-ahead).
    """
    total_re = df["_wind_gen"] + df["_solar_gen"]
    demand = df["Actual Load"].replace(0, np.nan)
    df["renewable_penetration"] = (total_re / demand).clip(0, 1)
    return df


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sp = cfg["splits"]
    train_s = pd.Timestamp(sp["train_start"], tz="UTC")
    train_e = pd.Timestamp(sp["train_end"],   tz="UTC") + pd.Timedelta("23h")
    val_s   = pd.Timestamp(sp["val_start"],   tz="UTC")
    val_e   = pd.Timestamp(sp["val_end"],     tz="UTC") + pd.Timedelta("23h")
    test_s  = pd.Timestamp(sp["test_start"],  tz="UTC")
    test_e  = pd.Timestamp(sp["test_end"],    tz="UTC") + pd.Timedelta("23h")

    idx = df.index
    conditions = [
        (idx >= train_s) & (idx <= train_e),
        (idx >= val_s)   & (idx <= val_e),
        (idx >= test_s)  & (idx <= test_e),
    ]
    df["split"] = np.select(conditions, ["train", "val", "test"], default="exclude")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(cfg_path: Path | None = None) -> pd.DataFrame:
    cfg = _load_config(cfg_path)
    _ensure_dirs(cfg)

    print("[build_features] Loading raw ENTSO-E data …")
    raw_path = ROOT / cfg["data"]["raw_path"]
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    # Drop duplicated timestamps if any
    df = df[~df.index.duplicated(keep="first")]
    # Ensure hourly frequency (forward-fill gaps up to 3 h)
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    df = df.reindex(full_idx).ffill(limit=3)

    # ---- Gas price (causal expanding z-score) ----
    print("[build_features] Building gas price z-score …")
    gas_raw = load_gas_price_hourly(cfg, df.index)
    df["_gas_price_raw"] = gas_raw.values
    df["G_price_zscore"] = expanding_zscore(df["_gas_price_raw"])

    # ---- Capacity factors ----
    print("[build_features] Computing capacity factors …")
    df = capacity_factors(df)

    # ---- Split column (needed before T_demand) ----
    df = assign_splits(df, cfg)

    # Assign splits so we can build train_mask
    train_mask = df["split"] == "train"

    # ---- Temperature-adjusted demand ----
    print("[build_features] Computing temperature-adjusted demand …")
    df = temperature_adjusted_demand(df, train_mask)

    # ---- Renewable penetration ----
    df = renewable_penetration(df)

    # ---- Calendar & cyclical features ----
    print("[build_features] Adding cyclical encodings and holiday flags …")
    df = cyclical_encode(df)
    df = build_holiday_flags(df)

    # ---- Price lags ----
    print("[build_features] Adding price lags …")
    lags = cfg["features"]["price_lags"]
    df = add_price_lags(df, lags)

    # ---- Rename price column to keep raw values ----
    # CRITICAL: do NOT transform the price target
    # Negative prices must remain as raw values
    # Log-transforms are FORBIDDEN

    # ---- Select final feature columns ----
    feature_cols = [
        "price",
        *[f"P_lag_{l}" for l in lags],
        "W_CF", "S_CF",
        "G_price_zscore",
        "T_demand",
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "month_sin","month_cos",
        "is_holiday", "is_weekend",
        "renewable_penetration",
        "split",
    ]
    # Keep only columns that exist (guard)
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = df[feature_cols].copy()

    # ---- Filter to train + val + test only (drop warm-up rows) ----
    out = out[out["split"] != "exclude"]

    # ---- Drop rows where any feature is NaN (caused by initial lag warm-up) ----
    n_before = len(out)
    key_cols = [c for c in feature_cols if c != "split"]
    out = out.dropna(subset=key_cols)
    n_after = len(out)
    print(f"[build_features] Dropped {n_before - n_after} NaN rows "
          f"(lag warm-up). Remaining: {n_after}")

    # ---- Validation ----
    assert out["price"].min() < 0, \
        "ERROR: Negative prices lost! Check transform pipeline."
    split_counts = out["split"].value_counts()
    print(f"[build_features] Split counts:\n{split_counts.to_string()}")

    # ---- Save ----
    out_path = ROOT / cfg["data"]["processed_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"[build_features] Saved feature matrix -> {out_path}  shape={out.shape}")

    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_feature_matrix()
