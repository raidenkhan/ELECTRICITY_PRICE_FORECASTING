#!/usr/bin/env python
# run_baselines.py — Baseline model evaluation for LF-GP-NRF comparison
# Baselines: (1) LEAR, (2) LightGBM per regime (existing pipeline), (3) Naive seasonal
# Usage: python src/experiments/run_baselines.py

import json
import os
import sys
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd
import torch  # noqa: F401 — kept for parity with the wider pipeline environment

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.preprocess import EPFPreprocessor
from src.experiments.evaluation import (
    compare_models,
    compute_crps_samples,
    compute_pinball,
    compute_winkler_score,
)

# ---------------------------------------------------------------------------
# Quantile levels shared across all baselines
# ---------------------------------------------------------------------------
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


# ===========================================================================
# Baseline 1 — LEAR (Lasso-Estimated AutoRegressive)
# ===========================================================================


class LEARBaseline:
    """LEAR electricity price forecasting baseline.

    Standard EPF benchmark from Lago et al. (Applied Energy 2021).
    Trains one LassoCV model per hour-of-day (24 models total), using
    calendar features, price lags, and supply/demand proxies.
    """

    def __init__(self):
        self.models: dict = {}  # {hour: fitted LassoCV}
        self.scalers: dict = {}  # {hour: StandardScaler for X}
        self.train_residuals: dict = {}  # {hour: np.ndarray} for CRPS σ estimation
        self.train_X: dict = {}  # {hour: X_train} retained for QuantileRegressor
        self.train_y: dict = {}  # {hour: y_train}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
        """Return column if present, else zeros."""
        if col in df.columns:
            return df[col].fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index)

    def build_features(self, df: pd.DataFrame):
        """Build LEAR feature matrix and target vector.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed hourly dataframe (output of EPFPreprocessor.process).

        Returns
        -------
        X : np.ndarray, shape (N, n_features)
        y : np.ndarray, shape (N,)
        """
        df = df.copy()

        # --- Price lags (same-hour previous days) ---
        # These are already in df if preprocess.py was run; otherwise compute them.
        for lag in [24, 48, 72, 168]:
            col = f"price_lag_{lag}"
            if col not in df.columns:
                df[col] = df["price"].shift(lag)

        # 7-day rolling mean/std of price, lagged by 24 h to avoid leakage
        rolling_price = df["price"].shift(24).rolling(window=24 * 7, min_periods=48)
        df["price_roll7d_mean"] = rolling_price.mean()
        df["price_roll7d_std"] = rolling_price.std().fillna(1.0)

        # --- Load / RE forecasts (raw columns if present) ---
        load_fc = LEARBaseline._safe_get(df, "load_forecast")
        solar_fc = LEARBaseline._safe_get(df, "solar_forecast")
        wind_on = LEARBaseline._safe_get(df, "wind_onshore_forecast")
        wind_off = LEARBaseline._safe_get(df, "wind_offshore_forecast")
        re_fc = solar_fc + wind_on + wind_off

        # --- Calendar encodings ---
        idx = pd.DatetimeIndex(df.index)
        hour_sin = np.sin(2 * np.pi * idx.hour / 24)
        hour_cos = np.cos(2 * np.pi * idx.hour / 24)
        dow_sin = np.sin(2 * np.pi * idx.dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * idx.dayofweek / 7)
        month = idx.month / 12.0

        feature_matrix = np.column_stack(
            [
                df["price_lag_24"].fillna(0).values,
                df["price_lag_48"].fillna(0).values,
                df["price_lag_72"].fillna(0).values,
                df["price_lag_168"].fillna(0).values,
                load_fc.values,
                re_fc.values,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
                month,
                LEARBaseline._safe_get(df, "gas_crisis_regime").values,
                df["price_roll7d_mean"].fillna(0).values,
                df["price_roll7d_std"].fillna(1.0).values,
            ]
        )

        y = df["price"].values
        return feature_matrix, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit one LassoCV per hour on the training slice.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data (preprocessed).
        """
        X_all, y_all = self.build_features(df_train)
        idx = pd.DatetimeIndex(df_train.index)

        print(f"  [LEAR] Fitting 24 LassoCV models on {len(df_train):,} training rows…")
        for h in range(24):
            mask = idx.hour == h
            X_h = X_all[mask]
            y_h = y_all[mask]

            if len(X_h) < 20:
                # Fallback: just store mean
                self.models[h] = None
                self.train_X[h] = X_h
                self.train_y[h] = y_h
                self.train_residuals[h] = np.zeros(len(y_h))
                continue

            scaler = StandardScaler()
            X_h_sc = scaler.fit_transform(X_h)
            self.scalers[h] = scaler

            model = LassoCV(cv=5, max_iter=5000, n_jobs=-1)
            model.fit(X_h_sc, y_h)
            self.models[h] = model

            # Store residuals for CRPS Gaussian approximation
            resid = y_h - model.predict(X_h_sc)
            self.train_residuals[h] = resid
            self.train_X[h] = X_h
            self.train_y[h] = y_h

        print("  [LEAR] Training complete.")

    # ------------------------------------------------------------------
    # Point prediction
    # ------------------------------------------------------------------

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return flat hourly point forecasts, shape (N,).

        Parameters
        ----------
        df_test : pd.DataFrame
            Test data (preprocessed).

        Returns
        -------
        np.ndarray, shape (N,)
        """
        X_all, _ = self.build_features(df_test)
        idx = pd.DatetimeIndex(df_test.index)
        preds = np.zeros(len(df_test))

        for h in range(24):
            mask = idx.hour == h
            if not mask.any():
                continue
            X_h = X_all[mask]
            model = self.models.get(h)
            if model is None:
                # Fallback: use training mean for this hour
                preds[mask] = self.train_y.get(h, np.array([0.0])).mean()
            else:
                scaler = self.scalers[h]
                X_h_sc = scaler.transform(X_h)
                preds[mask] = model.predict(X_h_sc)

        return preds  # (N,)

    # ------------------------------------------------------------------
    # Probabilistic prediction — quantiles
    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        df_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Predict quantiles for each hour using per-quantile QuantileRegressor.

        For each quantile τ and each hour h, a QuantileRegressor is fit on the
        training set for that hour, then evaluated on the test set.

        Parameters
        ----------
        df_test : pd.DataFrame
        quantiles : list of float, optional

        Returns
        -------
        dict {tau: np.ndarray (N,)}  — flat hourly arrays
        """
        if quantiles is None:
            quantiles = QUANTILE_LEVELS

        X_test_all, _ = self.build_features(df_test)
        idx_test = pd.DatetimeIndex(df_test.index)

        # Result container
        result = {tau: np.zeros(len(df_test)) for tau in quantiles}

        for h in range(24):
            mask_test = idx_test.hour == h
            if not mask_test.any():
                continue

            X_h_train = self.train_X.get(h)
            y_h_train = self.train_y.get(h)
            X_h_test = X_test_all[mask_test]

            if X_h_train is None or len(X_h_train) < 20:
                # Fallback: empirical quantiles of training prices
                for tau in quantiles:
                    result[tau][mask_test] = np.nanpercentile(
                        y_h_train if y_h_train is not None else [0.0],
                        tau * 100,
                    )
                continue

            scaler = self.scalers.get(h)
            if scaler is not None:
                X_h_train_sc = scaler.transform(X_h_train)
                X_h_test_sc = scaler.transform(X_h_test)
            else:
                X_h_train_sc = X_h_train
                X_h_test_sc = X_h_test

            for tau in quantiles:
                qr = QuantileRegressor(quantile=tau, alpha=0.01, solver="highs")
                try:
                    qr.fit(X_h_train_sc, y_h_train)
                    result[tau][mask_test] = qr.predict(X_h_test_sc)
                except Exception:
                    # Fallback to empirical quantile
                    result[tau][mask_test] = np.nanpercentile(y_h_train, tau * 100)

        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile_preds: dict,
    ) -> dict:
        """Compute MAE, RMSE, CRPS, pinball, Winkler 90%.

        Parameters
        ----------
        y_true : np.ndarray, shape (N_days, 24)
        y_pred : np.ndarray, shape (N_days, 24)
        quantile_preds : dict {tau: np.ndarray (N_days, 24) or (N,)}

        Returns
        -------
        dict of scalar metric values.
        """
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        N = len(y_true_flat)

        metrics = {}
        metrics["mae"] = float(mean_absolute_error(y_true_flat, y_pred_flat))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))

        # MAE on spike hours
        spike_mask = y_true_flat > 200.0
        neg_mask = y_true_flat < 0.0
        if spike_mask.any():
            metrics["mae_spike"] = float(
                np.mean(np.abs(y_true_flat[spike_mask] - y_pred_flat[spike_mask]))
            )
        else:
            metrics["mae_spike"] = float("nan")

        # CRPS — Gaussian approximation per hour-of-day using training σ
        # Build (N, 1000) samples from N(point, σ_h)
        np.random.seed(42)
        idx_hour = np.tile(np.arange(24), len(y_pred))[:N]
        samples_matrix = np.zeros((N, 1000))
        for h in range(24):
            mask_h = idx_hour == h
            resid = self.train_residuals.get(h, np.array([0.0]))
            sigma_h = max(float(np.std(resid)), 1e-3)
            n_h = int(mask_h.sum())
            if n_h > 0:
                mu_h = y_pred_flat[mask_h]
                noise = np.random.randn(n_h, 1000) * sigma_h
                samples_matrix[mask_h] = mu_h[:, None] + noise

        crps_all = compute_crps_samples(y_true_flat, samples_matrix)
        metrics["crps_mean"] = float(np.mean(crps_all))
        metrics["crps_spike"] = (
            float(np.mean(crps_all[spike_mask])) if spike_mask.any() else float("nan")
        )
        metrics["crps_negative"] = (
            float(np.mean(crps_all[neg_mask])) if neg_mask.any() else float("nan")
        )
        metrics["crps_base"] = float(
            np.mean(crps_all[(y_true_flat >= 0.0) & (y_true_flat <= 200.0)])
        )

        # Flatten quantile preds to (N,)
        q_flat = {}
        for tau, arr in quantile_preds.items():
            arr = np.asarray(arr)
            q_flat[tau] = arr.reshape(-1)[:N]

        # Pinball
        pinball = compute_pinball(y_true_flat, q_flat, QUANTILE_LEVELS)
        for tau, val in pinball.items():
            metrics[f"pinball_{int(round(tau * 100)):03d}"] = val

        # Winkler 90% (using q5 / q95)
        q05 = q_flat.get(0.05, np.zeros(N))
        q95 = q_flat.get(0.95, np.zeros(N))
        metrics["winkler_90"] = compute_winkler_score(y_true_flat, q05, q95, alpha=0.10)

        # Empirical coverage
        q25 = q_flat.get(0.25, np.zeros(N))
        q75 = q_flat.get(0.75, np.zeros(N))
        metrics["coverage_90"] = float(
            np.mean((y_true_flat >= q05) & (y_true_flat <= q95))
        )
        metrics["coverage_50"] = float(
            np.mean((y_true_flat >= q25) & (y_true_flat <= q75))
        )

        return metrics


# ===========================================================================
# Baseline 2 — LightGBM Probabilistic (per-quantile, no per-regime split)
# ===========================================================================


class LGBMBaseline:
    """Probabilistic LightGBM baseline using the 'golden features' from benchmark.py.

    One LightGBM quantile regressor per quantile level.  The model captures
    hour-of-day and day-of-week effects through the cyclic feature columns, so
    a single model per quantile covers all hours.
    """

    # Same golden feature list used in benchmark.py / EPFEvaluator
    GOLDEN_FEATURES = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "Residual_Load",
        "total_re_penetration",
        "dark_doldrums",
        "12h_price_range",
        "3h_neg_streak",
        "price_lag_24",
        "price_lag_48",
        "price_lag_168",
        "price_rolling48_lag1",
        "gas_crisis_regime",
    ]

    def __init__(self):
        self.models: dict = {}  # {tau: lgb.LGBMRegressor}
        self.features_used: list = []
        self.scaler = None
        self._scaling_constant = 50.0  # must match EPFPreprocessor

    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame):
        """Return (X, y) using golden features that exist in df.

        Target is ``price_asinh``; X uses all available golden features plus
        a 48-hour rolling mean of price_lag_24.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : pd.DataFrame
        y : np.ndarray
        """
        df = df.copy()

        # Derive rolling 48h mean of price_lag_24 (1-lag to avoid leakage)
        if "price_rolling48_lag1" not in df.columns and "price_lag_24" in df.columns:
            df["price_rolling48_lag1"] = (
                df["price_lag_24"].rolling(window=48, min_periods=1).mean()
            )

        available = [c for c in self.GOLDEN_FEATURES if c in df.columns]
        X = df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Target: asinh-transformed price
        if "price_asinh" in df.columns:
            y = df["price_asinh"].values
        else:
            y = np.arcsinh(df["price"].values / self._scaling_constant)

        return X, y

    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit one quantile LGBMRegressor per quantile level.

        Parameters
        ----------
        df_train : pd.DataFrame
        """
        X_train, y_train = self.build_features(df_train)
        self.features_used = list(X_train.columns)

        lgb_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
            min_child_samples=20,
        )

        print(
            f"  [LightGBM] Fitting {len(QUANTILE_LEVELS)} quantile models on "
            f"{len(X_train):,} rows, {len(self.features_used)} features…"
        )

        for tau in QUANTILE_LEVELS:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=tau,
                **lgb_params,
            )
            model.fit(X_train, y_train)
            self.models[tau] = model

        print("  [LightGBM] Training complete.")

    # ------------------------------------------------------------------

    def predict_quantiles(self, df_test: pd.DataFrame) -> dict:
        """Predict quantiles on test data; inverse-asinh back to EUR/MWh.

        Parameters
        ----------
        df_test : pd.DataFrame

        Returns
        -------
        dict {tau: np.ndarray (N_test,)} in EUR/MWh
        """
        X_test, _ = self.build_features(df_test)

        # Align columns in case test has extra/missing columns
        X_test = X_test.reindex(columns=self.features_used, fill_value=0.0)

        result = {}
        for tau, model in self.models.items():
            pred_asinh = model.predict(X_test)
            # Inverse asinh: price = 50 * sinh(pred_asinh)
            pred_eur = self._scaling_constant * np.sinh(pred_asinh)
            result[tau] = pred_eur  # shape (N,)

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def reshape_to_windows(arr: np.ndarray, n_hours: int = 24) -> np.ndarray:
        """Reshape flat hourly array to (N_days, 24), dropping incomplete days.

        Parameters
        ----------
        arr : np.ndarray, shape (N,)
        n_hours : int

        Returns
        -------
        np.ndarray, shape (N_days, n_hours)
        """
        arr = np.asarray(arr)
        n_complete = (len(arr) // n_hours) * n_hours
        return arr[:n_complete].reshape(-1, n_hours)

    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true_flat: np.ndarray,
        quantile_preds_flat: dict,
    ) -> dict:
        """Compute full metric suite given flat hourly arrays.

        Parameters
        ----------
        y_true_flat : np.ndarray, shape (N,)
        quantile_preds_flat : dict {tau: np.ndarray (N,)}

        Returns
        -------
        dict of scalar metric values.
        """
        y_true_flat = np.asarray(y_true_flat, dtype=np.float64)
        N = len(y_true_flat)

        # Align lengths
        q_flat = {}
        for tau, arr in quantile_preds_flat.items():
            arr = np.asarray(arr).reshape(-1)
            q_flat[tau] = arr[:N]

        # Point forecast = median
        point_pred = q_flat.get(0.50, np.zeros(N))

        metrics = {}
        metrics["mae"] = float(mean_absolute_error(y_true_flat, point_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_flat, point_pred)))

        spike_mask = y_true_flat > 200.0
        neg_mask = y_true_flat < 0.0
        if spike_mask.any():
            metrics["mae_spike"] = float(
                np.mean(np.abs(y_true_flat[spike_mask] - point_pred[spike_mask]))
            )
        else:
            metrics["mae_spike"] = float("nan")

        # CRPS — construct pseudo-samples by interpolating from quantile predictions
        # Build (N, n_quantiles) matrix then draw samples via inverse-CDF interpolation
        q_levels = np.array(QUANTILE_LEVELS, dtype=np.float64)
        q_matrix = np.column_stack([q_flat[tau] for tau in QUANTILE_LEVELS])  # (N, Q)

        np.random.seed(42)
        uniform_draws = np.random.uniform(0, 1, (N, 1000))  # (N, 1000)
        # Interpolate: for each observation, map U→quantile
        samples_matrix = np.zeros((N, 1000))
        for i in range(N):
            samples_matrix[i] = np.interp(uniform_draws[i], q_levels, q_matrix[i])

        crps_all = compute_crps_samples(y_true_flat, samples_matrix)
        metrics["crps_mean"] = float(np.mean(crps_all))
        metrics["crps_spike"] = (
            float(np.mean(crps_all[spike_mask])) if spike_mask.any() else float("nan")
        )
        metrics["crps_negative"] = (
            float(np.mean(crps_all[neg_mask])) if neg_mask.any() else float("nan")
        )
        metrics["crps_base"] = float(
            np.mean(crps_all[(y_true_flat >= 0.0) & (y_true_flat <= 200.0)])
        )

        # Pinball
        pinball = compute_pinball(y_true_flat, q_flat, QUANTILE_LEVELS)
        for tau, val in pinball.items():
            metrics[f"pinball_{int(round(tau * 100)):03d}"] = val

        # Winkler 90%
        q05 = q_flat.get(0.05, np.zeros(N))
        q95 = q_flat.get(0.95, np.zeros(N))
        metrics["winkler_90"] = compute_winkler_score(y_true_flat, q05, q95, alpha=0.10)

        # Coverage
        q25 = q_flat.get(0.25, np.zeros(N))
        q75 = q_flat.get(0.75, np.zeros(N))
        metrics["coverage_90"] = float(
            np.mean((y_true_flat >= q05) & (y_true_flat <= q95))
        )
        metrics["coverage_50"] = float(
            np.mean((y_true_flat >= q25) & (y_true_flat <= q75))
        )

        return metrics


# ===========================================================================
# Baseline 3 — Naive Seasonal (last-week same-hour)
# ===========================================================================


class NaiveSeasonalBaseline:
    """Naive seasonal (weekly) baseline.

    Point forecast: price_t = price_{t-168} (same hour, one week ago).
    Quantile forecast: empirical distribution of training errors stratified
    by hour-of-day and day-of-week, centred on the seasonal point forecast.
    """

    def __init__(self):
        # error_quantiles[h][dow][tau] = empirical τ-quantile of training errors
        self.error_quantiles: dict = {}
        self.hour_dow_mean: dict = {}  # mean error per (h, dow) for bias correction

    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> None:
        """Compute empirical error distribution from training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Preprocessed training frame (must contain ``price`` and
            ``price_lag_168`` columns after preprocessing).
        """
        df = df_train.copy()
        if "price_lag_168" not in df.columns:
            df["price_lag_168"] = df["price"].shift(168)

        df = df.dropna(subset=["price", "price_lag_168"])
        df["naive_error"] = df["price"] - df["price_lag_168"]

        idx = pd.DatetimeIndex(df.index)
        df["_hour"] = idx.hour
        df["_dow"] = idx.dayofweek

        print(f"  [Naive] Computing empirical error quantiles from {len(df):,} rows…")

        self.error_quantiles = {}
        self.hour_dow_mean = {}

        for h in range(24):
            self.error_quantiles[h] = {}
            self.hour_dow_mean[h] = {}
            for dow in range(7):
                mask = (df["_hour"] == h) & (df["_dow"] == dow)
                errors = df.loc[mask, "naive_error"].values
                if len(errors) < 5:
                    # Fall back to hour-level
                    errors = df.loc[df["_hour"] == h, "naive_error"].values
                if len(errors) < 2:
                    errors = df["naive_error"].values

                self.hour_dow_mean[h][dow] = float(np.mean(errors))
                q_dict = {}
                for tau in QUANTILE_LEVELS:
                    q_dict[tau] = float(np.percentile(errors, tau * 100))
                self.error_quantiles[h][dow] = q_dict

        print("  [Naive] Fitting complete.")

    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        df_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Return quantile predictions centred on seasonal point forecast.

        Parameters
        ----------
        df_test : pd.DataFrame
        quantiles : list of float, optional

        Returns
        -------
        dict {tau: np.ndarray (N,)} in EUR/MWh
        """
        if quantiles is None:
            quantiles = QUANTILE_LEVELS

        df = df_test.copy()
        if "price_lag_168" not in df.columns:
            df["price_lag_168"] = df["price"].shift(168)

        idx = pd.DatetimeIndex(df.index)
        n = len(df)
        result = {tau: np.zeros(n) for tau in quantiles}

        lag168 = (
            df["price_lag_168"]
            .fillna(df["price"].mean() if "price" in df.columns else 50.0)
            .values
        )
        hours = idx.hour
        dows = idx.dayofweek

        for i in range(n):
            h = int(hours[i])
            dow = int(dows[i])
            seasonal_pt = lag168[i]

            # Error quantiles for this (h, dow) stratum
            q_dict = self.error_quantiles.get(h, {}).get(dow, {})
            mean_err = self.hour_dow_mean.get(h, {}).get(dow, 0.0)

            for tau in quantiles:
                err_q = q_dict.get(tau, 0.0)
                # Centre on point forecast and shift by the quantile of errors
                # Remove mean bias, add quantile offset
                result[tau][i] = seasonal_pt + (err_q - mean_err) + mean_err

        return result

    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile_preds: dict,
    ) -> dict:
        """Compute MAE, RMSE, CRPS, pinball, Winkler 90%.

        Parameters
        ----------
        y_true : np.ndarray, shape (N_days, 24) or (N,)
        y_pred : np.ndarray, shape (N_days, 24) or (N,)
        quantile_preds : dict {tau: np.ndarray}

        Returns
        -------
        dict of scalar metrics.
        """
        y_true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        N = len(y_true_flat)

        q_flat = {}
        for tau, arr in quantile_preds.items():
            arr = np.asarray(arr).reshape(-1)
            q_flat[tau] = arr[:N]

        metrics = {}
        metrics["mae"] = float(mean_absolute_error(y_true_flat, y_pred_flat))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))

        spike_mask = y_true_flat > 200.0
        neg_mask = y_true_flat < 0.0
        if spike_mask.any():
            metrics["mae_spike"] = float(
                np.mean(np.abs(y_true_flat[spike_mask] - y_pred_flat[spike_mask]))
            )
        else:
            metrics["mae_spike"] = float("nan")

        # CRPS via Gaussian approximation from residuals between seasonal and actual
        # σ is estimated from the spread of the 5th–95th quantile range ÷ 3.29
        q05 = q_flat.get(0.05, np.zeros(N))
        q95 = q_flat.get(0.95, np.zeros(N))
        sigma_approx = np.maximum((q95 - q05) / 3.29, 1e-3)  # (N,) local spread

        np.random.seed(42)
        noise = np.random.randn(N, 1000) * sigma_approx[:, None]
        samples_matrix = y_pred_flat[:, None] + noise

        crps_all = compute_crps_samples(y_true_flat, samples_matrix)
        metrics["crps_mean"] = float(np.mean(crps_all))
        metrics["crps_spike"] = (
            float(np.mean(crps_all[spike_mask])) if spike_mask.any() else float("nan")
        )
        metrics["crps_negative"] = (
            float(np.mean(crps_all[neg_mask])) if neg_mask.any() else float("nan")
        )
        metrics["crps_base"] = float(
            np.mean(crps_all[(y_true_flat >= 0.0) & (y_true_flat <= 200.0)])
        )

        # Pinball
        pinball = compute_pinball(y_true_flat, q_flat, QUANTILE_LEVELS)
        for tau, val in pinball.items():
            metrics[f"pinball_{int(round(tau * 100)):03d}"] = val

        # Winkler 90%
        metrics["winkler_90"] = compute_winkler_score(y_true_flat, q05, q95, alpha=0.10)

        # Coverage
        q25 = q_flat.get(0.25, np.zeros(N))
        q75 = q_flat.get(0.75, np.zeros(N))
        metrics["coverage_90"] = float(
            np.mean((y_true_flat >= q05) & (y_true_flat <= q95))
        )
        metrics["coverage_50"] = float(
            np.mean((y_true_flat >= q25) & (y_true_flat <= q75))
        )

        return metrics


# ===========================================================================
# Main
# ===========================================================================


def main():
    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    DATA_DIR = "data/raw"
    OUT_DIR = "outputs"
    TABLES_DIR = os.path.join(OUT_DIR, "tables")
    FIGS_DIR = os.path.join(OUT_DIR, "figures")

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and preprocess data
    # -----------------------------------------------------------------------
    print("Loading raw data…")
    preprocessor = EPFPreprocessor()
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "Germany_master_entsoe_2015_2026.csv"))
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw = df_raw.set_index("date").sort_index().fillna(0)

    print("Preprocessing…")
    df = preprocessor.process(df_raw).dropna()

    # -----------------------------------------------------------------------
    # Time splits — match LF-GP-NRF exactly
    # -----------------------------------------------------------------------
    df_train = df.loc[:"2022-12-31"]
    df_test = df.loc["2024-01-01":"2024-12-31"]
    print(f"Train: {len(df_train):,}h  |  Test: {len(df_test):,}h")

    # -----------------------------------------------------------------------
    # y_true for test — align to complete 24 h windows
    # -----------------------------------------------------------------------
    y_true_flat = df_test["price"].values  # EUR/MWh
    n_complete = (len(y_true_flat) // 24) * 24
    y_true_flat = y_true_flat[:n_complete]
    y_true_2d = y_true_flat.reshape(-1, 24)  # (N_days, 24)
    print(f"Test windows: {len(y_true_2d)}")

    # Slice of test df aligned to complete windows
    df_test_aligned = df_test.iloc[:n_complete]

    all_metrics: dict = {}

    # -----------------------------------------------------------------------
    # Baseline 1 — LEAR
    # -----------------------------------------------------------------------
    print("\n=== LEAR Baseline ===")
    lear = LEARBaseline()
    lear.fit(df_train)

    lear_point_flat = lear.predict(df_test_aligned)  # (N,)
    lear_quantiles = lear.predict_quantiles(df_test_aligned)  # {tau: (N,)}

    # Reshape point to 2D (N_days, 24)
    lear_point_2d = lear_point_flat.reshape(-1, 24)

    # Reshape quantile arrays to (N_days, 24) for compatibility with compute_metrics
    lear_quantiles_2d = {
        tau: arr.reshape(-1, 24) for tau, arr in lear_quantiles.items()
    }

    lear_metrics = lear.compute_metrics(y_true_2d, lear_point_2d, lear_quantiles_2d)
    all_metrics["LEAR"] = lear_metrics
    print(
        f"  MAE={lear_metrics['mae']:.2f}  "
        f"RMSE={lear_metrics['rmse']:.2f}  "
        f"CRPS={lear_metrics['crps_mean']:.4f}  "
        f"Winkler90={lear_metrics['winkler_90']:.2f}"
    )

    # -----------------------------------------------------------------------
    # Baseline 2 — LightGBM
    # -----------------------------------------------------------------------
    print("\n=== LightGBM Baseline ===")
    lgbm = LGBMBaseline()
    lgbm.fit(df_train)

    lgbm_quantiles = lgbm.predict_quantiles(df_test)  # {tau: (N_test,)}

    # Truncate to aligned length
    lgbm_quantiles_aligned = {
        tau: arr[:n_complete] for tau, arr in lgbm_quantiles.items()
    }

    lgbm_metrics = lgbm.compute_metrics(y_true_flat, lgbm_quantiles_aligned)
    all_metrics["LightGBM"] = lgbm_metrics
    print(
        f"  MAE={lgbm_metrics['mae']:.2f}  "
        f"RMSE={lgbm_metrics['rmse']:.2f}  "
        f"CRPS={lgbm_metrics['crps_mean']:.4f}  "
        f"Winkler90={lgbm_metrics['winkler_90']:.2f}"
    )

    # -----------------------------------------------------------------------
    # Baseline 3 — Naive Seasonal
    # -----------------------------------------------------------------------
    print("\n=== Naive Seasonal Baseline ===")
    naive = NaiveSeasonalBaseline()
    naive.fit(df_train)

    naive_quantiles_flat = naive.predict_quantiles(
        df_test_aligned,
        quantiles=QUANTILE_LEVELS,
    )  # {tau: (N,)}

    # Median as point forecast — reshape to (N_days, 24)
    naive_point_2d = naive_quantiles_flat[0.50].reshape(-1, 24)

    # Reshape quantile arrays to (N_days, 24)
    naive_quantiles_2d = {
        tau: arr.reshape(-1, 24) for tau, arr in naive_quantiles_flat.items()
    }

    naive_metrics = naive.compute_metrics(y_true_2d, naive_point_2d, naive_quantiles_2d)
    all_metrics["Naive Seasonal"] = naive_metrics
    print(
        f"  MAE={naive_metrics['mae']:.2f}  "
        f"RMSE={naive_metrics['rmse']:.2f}  "
        f"CRPS={naive_metrics['crps_mean']:.4f}  "
        f"Winkler90={naive_metrics['winkler_90']:.2f}"
    )

    # -----------------------------------------------------------------------
    # Try to load LF-GP-NRF results if available
    # -----------------------------------------------------------------------
    lfgpnrf_path = os.path.join(TABLES_DIR, "results_LF-GP-NRF.csv")
    if os.path.exists(lfgpnrf_path):
        try:
            lfgpnrf_df = pd.read_csv(lfgpnrf_path, index_col=0)
            all_metrics["LF-GP-NRF"] = lfgpnrf_df["Value"].to_dict()
            print(f"\nLF-GP-NRF results loaded from {lfgpnrf_path}")
        except Exception as exc:
            print(f"\n[WARNING] Could not parse {lfgpnrf_path}: {exc}")

    # -----------------------------------------------------------------------
    # Save baseline metrics as JSON
    # -----------------------------------------------------------------------
    baseline_json_path = os.path.join(TABLES_DIR, "baseline_metrics.json")
    with open(baseline_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                model: {
                    mk: (
                        float(mv)
                        if not (isinstance(mv, float) and np.isnan(mv))
                        else None
                    )
                    for mk, mv in metrics.items()
                }
                for model, metrics in all_metrics.items()
            },
            fh,
            indent=2,
        )
    print(f"\nBaseline metrics saved → {baseline_json_path}")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics).T
    comparison_csv = os.path.join(TABLES_DIR, "model_comparison.csv")
    metrics_df.to_csv(comparison_csv, float_format="%.4f")

    display_cols = [
        c
        for c in [
            "mae",
            "rmse",
            "mae_spike",
            "crps_mean",
            "crps_spike",
            "crps_negative",
            "winkler_90",
            "coverage_90",
            "coverage_50",
        ]
        if c in metrics_df.columns
    ]

    print("\n=== Model Comparison ===")
    print(metrics_df[display_cols].to_string())

    # -----------------------------------------------------------------------
    # Comparison plot via compare_models
    # -----------------------------------------------------------------------
    if len(all_metrics) > 1:
        try:
            compare_models(all_metrics, FIGS_DIR)
        except Exception as exc:
            print(f"[WARNING] compare_models raised an error: {exc}")
            # Fallback: simple bar chart of MAE and CRPS
            _fallback_comparison_plot(all_metrics, FIGS_DIR)

    print(f"\nBaseline evaluation complete. Results saved to {TABLES_DIR}/")


# ---------------------------------------------------------------------------
# Fallback comparison plot (used if compare_models fails)
# ---------------------------------------------------------------------------


def _fallback_comparison_plot(all_metrics: dict, figs_dir: str) -> None:
    """Simple grouped bar chart of key metrics as a fallback."""
    os.makedirs(figs_dir, exist_ok=True)

    models = list(all_metrics.keys())
    metrics = ["mae", "crps_mean", "winkler_90"]
    labels = ["MAE", "CRPS (mean)", "Winkler 90%"]

    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    x = np.arange(n_models)

    palette = sns.color_palette("muted", n_colors=n_metrics)
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_models), 5))

    for gi, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [float(all_metrics[m].get(metric, float("nan"))) for m in models]
        offset = (gi - n_metrics / 2.0 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.92,
            color=palette[gi],
            label=label,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.3,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("EUR/MWh", fontsize=11)
    ax.set_title("Baseline Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(figs_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fallback] Comparison chart saved → {out_path}")


if __name__ == "__main__":
    main()
