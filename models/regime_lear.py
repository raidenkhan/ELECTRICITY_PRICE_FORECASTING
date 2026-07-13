"""
regime_lear.py â€” Regime-Conditional LEAR Models
==================================================
Trains one LEAR model per regime (3 total).
At prediction time, each model produces point + quantile forecasts.
The MRS-LEAR ensemble then soft-blends using HMM posterior probabilities.

Based on: Uniejewski et al. [12] + Lago et al. LEAR framework.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Build LEAR feature matrix: price lags + fundamentals + calendar."""
    lag_cols = [c for c in df.columns if c.startswith('P_lag_')]
    fundamental_cols = ['W_CF', 'S_CF', 'G_price_zscore', 'T_demand', 'renewable_penetration']
    cal_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'month_sin', 'month_cos', 'is_holiday', 'is_weekend']
    all_cols = lag_cols + fundamental_cols + cal_cols
    # Only keep cols present in df
    available = [c for c in all_cols if c in df.columns]
    X = df[available].values
    return np.nan_to_num(X, nan=0.0)


class RegimeLEAR:
    """LEAR model specialized for a single market regime."""

    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        self.models = {}   # hour â†’ ElasticNet
        self.scalers = {}  # hour â†’ StandardScaler
        self.residuals = {}  # hour â†’ np.ndarray

    def fit(self, df: pd.DataFrame) -> None:
        """Fit 24 per-hour ElasticNet models on regime-filtered data."""
        df = df.dropna(subset=['price'])
        n_samples = len(df)
        print(f"  [regime_lear] Regime {self.regime_id}: fitting on {n_samples} samples")

        if n_samples < 50:
            print(f"  [regime_lear] WARNING: Regime {self.regime_id} has <50 samples, using full data fallback")
            df_fallback = df  # keep all hours

        for h in range(24):
            df_h = df[df.index.hour == h]
            if len(df_h) < 10:
                # Fallback: use all hours if this regime is underrepresented
                df_h = df

            X = _prepare_features(df_h)
            y = df_h['price'].values

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # ElasticNetCV for proper regularisation strength selection
            # For speed, use a small grid; can expand for final runs
            try:
                model = ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    cv=3,
                    max_iter=5000,
                    n_jobs=1
                )
                model.fit(X_s, y)
            except Exception:
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
                model.fit(X_s, y)

            y_pred = model.predict(X_s)
            residuals = y - y_pred

            self.models[h] = model
            self.scalers[h] = scaler
            self.residuals[h] = residuals

    def predict(self, df: pd.DataFrame, n_bootstrap: int = 500) -> dict:
        """Return point predictions + quantile bands via residual bootstrap."""
        y_pred = np.zeros(len(df))
        quantiles = np.zeros((len(df), len(QUANTILE_LEVELS)))

        for h in range(24):
            idx = np.where(df.index.hour == h)[0]
            if len(idx) == 0:
                continue

            df_h = df.iloc[idx]
            X = _prepare_features(df_h)
            X_s = self.scalers[h].transform(X)
            preds = self.models[h].predict(X_s)
            y_pred[idx] = preds

            res = np.nan_to_num(self.residuals[h])
            sampled = np.random.choice(res, size=(len(preds), n_bootstrap), replace=True)
            boot = preds[:, None] + sampled
            q_vals = np.quantile(boot, QUANTILE_LEVELS, axis=1).T
            quantiles[idx] = q_vals

        return {'point': y_pred, 'quantiles': quantiles}


def train_regime_lears(feature_matrix_path: str | Path,
                       hmm_labels_path: str | Path,
                       out_dir: str | Path,
                       n_regimes: int = 3) -> None:
    """
    Main entry: train one LEAR per regime, save to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[regime_lear] Loading data...")
    df = pd.read_parquet(feature_matrix_path)
    labels = pd.read_parquet(hmm_labels_path)

    # Merge regime labels onto feature matrix
    df = df.join(labels[['hmm_regime']], how='left')
    df['hmm_regime'] = df['hmm_regime'].fillna(1).astype(int)  # default to Base

    train_df = df[df['split'] == 'train']
    print(f"[regime_lear] Training set: {len(train_df)} rows")
    print(f"[regime_lear] Regime distribution in train:")
    print(train_df['hmm_regime'].value_counts().sort_index())

    for k in range(n_regimes):
        regime_df = train_df[train_df['hmm_regime'] == k]
        print(f"\n[regime_lear] --- Regime {k} ({len(regime_df)} samples) ---")

        lear = RegimeLEAR(regime_id=k)
        lear.fit(regime_df)

        out_path = out_dir / f'lear_regime_{k}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(lear, f)
        print(f"  [regime_lear] Saved â†’ {out_path}")

    print("\n[regime_lear] Training complete.")


if __name__ == '__main__':
    train_regime_lears(
        feature_matrix_path=ROOT / 'data/processed/feature_matrix.parquet',
        hmm_labels_path=ROOT / 'data/regimes/hmm_regime_labels.parquet',
        out_dir=ROOT / 'models',
    )
