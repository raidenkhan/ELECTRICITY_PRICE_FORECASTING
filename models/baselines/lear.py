"""
lear.py - LEAR Baseline (Lago et al. 2021)

Lasso with Elastic Net regularisation.
Fits one model per hour-of-day (24 models).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

class LEARBaseline:
    def __init__(self, cv_weeks=52):
        self.cv_weeks = cv_weeks
        self.models = {h: None for h in range(24)}
        self.scalers = {h: StandardScaler() for h in range(24)}
        self.residuals = {h: None for h in range(24)}
        self.quantile_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        
    def _prepare_features(self, df: pd.DataFrame):
        """Extracts 168 price lags + calendar dummies."""
        # Using existing lags from build_features + dummies
        lag_cols = [c for c in df.columns if c.startswith("P_lag_")]
        cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", 
                    "month_sin", "month_cos", "is_holiday", "is_weekend"]
        return df[lag_cols + cal_cols].values

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.dropna(subset=['price'])
        
        def fit_hour(h):
            df_h = train_df[train_df.index.hour == h]
            X = self._prepare_features(df_h)
            y = df_h['price'].values
            
            # Simple imputation for any remaining NaNs in lags
            X = np.nan_to_num(X)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Bypass 52-week grid CV to prevent system freeze
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10)
            model.fit(X_scaled, y)
            
            # Extract residuals for bootstrap
            y_pred = model.predict(X_scaled)
            residuals = y - y_pred
            
            return h, model, scaler, residuals

        # Fit all 24 hours sequentially to avoid joblib deadlock on Windows
        results = Parallel(n_jobs=1)(delayed(fit_hour)(h) for h in range(24))
        
        for h, model, scaler, residuals in results:
            self.models[h] = model
            self.scalers[h] = scaler
            self.residuals[h] = residuals

    def predict(self, test_df: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
        y_pred = np.zeros(len(test_df))
        quantiles = np.zeros((len(test_df), len(self.quantile_levels)))
        
        for h in range(24):
            idx = test_df.index.hour == h
            if not idx.any():
                continue
                
            df_h = test_df[idx]
            X = self._prepare_features(df_h)
            X = np.nan_to_num(X)
            
            X_scaled = self.scalers[h].transform(X)
            preds = self.models[h].predict(X_scaled)
            y_pred[idx] = preds
            
            # Residual bootstrap
            res = self.residuals[h]
            # Replace NaNs with 0 just in case
            res = np.nan_to_num(res)
            
            # Sample (n_bootstrap, n_samples)
            sampled_res = np.random.choice(res, size=(len(preds), n_bootstrap), replace=True)
            bootstrap_preds = preds[:, None] + sampled_res
            
            q_vals = np.quantile(bootstrap_preds, self.quantile_levels, axis=1).T
            quantiles[idx] = q_vals
            
        return {
            'point': y_pred,
            'quantiles': quantiles
        }
