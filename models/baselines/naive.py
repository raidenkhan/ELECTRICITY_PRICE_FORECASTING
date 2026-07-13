"""
naive.py - Naive Persistence Baseline

This is the floor model.
Forecast for hour h on day D = price of hour h on day D-7.
(168-hour persistence).
Produces probabilistic output by assuming empirical error distribution,
but primarily used for point metrics or zero-residual point forecasts.
"""

import numpy as np
import pandas as pd
import torch

class NaivePersistence:
    def __init__(self):
        self.quantile_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        self.empirical_residuals = None
        
    def fit(self, train_df: pd.DataFrame):
        """Fit phase just extracts historical residuals for probabilisitic uncertainty."""
        # Train dataset target is price, lag 168 is P_lag_168
        # Residuals: y - y_pred
        y_true = train_df['price'].values
        y_pred = train_df['P_lag_168'].values
        # Drop NaNs
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        self.empirical_residuals = y_true[valid] - y_pred[valid]

    def predict(self, test_df: pd.DataFrame) -> dict:
        """
        Returns point predictions and quantiles.
        Predictions for a horizon of 24h are built by chunking test_df.
        We'll treat test_df as hourly and return hourly dict.
        """
        y_pred = test_df['P_lag_168'].values
        
        # Empirical quantiles from residuals
        res_quantiles = np.quantile(self.empirical_residuals, self.quantile_levels)
        
        # Add residuals to point predictions to form quantile forecasts
        # Shape: [N, 7]
        quantiles = y_pred[:, None] + res_quantiles[None, :]
        
        return {
            'point': y_pred,
            'quantiles': quantiles
        }
