"""
hr_lear_residuals.py — HR-LEAR Step 2: Residual Experts
=========================================================
Trains shallow regime-specific specialists to predict the 
residuals of the global anchor model.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

class ResidualExpert:
    """Shallow expert targeting residuals from a global baseline."""
    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        # Using ElasticNet as a robust'shallow' learner with implicit L2
        self.model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=2000)
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray):
        """Fit on weighted residuals."""
        # Drop non-feature numeric columns (leakage)
        cols_to_drop = ['anchor_pred', 'residual', 'price']
        X_clean = X.select_dtypes(include=[np.number]).drop(columns=cols_to_drop, errors='ignore')
        self.feature_names_ = X_clean.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y, sample_weight=weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols_to_drop = ['anchor_pred', 'residual', 'price']
        X_clean = X.select_dtypes(include=[np.number]).drop(columns=cols_to_drop, errors='ignore')
        # Ensure only features seen during fit are present
        X_clean = X_clean[self.feature_names_]
        
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

def train_residual_experts():
    print("[hr_lear_resid] Loading residuals and HMM labels...")
    res_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    feature_df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    
    # Join features for training
    data = res_df.join(feature_df.drop(columns=['price', 'split']), how='left')
    data = data.join(labels[[f'hmm_prob_{k}' for k in range(3)]], how='left')
    
    train_data = data[data['split'] == 'train']
    print(f"[hr_lear_resid] Training on {len(train_data)} rows...")

    experts = {}
    for k in range(3):
        print(f"[hr_lear_resid] Fitting Residual Expert for Regime {k}...")
        expert = ResidualExpert(k)
        
        # Base HMM probability weights
        weights = train_data[f'hmm_prob_{k}'].values
        # Ensure weights aren't all zero
        if weights.sum() < 1e-3:
            print(f"  Warning: Regime {k} has no mass in training set. Skipping.")
            continue
            
        expert.fit(train_data, train_data['residual'], weights)
        experts[k] = expert
        
    # Save experts
    models_dir = ROOT / 'models'
    with open(models_dir / 'hr_lear_residual_experts.pkl', 'wb') as f:
        pickle.dump(experts, f)
    print(f"[hr_lear_resid] Saved {len(experts)} experts to {models_dir / 'hr_lear_residual_experts.pkl'}")

if __name__ == '__main__':
    train_residual_experts()
