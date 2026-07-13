"""
src/experiments/expert_complexity_ablation.py
=============================================
Ablation study comparing different specialist complexities:
1. Baseline: No specialists (Anchor only)
2. Current: ElasticNet Specialists
3. Advanced: Gradient Boosting Specialists (LGBM)
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from models.hr_lear_residuals import ResidualExpert
from models.mrs_lear_ensemble import crps_from_quantiles

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

class GbmExpert:
    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        self.model = HistGradientBoostingRegressor()
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray):
        cols_to_drop = ['anchor_pred', 'residual', 'price']
        X_clean = X.select_dtypes(include=[np.number]).drop(columns=cols_to_drop, errors='ignore')
        self.feature_names_ = X_clean.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y, sample_weight=weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols_to_drop = ['anchor_pred', 'residual', 'price']
        X_clean = X.select_dtypes(include=[np.number]).drop(columns=cols_to_drop, errors='ignore')
        X_clean = X_clean[self.feature_names_]
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

def run_ablation():
    print("[ablation] Loading data...")
    res_df = pd.read_parquet(ROOT / 'data/processed/hr_lear_residuals.parquet')
    feature_df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    labels = pd.read_parquet(ROOT / 'data/regimes/hmm_regime_labels.parquet')
    
    data = res_df.join(feature_df.drop(columns=['price', 'split']), how='left')
    data = data.join(labels[[f'hmm_prob_{k}' for k in range(3)] + ['hmm_regime']], how='left')
    
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    
    # Load Global Anchor (Step 1)
    with open(ROOT / 'models/hr_lear_anchor.pkl', 'rb') as f:
        anchor = pickle.load(f)
    
    anchor_out = anchor.predict(test_data, n_bootstrap=100)
    y_anchor = anchor_out['point']
    q_anchor = anchor_out['quantiles']
    y_true = test_data['price'].values
    prob_matrix = test_data[[f'hmm_prob_{k}' for k in range(3)]].values
    max_probs = prob_matrix.max(axis=1)
    guard_mask = max_probs < 0.6

    results = []

    # Scenario 1: Anchor Only
    mae_a = np.mean(np.abs(y_true - y_anchor))
    crps_a = crps_from_quantiles(y_true, q_anchor, QUANTILE_LEVELS)
    results.append({'Scenario': 'Global Anchor Only', 'MAE': mae_a, 'CRPS': crps_a})

    # Scenario 2: ElasticNet Specialists (Current)
    print("[ablation] Training ElasticNet Specialists...")
    en_corrections = np.zeros((len(test_data), 3))
    for k in range(3):
        expert = ResidualExpert(k)
        weights = train_data[f'hmm_prob_{k}'].values
        expert.fit(train_data, train_data['residual'], weights)
        en_corrections[:, k] = expert.predict(test_data)
    
    weighted_corr_en = np.sum(prob_matrix * en_corrections, axis=1)
    weighted_corr_en[guard_mask] = 0.0
    y_en = y_anchor + weighted_corr_en
    q_en = q_anchor + weighted_corr_en[:, np.newaxis]
    
    results.append({
        'Scenario': 'ElasticNet Specialists', 
        'MAE': np.mean(np.abs(y_true - y_en)), 
        'CRPS': crps_from_quantiles(y_true, q_en, QUANTILE_LEVELS)
    })

    # Scenario 3: GBM Specialists (Advanced)
    print("[ablation] Training GBM Specialists...")
    gbm_corrections = np.zeros((len(test_data), 3))
    for k in range(3):
        expert = GbmExpert(k)
        weights = train_data[f'hmm_prob_{k}'].values
        expert.fit(train_data, train_data['residual'], weights)
        gbm_corrections[:, k] = expert.predict(test_data)
    
    weighted_corr_gbm = np.sum(prob_matrix * gbm_corrections, axis=1)
    weighted_corr_gbm[guard_mask] = 0.0
    y_gbm = y_anchor + weighted_corr_gbm
    q_gbm = q_anchor + weighted_corr_gbm[:, np.newaxis]
    
    results.append({
        'Scenario': 'GBM Specialists', 
        'MAE': np.mean(np.abs(y_true - y_gbm)), 
        'CRPS': crps_from_quantiles(y_true, q_gbm, QUANTILE_LEVELS)
    })

    # Scenario 4: No Uncertainty Guard (Ablating the SPI)
    print("[ablation] Ablating SPI (No Uncertainty Guard)...")
    y_no_guard = y_anchor + np.sum(prob_matrix * en_corrections, axis=1)
    results.append({
        'Scenario': 'ElasticNet (No SPI)', 
        'MAE': np.mean(np.abs(y_true - y_no_guard)), 
        'CRPS': np.nan # skip crps for brevity
    })

    # Display Results
    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("           EXPERIMENT: SPECIALIST COMPLEXITY ABLATION")
    print("="*50)
    print(res_df.to_string(index=False))
    print("="*50)
    
    res_df.to_csv(ROOT / 'results/tables/complexity_ablation.csv', index=False)

if __name__ == '__main__':
    run_ablation()
