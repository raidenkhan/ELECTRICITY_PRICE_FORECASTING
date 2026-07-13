import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.stats import rankdata

from src.data.preprocess import EPFPreprocessor
from src.models.ds_hdp_hmm import VariationalStickyHMM
from src.experiments.train_pipeline import EPFPipeline
from src.models.intensity import HawkesIntensity, detect_spikes

# ------------------- Helpers -------------------------------------------
def build_golden_features(df):
    feats = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
        'Residual_Load', 'total_re_penetration', 'dark_doldrums',
        '12h_price_range', '3h_neg_streak',
        'price_lag_24', 'price_lag_48', 'price_lag_168',
    ]
    df = df.copy()
    df['price_rolling48_lag1'] = df['price_lag_24'].rolling(48, min_periods=1).mean()
    is_spike = detect_spikes(df['price_lag_24'].fillna(df['price'].mean()).values)
    hi = HawkesIntensity(baseline=0.01, alpha=0.5, beta=0.8)
    df['spike_intensity'] = hi.fit_and_predict(is_spike)
    df['moc_distance'] = (df['Residual_Load'] - 38000) / 10000.0
    return df, [c for c in feats if c in df.columns] + ['price_rolling48_lag1', 'spike_intensity', 'moc_distance']

def apply_kalman_bias(df, pred_col, actual_col, Q=1e-3, R=1e-1):
    errors = (df[pred_col] - df[actual_col]).values
    x = 0.0; P = 1.0; bias_est = np.zeros(len(errors))
    for t in range(len(errors)):
        P += Q
        if t > 0:
            z = errors[t-1]; K = P / (P + R); x += K * (z - x); P = (1 - K) * P
        bias_est[t] = x
    return df[pred_col] - bias_est

def gumbel_copula_adjustment(u, v, theta=2.0):
    """
    Gumbel Copula adjustment for upper-tail dependence.
    u: Marginal (Regime Probability)
    v: Marginal (Price Rank/Probability)
    theta: Dependence parameter (> 1)
    """
    # Small epsilon for numerical stability
    eps = 1e-6
    u = np.clip(u, eps, 1-eps)
    v = np.clip(v, eps, 1-eps)
    # Gumbel Copula density C(u,v) proportional adjustment
    # c(u,v) = C(u,v) * (ln u * ln v)^(theta-1) / (u*v) * ((ln u)^theta + (ln v)^theta)^(2/theta - 2) * (1 + (theta-1)((ln u)^theta + (ln v)^theta)^(-1/theta))
    # Faster proxy: increase weight if both are in upper tail
    # Using a simple tail-power coupling
    adjusted = u * (v ** (theta - 1.0))
    return adjusted

# ------------------- Benchmark -----------------------------------------
def run_benchmark():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))
    out_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)
    
    pipeline = EPFPipeline(data_path=os.path.join(data_dir, 'Germany_master_entsoe_2015_2026.csv'),
                           comm_path=os.path.join(data_dir, 'commodities.csv'),
                           flow_path=os.path.join(data_dir, 'cross_border_flows.csv'))

    df_raw = pipeline.load_and_merge()
    df_e, features = build_golden_features(pipeline.preprocessor.process(df_raw).dropna())
    df_train = df_e.loc['2022-01-01':'2023-12-31']
    df_test  = df_e.loc['2024-01-01':'2024-06-30']
    
    # HMM
    daily_tr = pipeline.preprocessor.extract_daily_hmm_features(df_train); daily_te = pipeline.preprocessor.extract_daily_hmm_features(df_test)
    scaler = StandardScaler(); obs_tr = torch.tensor(scaler.fit_transform(daily_tr.values), dtype=torch.float32)
    hmm = VariationalStickyHMM(obs_dim=obs_tr.shape[1], K_max=6)
    opt = torch.optim.Adam(hmm.parameters(), lr=0.01)
    for _ in range(120): opt.zero_grad(); hmm.compute_loss(obs_tr).backward(); opt.step()
    
    df_train = df_train.copy(); df_train['date_only'] = df_train.index.date
    daily_tr['regime'] = hmm.viterbi(obs_tr).numpy()
    df_train['regime'] = df_train['date_only'].map(daily_tr.set_index(daily_tr.index.date)['regime'].to_dict())
    
    df_test = df_test.copy(); df_test['date_only'] = df_test.index.date
    obs_te = torch.tensor(scaler.transform(daily_te.values), dtype=torch.float32)
    log_a = F.log_softmax(hmm.log_trans, dim=1); log_pi = F.log_softmax(hmm.log_init, dim=0); log_emit = hmm.emission_log_prob(obs_te)
    alpha = torch.zeros(obs_te.shape[0], 6); alpha[0] = log_pi + log_emit[0]
    for t in range(1, obs_te.shape[0]): alpha[t] = torch.logsumexp(alpha[t-1].unsqueeze(1) + log_a, 0) + log_emit[t]
    df_test['soft_probs'] = df_test['date_only'].map(daily_te.assign(p=list(torch.softmax(alpha, 1).detach().numpy())).set_index(daily_te.index.date)['p'].to_dict())

    # Build Experts
    r_stats = df_train.groupby('regime')['price'].agg(['mean', (lambda x: (x < 0).mean())])
    etype = {k: ('solar' if r['<lambda_0>'] > 0.05 else ('evt' if r['mean'] > 130 else 'tft')) for k, r in r_stats.iterrows()}
    experts = {}
    for k, et in etype.items():
        sub = df_train[df_train['regime'] == k]
        if len(sub) < 24: continue
        params = dict(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42)
        if et == 'evt': params.update(objective='quantile', alpha=0.9)
        elif et == 'solar': params.update(objective='quantile', alpha=0.1)
        m = lgb.LGBMRegressor(**params); m.fit(sub[features].apply(pd.to_numeric, errors='coerce').fillna(0), sub['price_asinh'])
        experts[k] = m

    # Inference
    def compute_mixture(method='hmm'):
        X_test = df_test[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        preds = {k: m.predict(X_test) for k, m in experts.items()}
        
        # Marginal for Copula: Normal expert's rank-normalized price
        # We need a 'base' expert for the Copula marginal.
        base_k = [k for k,v in etype.items() if v == 'tft'][0]
        v_marginal = rankdata(preds[base_k]) / len(preds[base_k])
        
        pred_asinh = np.zeros(len(df_test))
        for i in range(len(df_test)):
            p_k_hmm = df_test['soft_probs'].iloc[i]
            if method == 'copula':
                weights = np.zeros(6)
                for k in experts.keys():
                    if etype[k] == 'evt':
                        # Apply Gumbel Copula adjustment for upper tail
                        weights[k] = gumbel_copula_adjustment(p_k_hmm[k], v_marginal[i], theta=2.5)
                    else:
                        weights[k] = p_k_hmm[k]
                if weights.sum() > 0: weights /= weights.sum()
                else: weights = p_k_hmm
            else:
                weights = p_k_hmm
                
            pred_asinh[i] = sum(weights[k] * preds[k][i] for k in experts.keys())
            
        return apply_kalman_bias(df_test.assign(p=pipeline.preprocessor.inverse_transform_price(pred_asinh)), 'p', 'price')

    print("Running Baseline Fisher/HMM Mixture...")
    pred_base = compute_mixture(method='hmm')
    print("Running Phase 12: Copula-Regime Switcher...")
    pred_copula = compute_mixture(method='copula')
    
    y = df_test['price']; s = y > 200
    res_df = pd.DataFrame([
        {'Model': 'Fisher Gating (P11)', 'MAE': mean_absolute_error(y, pred_base), 'Spike-MAE': mean_absolute_error(y[s], pred_base[s])},
        {'Model': 'Copula Switcher (P12)', 'MAE': mean_absolute_error(y, pred_copula), 'Spike-MAE': mean_absolute_error(y[s], pred_copula[s])}
    ])
    
    print("\n--- Ablation Results: Copula-Regime Switching ---")
    print(res_df.to_string(index=False))

    plt.figure(figsize=(10, 6)); sns.barplot(x='Model', y='Spike-MAE', data=res_df, palette='rocket')
    plt.title('Impact of Copula-Regime Switching on Spike Accuracy'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_copula.png')); plt.close()
    print(f"Results saved to {out_dir}")

if __name__ == '__main__':
    run_benchmark()
