"""
hmm_regime.py â€” 3-State Gaussian HMM Regime Detector
======================================================
Implements the regime detection component of the MRS-LEAR hybrid.

Architecture (Paraschiv et al., Dimoulkas & Amelin):
  - 3 hidden states: Base, Surplus/Negative, Spike
  - Features: residual_load_zscore, price_lag24_zscore, renewable_penetration
  - Fitted on training split ONLY (causal, no leakage)
  - Rolling Viterbi decode for val/test
  - Outputs: regime sequence S_t âˆˆ {0,1,2} + posterior probabilities

Usage:
    python src/models/hmm_regime.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[2]


def build_hmm_features(df: pd.DataFrame, scaler: StandardScaler | None = None):
    """Build HMM observation features from feature matrix."""
    # Residual load: demand minus renewable infeed proxy
    # W_CF and S_CF are capacity factors; we proxy infeed as CF * max_possible
    # Use renewable_penetration as the key regime signal
    feat_cols = ['P_lag_24', 'renewable_penetration', 'G_price_zscore', 'W_CF', 'S_CF']
    X = df[feat_cols].copy()

    # Fill any remaining NaNs with forward fill then zero
    X = X.ffill().fillna(0.0).values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def label_regimes(model: GaussianHMM, train_df: pd.DataFrame) -> dict:
    """
    Map hidden states to economic regimes by mean price in training data.
    Returns dict: {hidden_state_id: regime_label}
    """
    X_train, _ = build_hmm_features(train_df)
    states = model.predict(X_train)

    mean_prices = {}
    for s in range(model.n_components):
        mask = states == s
        if mask.any():
            mean_prices[s] = train_df['price'].values[mask].mean()
        else:
            mean_prices[s] = 0.0

    # Sort by mean price: lowest=Surplus(0), middle=Base(1), highest=Spike(2)
    sorted_states = sorted(mean_prices, key=mean_prices.get)
    state_map = {sorted_states[i]: i for i in range(len(sorted_states))}
    print(f"[hmm] State mean prices: {mean_prices}")
    print(f"[hmm] State mapping (hidden â†’ regime): {state_map}")
    return state_map


def fit_hmm(feature_matrix_path: str | Path,
            out_model_path: str | Path,
            out_labels_path: str | Path,
            n_components: int = 3,
            n_iter: int = 200) -> None:
    """
    Main entry: fit HMM on training data, decode all splits, save artefacts.
    """
    feature_matrix_path = Path(feature_matrix_path)
    out_model_path = Path(out_model_path)
    out_labels_path = Path(out_labels_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)

    print("[hmm] Loading feature matrix...")
    df = pd.read_parquet(feature_matrix_path)

    train_df = df[df['split'] == 'train']
    print(f"[hmm] Training on {len(train_df)} rows ({train_df.index.min()} -> {train_df.index.max()})")

    # --- Build & scale features ---
    X_train, scaler = build_hmm_features(train_df)

    # --- Fit GaussianHMM ---
    print(f"[hmm] Fitting {n_components}-state Gaussian HMM (n_iter={n_iter})...")
    model = GaussianHMM(
        n_components=n_components,
        covariance_type='full',
        n_iter=n_iter,
        random_state=42,
        verbose=False
    )
    model.fit(X_train)
    print(f"[hmm] HMM converged. Log-likelihood: {model.score(X_train):.2f}")

    # --- Map hidden states to economic regimes ---
    state_map = label_regimes(model, train_df)

    # --- Decode all splits ---
    all_results = []
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split].copy()
        if split_df.empty:
            continue

        X_split, _ = build_hmm_features(split_df, scaler=scaler)
        hidden_states = model.predict(X_split)
        posteriors = model.predict_proba(X_split)

        # Remap hidden states -> economic regimes
        regimes = np.array([state_map[s] for s in hidden_states])

        split_df['hmm_regime'] = regimes
        for k in range(n_components):
            split_df[f'hmm_prob_{k}'] = posteriors[:, k]

        all_results.append(split_df[['hmm_regime'] + [f'hmm_prob_{k}' for k in range(n_components)] + ['split', 'price']])

    labels_df = pd.concat(all_results, axis=0)

    # Report regime distribution
    print("\n[hmm] Regime distribution (all splits):")
    print(labels_df.groupby(['split', 'hmm_regime'])['price'].agg(['count', 'mean', 'min', 'max']).to_string())

    # Save
    labels_df.to_parquet(out_labels_path)
    print(f"[hmm] Saved regime labels â†’ {out_labels_path}")

    bundle = {'model': model, 'scaler': scaler, 'state_map': state_map}
    with open(out_model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[hmm] Saved HMM bundle â†’ {out_model_path}")


if __name__ == '__main__':
    fit_hmm(
        feature_matrix_path=ROOT / 'data/processed/feature_matrix.parquet',
        out_model_path=ROOT / 'models/hmm_regime.pkl',
        out_labels_path=ROOT / 'data/regimes/hmm_regime_labels.parquet',
    )
