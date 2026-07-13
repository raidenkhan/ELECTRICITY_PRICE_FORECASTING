"""
mrs_lear_ensemble.py â€” MRS-LEAR Soft-Blend Ensemble
======================================================
Final ensemble: F_t = Î£_k P(S_t = k | F_t) Ã— LEAR_k(X_t)

Architecture (Paraschiv et al. + Kosater & Mosler):
  - Load 3 regime-specific LEAR models
  - Weight predictions by HMM posterior probabilities
  - Compute CRPS, MAE, RMSE, regime-stratified metrics
  - Save predictions + full comparison table
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add src/models to path for local imports
sys.path.append(str(Path(__file__).parent))
try:
    from regime_lear import RegimeLEAR
    # Hack for pickle: map __main__.RegimeLEAR to the imported class
    # because it was pickled while running regime_lear.py as __main__
    import __main__
    __main__.RegimeLEAR = RegimeLEAR
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]


def crps_from_quantiles(y_true: np.ndarray, quantiles: np.ndarray,
                         q_levels: list) -> float:
    """Approximate CRPS via quantile scoring (standard decomposition)."""
    scores = []
    for i, q in enumerate(q_levels):
        err = y_true - quantiles[:, i]
        pinball = np.where(err >= 0, q * err, (q - 1) * err)
        scores.append(pinball.mean())
    return float(2 * np.mean(scores))


def evaluate_mrs_lear(feature_matrix_path: str | Path,
                      hmm_labels_path: str | Path,
                      models_dir: str | Path,
                      out_predictions_path: str | Path,
                      out_results_path: str | Path,
                      n_regimes: int = 3,
                      split: str = 'test') -> pd.DataFrame:
    """
    Run soft-blended MRS-LEAR inference on the specified split.
    Returns metrics DataFrame.
    """
    feature_matrix_path = Path(feature_matrix_path)
    hmm_labels_path = Path(hmm_labels_path)
    models_dir = Path(models_dir)
    out_predictions_path = Path(out_predictions_path)
    out_results_path = Path(out_results_path)
    out_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out_results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[mrs_lear] Loading {split} data...")
    df = pd.read_parquet(feature_matrix_path)
    labels = pd.read_parquet(hmm_labels_path)
    df = df.join(labels[[f'hmm_prob_{k}' for k in range(n_regimes)] + ['hmm_regime']], how='left')

    test_df = df[df['split'] == split].copy()
    print(f"[mrs_lear] {split} set: {len(test_df)} rows")

    # Load regime LEAR models
    lears = {}
    for k in range(n_regimes):
        path = models_dir / f'lear_regime_{k}.pkl'
        with open(path, 'rb') as f:
            lears[k] = pickle.load(f)
        print(f"[mrs_lear] Loaded LEAR regime {k} from {path}")

    # Per-regime predictions
    regime_preds = {}
    regime_quantiles = {}
    for k in range(n_regimes):
        out = lears[k].predict(test_df, n_bootstrap=500)
        regime_preds[k] = out['point']
        regime_quantiles[k] = out['quantiles']

    # Soft blend using HMM posteriors
    prob_matrix = np.zeros((len(test_df), n_regimes))
    for k in range(n_regimes):
        col = f'hmm_prob_{k}'
        if col in test_df.columns:
            prob_matrix[:, k] = test_df[col].values
        else:
            prob_matrix[:, k] = 1.0 / n_regimes

    # Normalize
    prob_matrix = prob_matrix / (prob_matrix.sum(axis=1, keepdims=True) + 1e-9)

    # Blended point forecast
    blended_point = np.zeros(len(test_df))
    for k in range(n_regimes):
        blended_point += prob_matrix[:, k] * regime_preds[k]

    # Blended quantiles
    blended_quantiles = np.zeros((len(test_df), len(QUANTILE_LEVELS)))
    for k in range(n_regimes):
        for j in range(len(QUANTILE_LEVELS)):
            blended_quantiles[:, j] += prob_matrix[:, k] * regime_quantiles[k][:, j]

    # Ground truth
    y_true = test_df['price'].values

    # Metrics
    mae = float(np.mean(np.abs(blended_point - y_true)))
    rmse = float(np.sqrt(np.mean((blended_point - y_true) ** 2)))
    crps = crps_from_quantiles(y_true, blended_quantiles, QUANTILE_LEVELS)

    print(f"\n[mrs_lear] === MRS-LEAR {split.upper()} RESULTS ===")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  CRPS : {crps:.4f}")

    # Regime-stratified CRPS
    regimes = test_df['hmm_regime'].fillna(1).astype(int).values
    print("\n[mrs_lear] Regime-stratified CRPS:")
    regime_crps = {}
    for k in range(n_regimes):
        mask = regimes == k
        if mask.sum() > 0:
            rc = crps_from_quantiles(y_true[mask], blended_quantiles[mask], QUANTILE_LEVELS)
            regime_crps[k] = rc
            n = mask.sum()
            mean_p = y_true[mask].mean()
            print(f"  Regime {k}: CRPS={rc:.4f}  n={n}  mean_price={mean_p:.2f}")

    # Compare vs baselines
    lear_crps = 8.615
    sarima_crps = 18.324
    print(f"\n[mrs_lear] Comparison:")
    print(f"  LEAR baseline   : CRPS={lear_crps:.4f}")
    print(f"  SARIMA-GARCH    : CRPS={sarima_crps:.4f}")
    print(f"  MRS-LEAR (ours) : CRPS={crps:.4f}  {'âœ“ BEATS LEAR' if crps < lear_crps else 'âœ— Below LEAR'}")

    # Save predictions
    pred_df = pd.DataFrame({
        'datetime': test_df.index,
        'y_true': y_true,
        'y_pred': blended_point,
        'regime': regimes,
    })
    for j, q in enumerate(QUANTILE_LEVELS):
        pred_df[f'q{int(q*100):02d}'] = blended_quantiles[:, j]
    pred_df.to_parquet(out_predictions_path)
    print(f"\n[mrs_lear] Saved predictions â†’ {out_predictions_path}")

    # Save metrics
    results = {
        'model': 'MRS-LEAR',
        'MAE': mae, 'RMSE': rmse, 'CRPS': crps,
        'CRPS_regime_0': regime_crps.get(0, np.nan),
        'CRPS_regime_1': regime_crps.get(1, np.nan),
        'CRPS_regime_2': regime_crps.get(2, np.nan),
        'CRPS_improvement_vs_LEAR': ((lear_crps - crps) / lear_crps) * 100,
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(out_results_path, index=False)
    print(f"[mrs_lear] Saved metrics â†’ {out_results_path}")

    return results_df


if __name__ == '__main__':
    evaluate_mrs_lear(
        feature_matrix_path=ROOT / 'data/processed/feature_matrix.parquet',
        hmm_labels_path=ROOT / 'data/regimes/hmm_regime_labels.parquet',
        models_dir=ROOT / 'models',
        out_predictions_path=ROOT / 'results/predictions/mrs_lear_predictions.parquet',
        out_results_path=ROOT / 'results/tables/mrs_lear_results.csv',
        split='test',
    )
