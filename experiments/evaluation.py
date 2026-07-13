# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# evaluation.py — Full evaluation suite for LF-GP-NRF and baseline models.
#
# Implements:
#   - CRPS (vectorised, sample-based)
#   - Energy Score (24-dim joint)
#   - Pinball / Quantile Loss
#   - Winkler Interval Score
#   - Diebold-Mariano test (Newey-West HAC variance)
#   - EPFEvaluator class (run, compute_all_metrics, plots, tables)
#   - compare_models utility

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import properscoring  # noqa: F401 — available for direct use by module consumers
import torch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Seaborn / matplotlib global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
_PALETTE = sns.color_palette("muted")


# ===========================================================================
# 1.  CRPS — Continuous Ranked Probability Score
# ===========================================================================


def compute_crps_samples(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute per-observation CRPS from an ensemble of samples.

    Uses the energy-form identity::

        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]

    The second expectation is approximated without O(S²) cost by splitting
    the S samples into two random halves of equal size and computing the
    mean absolute difference between them.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Observed values.
    samples : np.ndarray, shape (N, S)
        Ensemble of predictive samples; S must be >= 2.

    Returns
    -------
    crps : np.ndarray, shape (N,)
        Per-observation CRPS values (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2 or y_true.ndim != 1:
        raise ValueError(
            f"Expected y_true (N,) and samples (N,S); "
            f"got y_true {y_true.shape} and samples {samples.shape}."
        )

    N, S = samples.shape

    # E[|X - y|]
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)  # (N,)

    # E[|X - X'|] via random permutation — O(N*S) instead of O(N*S^2)
    perm = samples[:, np.random.permutation(S)]
    term2 = np.mean(np.abs(samples - perm), axis=1) * 0.5  # (N,)

    return term1 - term2  # (N,)


# ===========================================================================
# 2.  Energy Score — joint 24-hour probabilistic calibration
# ===========================================================================


def compute_energy_score(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Multivariate Energy Score over 24-hour day-ahead price vectors.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H)
        Observed 24-hour price vectors (H = 24).
    samples : np.ndarray, shape (N, S, H)
        Ensemble of predictive samples; S must be >= 2.

    Returns
    -------
    es : float
        Mean Energy Score across all N forecast days (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)

    if y_true.ndim != 2 or samples.ndim != 3:
        raise ValueError(
            f"Expected y_true (N,H) and samples (N,S,H); "
            f"got y_true {y_true.shape} and samples {samples.shape}."
        )

    N, S, H = samples.shape

    # term1: E_s[ ||X^(s) - y|| ] averaged over days
    # samples: (N, S, H), y_true: (N, H) → diff: (N, S, H)
    diff1 = samples - y_true[:, None, :]  # (N, S, H)
    norm1 = np.linalg.norm(diff1, axis=2)  # (N, S)
    term1 = np.mean(norm1)  # scalar

    # term2: 0.5 * E_{s,s'}[ ||X^(s) - X^(s')|| ] — approx with two halves
    half = S // 2
    perm_idx = np.random.permutation(S)
    half_a = samples[:, perm_idx[:half], :]  # (N, half, H)
    half_b = samples[:, perm_idx[half : 2 * half], :]  # (N, half, H)
    diff2 = half_a - half_b  # (N, half, H)
    norm2 = np.linalg.norm(diff2, axis=2)  # (N, half)
    term2 = 0.5 * np.mean(norm2)  # scalar

    return float(term1 - term2)


# ===========================================================================
# 3.  Pinball / Quantile Loss
# ===========================================================================


def compute_pinball(
    y_true: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
    quantile_levels: List[float],
) -> Dict[float, float]:
    """Compute mean pinball loss for each quantile level.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H) or (N,)
        Observed values.
    quantile_preds : dict {tau: np.ndarray shape (N, H) or (N,)}
        Predicted quantiles for each level tau.
    quantile_levels : list of float
        Quantile levels, e.g. [0.05, 0.10, …, 0.95].

    Returns
    -------
    dict {tau: float}
        Mean pinball loss per quantile level.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    results: Dict[float, float] = {}

    for tau in quantile_levels:
        if tau not in quantile_preds:
            continue
        q = np.asarray(quantile_preds[tau], dtype=np.float64)
        err = y_true - q
        # pinball(tau, y, q) = tau * max(err, 0) + (1 - tau) * max(-err, 0)
        loss = np.where(err >= 0, tau * err, (tau - 1.0) * err)
        results[tau] = float(np.mean(loss))

    return results


# ===========================================================================
# 4.  Winkler Interval Score
# ===========================================================================


def compute_winkler_score(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float,
) -> float:
    """Compute the mean Winkler interval score.

    For a (1 - alpha) prediction interval [q_lo, q_hi]::

        WS = (q_hi - q_lo)
             + (2/alpha)(q_lo - y) * 1{y < q_lo}
             + (2/alpha)(y - q_hi) * 1{y > q_hi}

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, H)
        Observed values.
    q_lo : np.ndarray, same shape as y_true
        Lower bound of the prediction interval.
    q_hi : np.ndarray, same shape as y_true
        Upper bound of the prediction interval.
    alpha : float
        Miscoverage level, e.g. 0.10 for a 90% PI.

    Returns
    -------
    float
        Mean Winkler score (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_lo = np.asarray(q_lo, dtype=np.float64)
    q_hi = np.asarray(q_hi, dtype=np.float64)

    width = q_hi - q_lo
    pen_lo = (2.0 / alpha) * np.maximum(q_lo - y_true, 0.0)
    pen_hi = (2.0 / alpha) * np.maximum(y_true - q_hi, 0.0)

    return float(np.mean(width + pen_lo + pen_hi))


# ===========================================================================
# 5.  Diebold-Mariano Test
# ===========================================================================


def dm_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 24,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0, where d_t = loss1(t) - loss2(t).

    The long-run variance is estimated via the Newey-West HAC estimator
    with bandwidth h (matching the forecast horizon)::

        gamma_0 = Var(d)
        gamma_j = Cov(d[j:], d[:-j])   for j = 1, ..., h
        sigma_NW^2 = gamma_0 + 2 * sum_{j=1}^{h} (1 - j/(h+1)) * gamma_j

        DM = d_bar / sqrt(sigma_NW^2 / T)

    Parameters
    ----------
    loss1 : np.ndarray, shape (N,)
        Per-observation loss for model 1.
    loss2 : np.ndarray, shape (N,)
        Per-observation loss for model 2.
    h : int
        Forecast horizon in hours; used as HAC lag truncation. Default 24.

    Returns
    -------
    dm_stat : float
        Diebold-Mariano test statistic.
    p_value : float
        Two-tailed p-value under the standard normal distribution.
    """
    loss1 = np.asarray(loss1, dtype=np.float64)
    loss2 = np.asarray(loss2, dtype=np.float64)

    d = loss1 - loss2
    T = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance
    d_centred = d - d_bar
    gamma_0 = np.var(d, ddof=0)

    sigma_nw = gamma_0
    for j in range(1, h + 1):
        # Sample autocovariance at lag j (biased estimator, consistent with NW)
        gamma_j = np.mean(d_centred[j:] * d_centred[:-j])
        weight = 1.0 - j / (h + 1.0)
        sigma_nw += 2.0 * weight * gamma_j

    # Guard against numerical zero / negative variance
    sigma_nw = max(sigma_nw, 1e-12)

    dm_stat = d_bar / np.sqrt(sigma_nw / T)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


# ===========================================================================
# 6.  EPFEvaluator — full evaluation pipeline
# ===========================================================================


class EPFEvaluator:
    """End-to-end evaluator for the LF-GP-NRF electricity price forecasting model.

    Parameters
    ----------
    model :
        Trained LFGPNRFModel (or any model exposing ``predict_distribution``).
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding dicts with keys ``'history'``, ``'future_exog'``,
        ``'target_raw'``.
    device : str
        Torch device string.  Default ``'cuda'``.
    scaling_constant : float
        Asinh scaling constant c used in ``price = c * sinh(p_tilde)``.
        Default 50.0.
    """

    # Quantile levels used throughout the evaluation
    QUANTILE_LEVELS: List[float] = [round(q * 0.05, 2) for q in range(1, 20)]
    # [0.05, 0.10, ..., 0.95]

    def __init__(
        self,
        model,
        test_loader,
        device: str = "cuda",
        scaling_constant: float = 50.0,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.scaling_constant = scaling_constant

    # ------------------------------------------------------------------
    # 6.1  Run inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> Dict[str, np.ndarray]:
        """Iterate over the test DataLoader and collect full predictive samples.

        Calls ``model.predict_distribution(history, future_exog)`` which
        returns a dict with keys ``'samples'``, ``'quantiles'``, ``'mean'``,
        ``'gp_mean'``, ``'gp_var'``, ``'z_paths'``.  This method unpacks
        that dict and stacks results across all batches.

        Returns
        -------
        dict with keys:
            ``'samples'``   : np.ndarray (N_test, S, 24)  EUR/MWh
            ``'y_true'``    : np.ndarray (N_test, 24)      EUR/MWh
            ``'mean'``      : np.ndarray (N_test, 24)      EUR/MWh
            ``'quantiles'`` : dict[float, np.ndarray (N_test, 24)]
            ``'z_paths'``   : np.ndarray (N_test, P, 24, D) latent forces
            ``'gp_mean'``   : np.ndarray (N_test, P, 24)
            ``'gp_var'``    : np.ndarray (N_test, P, 24)
        """
        self.model.eval()
        all_samples: List[np.ndarray] = []
        all_y_true: List[np.ndarray] = []
        all_mean: List[np.ndarray] = []
        all_z_paths: List[np.ndarray] = []
        all_gp_mean: List[np.ndarray] = []
        all_gp_var: List[np.ndarray] = []
        # quantile accumulator: tau -> list of (B, 24) arrays
        quant_accum: Dict[float, List[np.ndarray]] = {}

        for batch in self.test_loader:
            history = batch["history"].to(self.device)  # (B, T_hist, F)
            future_exog = batch["future_exog"].to(self.device)  # (B, 24, F_exog)

            # target_raw is already in EUR/MWh (inverse-asinh applied in dataset)
            y_true_batch = batch["target_raw"].cpu().numpy()  # (B, 24)

            # predict_distribution returns a rich dict
            pred = self.model.predict_distribution(history, future_exog)

            # --- samples (B, S, 24) ---
            samples_t = pred["samples"]
            if isinstance(samples_t, torch.Tensor):
                samples_t = samples_t.cpu().numpy()
            all_samples.append(samples_t.astype(np.float32))

            # --- forecast mean (B, 24) ---
            mean_t = pred["mean"]
            if isinstance(mean_t, torch.Tensor):
                mean_t = mean_t.cpu().numpy()
            all_mean.append(mean_t.astype(np.float32))

            # --- quantiles dict[float -> (B, 24)] ---
            for tau, q_tensor in pred["quantiles"].items():
                tau_key = round(float(tau), 2)
                if tau_key not in quant_accum:
                    quant_accum[tau_key] = []
                q_arr = (
                    q_tensor.cpu().numpy()
                    if isinstance(q_tensor, torch.Tensor)
                    else q_tensor
                )
                quant_accum[tau_key].append(q_arr.astype(np.float32))

            # --- latent forces (B, P, 24, D) ---
            z_t = pred["z_paths"]
            if isinstance(z_t, torch.Tensor):
                z_t = z_t.cpu().numpy()
            all_z_paths.append(z_t.astype(np.float32))

            # --- GP mean/var (B, P, 24) ---
            gpm = pred["gp_mean"]
            gpv = pred["gp_var"]
            if isinstance(gpm, torch.Tensor):
                gpm = gpm.cpu().numpy()
            if isinstance(gpv, torch.Tensor):
                gpv = gpv.cpu().numpy()
            all_gp_mean.append(gpm.astype(np.float32))
            all_gp_var.append(gpv.astype(np.float32))

            all_y_true.append(y_true_batch.astype(np.float32))

        samples_all = np.concatenate(all_samples, axis=0)  # (N, S, 24)
        y_true_all = np.concatenate(all_y_true, axis=0)  # (N, 24)
        mean_all = np.concatenate(all_mean, axis=0)  # (N, 24)
        z_paths_all = np.concatenate(all_z_paths, axis=0)  # (N, P, 24, D)
        gp_mean_all = np.concatenate(all_gp_mean, axis=0)  # (N, P, 24)
        gp_var_all = np.concatenate(all_gp_var, axis=0)  # (N, P, 24)
        quantiles_all = {
            tau: np.concatenate(arrays, axis=0)  # (N, 24)
            for tau, arrays in quant_accum.items()
        }

        return {
            "samples": samples_all,
            "y_true": y_true_all,
            "mean": mean_all,
            "quantiles": quantiles_all,
            "z_paths": z_paths_all,
            "gp_mean": gp_mean_all,
            "gp_var": gp_var_all,
        }

    # ------------------------------------------------------------------
    # 6.2  Compute all metrics
    # ------------------------------------------------------------------

    def compute_all_metrics(
        self,
        samples_all: np.ndarray,
        y_true_all: np.ndarray,
    ) -> Dict[str, float]:
        """Compute the full metric suite described in the MPhil research plan.

        Parameters
        ----------
        samples_all : np.ndarray, shape (N, S, 24)
            Full ensemble of predictive price samples in EUR/MWh.
        y_true_all : np.ndarray, shape (N, 24)
            Observed 24-hour price vectors in EUR/MWh.

        Returns
        -------
        dict
            All scalar metric values keyed by string names.
        """
        samples_all = np.asarray(samples_all, dtype=np.float64)
        y_true_all = np.asarray(y_true_all, dtype=np.float64)

        N, S, H = samples_all.shape  # (days, samples, hours)

        # ------------------------------------------------------------------
        # Flatten to per-observation arrays for scalar metrics
        # y_flat: (N*H,),  samples_flat: (N*H, S)
        # ------------------------------------------------------------------
        y_flat = y_true_all.reshape(-1)  # (N*H,)
        samples_flat = samples_all.transpose(0, 2, 1).reshape(N * H, S)
        # Note: transpose makes shape (N, H, S) then flatten N,H → (N*H, S)

        sample_mean_flat = samples_flat.mean(axis=1)  # (N*H,)

        # ------------------------------------------------------------------
        # CRPS — overall
        # ------------------------------------------------------------------
        crps_all = compute_crps_samples(y_flat, samples_flat)  # (N*H,)
        metrics: Dict[str, float] = {}
        metrics["crps_mean"] = float(np.mean(crps_all))

        # ------------------------------------------------------------------
        # CRPS — conditional on price regime
        # ------------------------------------------------------------------
        spike_mask = y_flat > 200.0
        neg_mask = y_flat < 0.0
        base_mask = (y_flat >= 0.0) & (y_flat <= 200.0)

        metrics["crps_spike"] = (
            float(np.mean(crps_all[spike_mask])) if spike_mask.any() else float("nan")
        )
        metrics["crps_negative"] = (
            float(np.mean(crps_all[neg_mask])) if neg_mask.any() else float("nan")
        )
        metrics["crps_base"] = (
            float(np.mean(crps_all[base_mask])) if base_mask.any() else float("nan")
        )

        # ------------------------------------------------------------------
        # Energy Score — 24-dim joint
        # ------------------------------------------------------------------
        metrics["energy_score"] = compute_energy_score(y_true_all, samples_all)

        # ------------------------------------------------------------------
        # MAE / RMSE — using sample mean as point forecast
        # ------------------------------------------------------------------
        mae_flat = np.abs(sample_mean_flat - y_flat)
        metrics["mae"] = float(np.mean(mae_flat))
        metrics["rmse"] = float(np.sqrt(np.mean((sample_mean_flat - y_flat) ** 2)))

        # MAE on spike hours
        metrics["mae_spike"] = (
            float(np.mean(mae_flat[spike_mask])) if spike_mask.any() else float("nan")
        )

        # ------------------------------------------------------------------
        # Compute sample quantiles — shape (N*H, n_quantiles)
        # ------------------------------------------------------------------
        q_levels = self.QUANTILE_LEVELS
        quantile_matrix = np.quantile(samples_flat, q_levels, axis=1).T
        # quantile_matrix: (N*H, n_quantiles)

        quantile_preds = {tau: quantile_matrix[:, i] for i, tau in enumerate(q_levels)}

        # ------------------------------------------------------------------
        # Pinball losses
        # ------------------------------------------------------------------
        pinball_results = compute_pinball(y_flat, quantile_preds, q_levels)
        for tau, val in pinball_results.items():
            key = f"pinball_{int(round(tau * 100)):03d}"
            metrics[key] = val

        # ------------------------------------------------------------------
        # Winkler scores
        # ------------------------------------------------------------------
        # 90% PI: q_5 to q_95  (alpha = 0.10)
        q05 = quantile_preds.get(0.05, quantile_matrix[:, 0])
        q95 = quantile_preds.get(0.95, quantile_matrix[:, -1])
        metrics["winkler_90"] = compute_winkler_score(y_flat, q05, q95, alpha=0.10)

        # 50% PI: q_25 to q_75  (alpha = 0.50)
        q25 = quantile_preds.get(0.25, quantile_matrix[:, q_levels.index(0.25)])
        q75 = quantile_preds.get(0.75, quantile_matrix[:, q_levels.index(0.75)])
        metrics["winkler_50"] = compute_winkler_score(y_flat, q25, q75, alpha=0.50)

        # ------------------------------------------------------------------
        # Empirical coverage
        # ------------------------------------------------------------------
        metrics["coverage_90"] = float(np.mean((y_flat >= q05) & (y_flat <= q95)))
        metrics["coverage_50"] = float(np.mean((y_flat >= q25) & (y_flat <= q75)))

        return metrics

    # ------------------------------------------------------------------
    # 6.3  Plot: forecast examples
    # ------------------------------------------------------------------

    def plot_forecast_examples(
        self,
        samples_all: np.ndarray,
        y_true_all: np.ndarray,
        save_dir: str,
        n_examples: int = 6,
    ) -> None:
        """Plot n_examples random forecast days with prediction intervals.

        Saves to ``save_dir/forecast_examples.png``.

        Parameters
        ----------
        samples_all : np.ndarray, shape (N, S, 24)
        y_true_all  : np.ndarray, shape (N, 24)
        save_dir    : str
        n_examples  : int, default 6
        """
        os.makedirs(save_dir, exist_ok=True)

        samples_all = np.asarray(samples_all, dtype=np.float64)
        y_true_all = np.asarray(y_true_all, dtype=np.float64)

        N = samples_all.shape[0]
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(N, size=min(n_examples, N), replace=False)
        idx = np.sort(idx)

        n_cols = 3
        n_rows = int(np.ceil(n_examples / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            sharex=True,
        )
        axes = np.array(axes).flatten()

        hours = np.arange(1, 25)

        for plot_i, day_i in enumerate(idx):
            ax = axes[plot_i]
            samps = samples_all[day_i]  # (S, 24)
            y = y_true_all[day_i]  # (24,)

            pred_mean = samps.mean(axis=0)
            q05 = np.quantile(samps, 0.05, axis=0)
            q25 = np.quantile(samps, 0.25, axis=0)
            q75 = np.quantile(samps, 0.75, axis=0)
            q95 = np.quantile(samps, 0.95, axis=0)

            # 90% PI — light shading
            ax.fill_between(
                hours,
                q05,
                q95,
                alpha=0.25,
                color=_PALETTE[0],
                label="90% PI",
            )
            # 50% PI — darker shading
            ax.fill_between(
                hours,
                q25,
                q75,
                alpha=0.50,
                color=_PALETTE[0],
                label="50% PI",
            )
            # Predictive mean
            ax.plot(
                hours,
                pred_mean,
                color=_PALETTE[0],
                lw=1.8,
                label="Pred. mean",
                zorder=3,
            )
            # Actual price
            ax.plot(
                hours, y, color=_PALETTE[1], lw=1.8, ls="--", label="Actual", zorder=4
            )

            ax.set_title(f"Day {day_i}", fontsize=10)
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Price (EUR/MWh)")

        # Hide unused subplots
        last_plot_i = len(idx) - 1
        for k in range(last_plot_i + 1, len(axes)):
            axes[k].set_visible(False)

        # Shared legend
        handles = [
            mpatches.Patch(color=_PALETTE[0], alpha=0.25, label="90% PI"),
            mpatches.Patch(color=_PALETTE[0], alpha=0.50, label="50% PI"),
            plt.Line2D([0], [0], color=_PALETTE[0], lw=1.8, label="Pred. mean"),
            plt.Line2D([0], [0], color=_PALETTE[1], lw=1.8, ls="--", label="Actual"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle("LF-GP-NRF — Forecast Examples", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])

        out_path = os.path.join(save_dir, "forecast_examples.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EPFEvaluator] Saved forecast examples → {out_path}")

    # ------------------------------------------------------------------
    # 6.4  Plot: CRPS by hour-of-day
    # ------------------------------------------------------------------

    def plot_crps_by_hour(
        self,
        samples_all: np.ndarray,
        y_true_all: np.ndarray,
        save_dir: str,
    ) -> None:
        """Bar chart of mean CRPS for each of the 24 hours in the day.

        Saves to ``save_dir/crps_by_hour.png``.

        Parameters
        ----------
        samples_all : np.ndarray, shape (N, S, 24)
        y_true_all  : np.ndarray, shape (N, 24)
        save_dir    : str
        """
        os.makedirs(save_dir, exist_ok=True)

        samples_all = np.asarray(samples_all, dtype=np.float64)
        y_true_all = np.asarray(y_true_all, dtype=np.float64)

        N, S, H = samples_all.shape
        crps_by_hour = np.zeros(H)

        for h in range(H):
            y_h = y_true_all[:, h]  # (N,)
            samps_h = samples_all[:, :, h]  # (N, S)
            crps_by_hour[h] = float(np.mean(compute_crps_samples(y_h, samps_h)))

        fig, ax = plt.subplots(figsize=(11, 4))
        hours = np.arange(H)
        bars = ax.bar(
            hours,
            crps_by_hour,
            color=_PALETTE[0],
            edgecolor="white",
            linewidth=0.5,
        )

        # Colour-code top-5 worst hours in a contrasting palette
        worst5 = np.argsort(crps_by_hour)[-5:]
        for h in worst5:
            bars[h].set_color(_PALETTE[3])

        ax.set_xticks(hours)
        ax.set_xticklabels([str(h) for h in range(H)], fontsize=8)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean CRPS (EUR/MWh)")
        ax.set_title("LF-GP-NRF — Hourly CRPS Profile", fontweight="bold")

        # Annotate mean line
        mean_crps = crps_by_hour.mean()
        ax.axhline(
            mean_crps,
            color="k",
            ls="--",
            lw=1.2,
            alpha=0.7,
            label=f"Daily mean = {mean_crps:.2f}",
        )
        ax.legend(fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(save_dir, "crps_by_hour.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EPFEvaluator] Saved CRPS-by-hour → {out_path}")

    # ------------------------------------------------------------------
    # 6.5  Plot: latent force trajectories
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plot_latent_forces(
        self,
        model,
        sample_batch: Dict[str, torch.Tensor],
        save_dir: str,
    ) -> None:
        """Visualise the latent SDE force trajectories z_1, z_2, z_3.

        Selects three representative days from `sample_batch`:
        - one "normal" day  (all prices in [0, 100] EUR),
        - one "spike" day   (any price > 200 EUR),
        - one "negative" day (any price < 0 EUR).

        If a category has no matching day the first available day is used.

        Saves to ``save_dir/latent_forces.png``.

        Parameters
        ----------
        model :
            Trained LFGPNRFModel exposing ``encoder``, ``latent_sde``, and
            ``gp_likelihood``.  The method gracefully falls back to a simple
            call if those sub-modules are not found.
        sample_batch : dict
            A DataLoader batch (keys: ``'history'``, ``'future_exog'``,
            ``'target_raw'``).
        save_dir : str
        """
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        history = sample_batch["history"].to(self.device)
        future_exog = sample_batch["future_exog"].to(self.device)
        target_raw = sample_batch["target_raw"].cpu().numpy()  # (B, 24)

        B = history.shape[0]

        # --- Identify example day indices -----------------------------------
        def _find_day(condition_fn, fallback=0):
            for i in range(B):
                if condition_fn(target_raw[i]):
                    return i
            return fallback

        day_normal = _find_day(lambda y: np.all((y >= 0) & (y <= 100)))
        day_spike = _find_day(lambda y: np.any(y > 200))
        day_neg = _find_day(lambda y: np.any(y < 0))

        example_days = {
            "Normal (0–100 €)": day_normal,
            "Spike (>200 €)": day_spike,
            "Negative (<0 €)": day_neg,
        }

        # --- Extract z_paths ------------------------------------------------
        try:
            enc_out = model.encoder(history)
            z0_mean = enc_out["z0_mean"]
            z0_logvar = enc_out["z0_logvar"]
            context = enc_out["context"]

            sde_out = model.latent_sde(
                z0_mean, z0_logvar, context, future_exog, n_paths=8
            )
            z_paths = sde_out["z_paths"].cpu().numpy()
            # z_paths: (B, n_paths, 24, latent_dim)
        except AttributeError:
            print(
                "[EPFEvaluator] plot_latent_forces: model sub-modules not "
                "found; skipping latent force plot."
            )
            return

        n_force_dims = min(z_paths.shape[-1], 3)  # z_1, z_2, z_3
        hours = np.arange(1, 25)

        fig, axes = plt.subplots(
            len(example_days),
            n_force_dims,
            figsize=(5 * n_force_dims, 3.5 * len(example_days)),
            sharex=True,
        )
        # Ensure 2-D indexing even for a single row
        if len(example_days) == 1:
            axes = axes[np.newaxis, :]
        if n_force_dims == 1:
            axes = axes[:, np.newaxis]

        dim_labels = [f"$z_{d + 1}(t)$" for d in range(n_force_dims)]
        day_colors = [_PALETTE[0], _PALETTE[3], _PALETTE[1]]

        for row_i, (label, day_i) in enumerate(example_days.items()):
            z_day = z_paths[day_i]  # (n_paths, 24, latent_dim)
            col_color = day_colors[row_i % len(day_colors)]

            for col_i in range(n_force_dims):
                ax = axes[row_i, col_i]
                z_dim = z_day[:, :, col_i]  # (n_paths, 24)

                z_mean = z_dim.mean(axis=0)
                z_q10 = np.quantile(z_dim, 0.10, axis=0)
                z_q90 = np.quantile(z_dim, 0.90, axis=0)

                # Individual paths (transparent)
                for p in range(z_dim.shape[0]):
                    ax.plot(hours, z_dim[p], color=col_color, alpha=0.15, lw=0.8)

                # Mean path
                ax.plot(hours, z_mean, color=col_color, lw=2.0, label="Mean path")
                # 80% band
                ax.fill_between(
                    hours, z_q10, z_q90, color=col_color, alpha=0.25, label="80% band"
                )

                ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.5)
                ax.set_ylabel(dim_labels[col_i], fontsize=10)
                ax.set_xlabel("Hour", fontsize=9)

                if col_i == 0:
                    ax.set_title(label, fontsize=10, fontweight="bold", loc="left")

        fig.suptitle(
            "LF-GP-NRF — Latent Force Trajectories", fontsize=13, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(save_dir, "latent_forces.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EPFEvaluator] Saved latent forces → {out_path}")

    # ------------------------------------------------------------------
    # 6.6  Save results table
    # ------------------------------------------------------------------

    def save_results_table(
        self,
        metrics_dict: Dict[str, float],
        save_dir: str,
        model_name: str = "LF-GP-NRF",
    ) -> None:
        """Persist the metric dictionary as CSV and LaTeX.

        Parameters
        ----------
        metrics_dict : dict
            Output of ``compute_all_metrics``.
        save_dir : str
            Root output directory; a ``tables/`` sub-directory is created.
        model_name : str
            Used as suffix in the filename and as the column header.
        """
        table_dir = os.path.join(save_dir, "tables")
        os.makedirs(table_dir, exist_ok=True)

        safe_name = model_name.replace(" ", "_").replace("/", "-")

        # Build a tidy DataFrame: one row per metric
        df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=[model_name])
        df.index.name = "Metric"

        # CSV
        csv_path = os.path.join(table_dir, f"results_{safe_name}.csv")
        df.to_csv(csv_path, float_format="%.4f")
        print(f"[EPFEvaluator] Saved CSV table → {csv_path}")

        # LaTeX — formatted for easy inclusion in a thesis
        latex_path = os.path.join(table_dir, f"results_{safe_name}.tex")
        latex_str = _df_to_latex(
            df,
            caption=f"Evaluation metrics for {model_name} on the held-out test set.",
            label=f"tab:results_{safe_name.lower()}",
        )
        with open(latex_path, "w", encoding="utf-8") as fh:
            fh.write(latex_str)
        print(f"[EPFEvaluator] Saved LaTeX table → {latex_path}")


# ===========================================================================
# 7.  compare_models — cross-model summary
# ===========================================================================


def compare_models(results_dict: Dict[str, Dict[str, float]], save_dir: str) -> None:
    """Summarise and compare metrics across multiple models.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: metrics_dict}``.  Each ``metrics_dict`` is the output
        of ``EPFEvaluator.compute_all_metrics``.
    save_dir : str
        Directory where comparison outputs are saved.

    Outputs
    -------
    - ``save_dir/tables/model_comparison.csv``
    - ``save_dir/tables/model_comparison.tex``
    - ``save_dir/model_comparison.png``
    """
    os.makedirs(save_dir, exist_ok=True)
    table_dir = os.path.join(save_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # Build summary DataFrame: models as columns, metrics as rows
    df = pd.DataFrame(results_dict).T  # (n_models, n_metrics)
    df.index.name = "Model"

    # CSV
    csv_path = os.path.join(table_dir, "model_comparison.csv")
    df.to_csv(csv_path, float_format="%.4f")
    print(f"[compare_models] Saved comparison CSV → {csv_path}")

    # LaTeX
    tex_path = os.path.join(table_dir, "model_comparison.tex")
    # Select a readable subset for the thesis table
    display_cols = [
        "crps_mean",
        "crps_spike",
        "crps_negative",
        "energy_score",
        "mae",
        "rmse",
        "winkler_90",
        "coverage_90",
        "coverage_50",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    latex_str = _df_to_latex(
        df[display_cols].T,  # metrics as rows, models as columns
        caption="Comparative evaluation of LF-GP-NRF and baseline models.",
        label="tab:model_comparison",
    )
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    print(f"[compare_models] Saved comparison LaTeX → {tex_path}")

    # ------------------------------------------------------------------
    # Bar chart: CRPS overall + spike + negative
    # ------------------------------------------------------------------
    crps_cols = ["crps_mean", "crps_spike", "crps_negative"]
    crps_cols_present = [c for c in crps_cols if c in df.columns]

    crps_df = df[crps_cols_present].copy()
    crps_df.columns = [c.replace("crps_", "CRPS ").title() for c in crps_df.columns]

    n_models = len(crps_df)
    n_groups = len(crps_df.columns)
    bar_width = 0.8 / n_groups
    x = np.arange(n_models)

    palette = sns.color_palette("muted", n_colors=n_groups)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_models), 5))

    for gi, col in enumerate(crps_df.columns):
        offset = (gi - n_groups / 2.0 + 0.5) * bar_width
        vals = crps_df[col].values.astype(float)
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.92,
            color=palette[gi],
            label=col,
            edgecolor="white",
            linewidth=0.6,
        )
        # Annotate bar tops
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.3,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(list(crps_df.index), rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("CRPS (EUR/MWh)", fontsize=11)
    ax.set_title(
        "Model Comparison — CRPS by Price Regime", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_models] Saved comparison chart → {out_path}")


# ===========================================================================
# 8.  Internal helpers
# ===========================================================================


def _df_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_fmt: str = "{:.4f}",
) -> str:
    """Render a DataFrame as a publication-quality LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
    caption : str
    label : str
    float_fmt : str

    Returns
    -------
    str
        Full LaTeX table (``table`` + ``tabular`` environment).
    """
    n_cols = len(df.columns)
    col_spec = "l" + "r" * n_cols

    def _fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return float_fmt.format(v)
        return str(v)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Header row
    header = "    " + df.index.name if df.index.name else "    Metric"
    header += " & " + " & ".join(str(c) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    # Data rows
    for idx_val, row in df.iterrows():
        row_str = "    " + str(idx_val)
        row_str += " & " + " & ".join(_fmt(v) for v in row.values)
        row_str += r" \\"
        lines.append(row_str)

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


# ===========================================================================
# 9.  Module self-test (no model required)
# ===========================================================================


def _self_test() -> None:
    """Quick sanity-check of the standalone metric functions."""
    rng = np.random.default_rng(0)
    N, S, H = 50, 200, 24

    y_1d = rng.normal(50, 30, size=(N,))
    samps_1d = rng.normal(50, 30, size=(N, S))
    y_2d = rng.normal(50, 30, size=(N, H))
    samps_3d = rng.normal(50, 30, size=(N, S, H))

    # CRPS
    crps = compute_crps_samples(y_1d, samps_1d)
    assert crps.shape == (N,), f"CRPS shape mismatch: {crps.shape}"
    assert np.all(crps >= 0), "CRPS must be non-negative"
    print(
        f"  CRPS mean = {crps.mean():.4f}  (expected ~{np.abs(rng.normal(0, 1, 10000)).mean():.2f}*30)"
    )

    # Energy Score
    es = compute_energy_score(y_2d, samps_3d)
    assert isinstance(es, float) and es >= 0, f"ES invalid: {es}"
    print(f"  Energy Score = {es:.4f}")

    # Pinball
    q_preds = {tau: np.quantile(samps_1d, tau, axis=1) for tau in [0.1, 0.5, 0.9]}
    pb = compute_pinball(y_1d, q_preds, [0.1, 0.5, 0.9])
    assert set(pb.keys()) == {0.1, 0.5, 0.9}
    print(f"  Pinball (0.5) = {pb[0.5]:.4f}")

    # Winkler
    q05 = np.quantile(samps_1d, 0.05, axis=1)
    q95 = np.quantile(samps_1d, 0.95, axis=1)
    ws = compute_winkler_score(y_1d, q05, q95, alpha=0.10)
    assert isinstance(ws, float) and ws >= 0
    print(f"  Winkler (90%) = {ws:.4f}")

    # DM test — two identical loss series → DM stat ≈ 0
    loss_a = rng.standard_normal(N * H)
    dm_stat, p_val = dm_test(loss_a, loss_a, h=24)
    assert abs(dm_stat) < 1e-6, f"DM stat for equal losses should be ~0: {dm_stat}"
    # Two clearly different loss series
    loss_b = loss_a + 5.0
    dm_stat2, p_val2 = dm_test(loss_b, loss_a, h=24)
    assert dm_stat2 > 0, f"DM stat should be positive: {dm_stat2}"
    assert p_val2 < 0.01, f"Should reject H0 easily: p={p_val2}"
    print(f"  DM stat (diff series) = {dm_stat2:.4f}, p = {p_val2:.6f}")

    print("\n[evaluation.py] All self-tests passed.")


if __name__ == "__main__":
    _self_test()
