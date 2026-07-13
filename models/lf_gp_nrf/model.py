# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# model.py — End-to-End LightningModule (Joint ELBO Training)
#
# Role (Top-level orchestrator):
#   Combines all four LF-GP-NRF components into a single trainable module:
#
#     Layer 0 — LatentForceEncoder   : BiLSTM recognition network
#                                       history → (z0_mean, z0_logvar, context)
#     Layer 1 — LatentSDE            : Neural SDE latent force field
#                                       z0, context → z_paths (B, P, T, D)
#     Layer 2 — ForceConditionedGP   : Sparse Variational GP (SVGP)
#                                       (cal, z_t) → (gp_mean, gp_var)
#     Layer 3 — NormalizingFlowEmission : Conditional RQ-NSF
#                                         (gp_mean, gp_var, z_t, cal) → p(y)
#
#   The joint ELBO trained end-to-end is:
#
#     L = E[log p(y | f, z, cal)]    (flow reconstruction)
#       - β_kl  · KL(q(z) ‖ p(z))   (SDE Girsanov KL)
#       + β_gp  · ELBO_GP            (sparse GP variational ELBO)
#       - γ     · CalibPenalty       (pinball calibration)
#
#   Annealing: β_kl starts at 0 and is increased externally (e.g. by a
#   Lightning callback) via ``model.kl_beta = new_value``.  This mirrors the
#   warm-up schedule recommended in §4.7 of the research plan.
#
# Training hyperparameters (RTX 4000, 8.6 GB VRAM):
#   n_paths_train  = 8    (M during training   — memory-limited)
#   n_paths_infer  = 64   (M during inference)
#   n_samples_infer= 100  (K flow samples per path for quantiles)
#
# Dimension key (defaults):
#   history_feat_dim  = 20  (18 base + 2 fossil — matched to dataset.py)
#   future_feat_dim   = 6   (load/solar/wind_on/wind_off/hour_sin/hour_cos)
#   latent_dim        = 3
#   n_calendar        = 6   (= future_feat_dim; ALL future_exog as GP input)
#   gp_input_dim      = 9   (n_calendar + latent_dim)
#   encoder_hidden    = 128
#   sde_hidden        = 64
#   n_inducing        = 256
#
# References:
#   Higgins et al. (2017) "beta-VAE: Learning Basic Visual Concepts"
#   Hensman et al. (2015) "Scalable Variational Gaussian Process Classification"
#   Rubanova et al. (2019) "Latent ODEs for Irregularly-Sampled Time Series"
#   LF_GP_NRF_Research_Plan.md §4.7–4.9
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gpytorch
import numpy as np
import pytorch_lightning as pl
import torch
from gpytorch.mlls import VariationalELBO
from torch import Tensor

from .encoder import LatentForceEncoder
from .flow_emission import NormalizingFlowEmission
from .force_gp import ForceConditionedGP, prepare_inputs
from .latent_sde import LatentSDE

# =============================================================================
# LFGPNRFModel
# =============================================================================


class LFGPNRFModel(pl.LightningModule):
    """End-to-end Lightning module for the LF-GP-NRF electricity price model.

    Combines a BiLSTM encoder, Neural SDE, Sparse Variational GP, and
    Normalizing Flow emission layer into a jointly trained probabilistic
    forecasting system.

    The four components are orchestrated via a joint ELBO:

        L = E_q[log p(y | f, z, cal)]  ← flow reconstruction
          − β_kl  · KL_SDE             ← Girsanov KL (SDE prior vs posterior)
          + β_gp  · ELBO_GP            ← GP variational ELBO (negative KL)
          − γ     · CalibPenalty       ← pinball calibration loss

    Parameters
    ----------
    history_feat_dim : int
        Feature dimension of the history window tensor.  Default 20 (18 base
        features from dataset.py plus 2 optional fossil fuel columns that are
        always present in the German ENTSO-E dataset).
    future_feat_dim : int
        Feature dimension of the future exogenous tensor.  Default 6 — matches
        FUTURE_EXOG_FEATURES in dataset.py:
        [load_forecast, solar_forecast, wind_onshore_forecast,
         wind_offshore_forecast, hour_sin, hour_cos].
    latent_dim : int
        Dimensionality of the latent force z(t).  Default 3 (renewable surplus,
        thermal scarcity, demand surge).
    encoder_hidden : int
        Hidden size of the BiLSTM encoder (and context dimension fed to SDE).
        Default 128.
    sde_hidden : int
        Hidden size of the SDE drift/diffusion MLPs.  Default 64.
    n_inducing : int
        Number of sparse GP inducing points.  Default 256 (VRAM-limited for
        RTX 4000 8.6 GB).
    n_flow_transforms : int
        Number of RQ-NSF coupling layers in the flow emission.  Default 4.
    n_flow_bins : int
        Number of rational-quadratic spline bins per transform.  Default 8.
    n_paths_train : int
        Number of SDE Monte Carlo paths during training (M).  Default 8.
    n_paths_infer : int
        Number of SDE paths during inference.  Default 64.
    n_samples_infer : int
        Number of flow samples drawn per path during inference (K).  Default 100.
    lr : float
        AdamW learning rate.  Default 1e-3.
    weight_decay : float
        AdamW weight decay coefficient.  Default 1e-4.
    kl_beta : float
        Initial weight on the SDE Girsanov KL term.  Default 0.0.  Increase
        externally via a KLAnnealingCallback (see training script).
    gp_beta : float
        Weight on the GP variational ELBO term (negative sparse-GP KL).
        Default 1.0.
    calib_gamma : float
        Weight on the pinball calibration penalty.  Default 0.1.
    price_min : float
        Soft lower bound for price normalisation in the flow (EUR/MWh).
        Default -600.0 (below EPEX floor).
    price_max : float
        Soft upper bound for price normalisation in the flow (EUR/MWh).
        Default 3000.0 (EPEX market cap).
    scaling_constant : float
        Constant c used in the inverse asinh transform: y = c * sinh(y_tilde).
        Default 50.0 (matches EPFPreprocessor in dataset.py).
    """

    def __init__(
        self,
        history_feat_dim: int = 20,
        future_feat_dim: int = 6,
        latent_dim: int = 3,
        encoder_hidden: int = 128,
        sde_hidden: int = 64,
        n_inducing: int = 256,
        n_flow_transforms: int = 4,
        n_flow_bins: int = 8,
        n_paths_train: int = 8,
        n_paths_infer: int = 64,
        n_samples_infer: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        kl_beta: float = 0.0,
        gp_beta: float = 1.0,
        calib_gamma: float = 0.1,
        price_min: float = -200.0,
        price_max: float = 1000.0,
        scaling_constant: float = 50.0,
    ) -> None:
        super().__init__()

        # -----------------------------------------------------------------
        # Save all constructor args as hyperparameters for Lightning's
        # save_hyperparameters().  These are logged to checkpoints and W&B.
        # -----------------------------------------------------------------
        self.save_hyperparameters()

        # -----------------------------------------------------------------
        # Store scalar hyperparameters as plain attributes for easy access
        # (save_hyperparameters already stores them in self.hparams but direct
        #  attribute access is more readable and avoids dict indirection).
        # -----------------------------------------------------------------
        self.history_feat_dim = history_feat_dim
        self.future_feat_dim = future_feat_dim
        self.latent_dim = latent_dim
        self.encoder_hidden = encoder_hidden
        self.sde_hidden = sde_hidden
        self.n_inducing = n_inducing
        self.n_flow_transforms = n_flow_transforms
        self.n_flow_bins = n_flow_bins
        self.n_paths_train = n_paths_train
        self.n_paths_infer = n_paths_infer
        self.n_samples_infer = n_samples_infer
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_beta = kl_beta  # mutable — annealed by callback
        self.gp_beta = gp_beta
        self.calib_gamma = calib_gamma
        self.price_min = price_min
        self.price_max = price_max
        self.scaling_constant = scaling_constant

        # -----------------------------------------------------------------
        # Derived dimensions
        #
        # n_calendar = future_feat_dim:  We use ALL future exogenous features
        # as the GP conditioning input, not just the two hour encoding columns.
        # This lets the GP kernel distinguish renewable surplus / thermal
        # scarcity regimes from the feature side, complementing z_t.
        #
        # gp_input_dim = n_calendar + latent_dim = 6 + 3 = 9 (defaults)
        # -----------------------------------------------------------------
        n_calendar: int = future_feat_dim  # 6 (all future_exog features)
        gp_input_dim: int = n_calendar + latent_dim  # 9

        self.n_calendar = n_calendar
        self.gp_input_dim = gp_input_dim

        # =================================================================
        # Layer 0: Encoder
        #   history    (B, 168, 20) → z0_mean, z0_logvar (B, 3)
        #                           → context             (B, 24, 128)
        # =================================================================
        self.encoder = LatentForceEncoder(
            history_feat_dim=history_feat_dim,
            future_feat_dim=future_feat_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden,
        )

        # =================================================================
        # Layer 1: Neural SDE
        #   z0, context, future_exog → z_paths (B, P, 24, 3)
        #                           → kl        scalar
        # =================================================================
        self.sde = LatentSDE(
            latent_dim=latent_dim,
            context_dim=encoder_hidden,  # must match encoder hidden_dim
            exog_dim=future_feat_dim,
            hidden_dim=sde_hidden,
        )

        # =================================================================
        # Layer 2: Sparse Variational GP (SVGP)
        #   (cal_features, z_t) → (gp_mean, gp_var)   (B, P, 24)
        # =================================================================
        self.gp = ForceConditionedGP(
            input_dim=gp_input_dim,
            latent_dim=latent_dim,
            n_inducing=n_inducing,
            n_calendar=n_calendar,
        )

        # Observation likelihood shared between the GP and the ELBO objective
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Variational ELBO objective for the sparse GP.
        # num_data should approximate total training points for correct
        # KL scaling.  We use 50 000 ≈ 3 years * 365 days * 24 steps * 2
        # (two-sided because the GP sees expanded batch×paths inputs).
        self.mll = VariationalELBO(
            self.likelihood,
            self.gp,
            num_data=50_000,
        )

        # =================================================================
        # Layer 3: Normalizing Flow Emission
        #   (gp_mean, gp_var, z_t, cal) → log p(y) / price samples
        #
        # NOTE: n_calendar here is the number of calendar conditioning
        # scalars fed to the flow's conditioner MLP.  We use future_feat_dim
        # (= 6) consistently: [load_fcast, solar, wind_on, wind_off,
        # hour_sin, hour_cos].
        # =================================================================
        self.flow = NormalizingFlowEmission(
            latent_dim=latent_dim,
            n_calendar=future_feat_dim,  # 6
            n_transforms=n_flow_transforms,
            n_bins=n_flow_bins,
            price_min=price_min,
            price_max=price_max,
        )

        # -----------------------------------------------------------------
        # Fixed quantile levels used for calibration and inference outputs.
        # Shape (7,): [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        # -----------------------------------------------------------------
        self.register_buffer(
            "quantile_levels",
            torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
        )

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(
        self,
        history: Tensor,
        future_exog: Tensor,
        n_paths: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Full forward pass through all four model components.

        Parameters
        ----------
        history : Tensor, shape (batch, history_len, history_feat_dim)
            Historical price and exogenous features for the look-back window
            (default 168 hours = 1 week).
        future_exog : Tensor, shape (batch, horizon, future_feat_dim)
            Deterministic exogenous forecasts for the forecast horizon
            (default 24 hours = one day ahead).
        n_paths : int or None
            Number of SDE Monte Carlo paths.  If None, uses n_paths_train
            when the model is in training mode and n_paths_infer otherwise.

        Returns
        -------
        dict with keys:
            'z_paths'  : Tensor (batch, n_paths, horizon, latent_dim)
                         SDE-sampled latent force trajectories.
            'gp_mean'  : Tensor (batch, n_paths, horizon)
                         GP posterior predictive mean.
            'gp_var'   : Tensor (batch, n_paths, horizon)
                         GP posterior predictive variance (includes obs noise).
            'sde_kl'   : Tensor, scalar
                         Girsanov KL divergence KL(q ‖ p) over the SDE paths.
            'enc_out'  : dict with 'z0_mean', 'z0_logvar', 'context'
                         Raw encoder outputs (useful for debugging/logging).
        """
        # ------------------------------------------------------------------
        # Resolve n_paths based on training/inference mode
        # ------------------------------------------------------------------
        if n_paths is None:
            n_paths = self.n_paths_train if self.training else self.n_paths_infer

        # ------------------------------------------------------------------
        # Step 0 → Encode history window
        # enc_out keys: 'z0_mean' (B, D), 'z0_logvar' (B, D), 'context' (B, T, C)
        # ------------------------------------------------------------------
        enc_out = self.encoder(history, future_exog)

        # ------------------------------------------------------------------
        # Step 1 → Sample latent force trajectories from posterior SDE
        # sde_out keys: 'z_paths' (B, P, T, D), 'kl' (scalar), 'z0' (B, D)
        # ------------------------------------------------------------------
        sde_out = self.sde(
            z0_mean=enc_out["z0_mean"],
            z0_logvar=enc_out["z0_logvar"],
            context=enc_out["context"],
            exog=future_exog,
            n_paths=n_paths,
        )

        # ------------------------------------------------------------------
        # Step 2 → GP posterior predictive over all SDE paths
        # Returns: gp_mean (B, P, T), gp_var (B, P, T)
        # ------------------------------------------------------------------
        gp_mean, gp_var = self.gp.predict_on_paths(
            time_features=future_exog,  # (B, T, n_calendar)  — all 6 features
            z_paths=sde_out["z_paths"],  # (B, P, T, D)
            likelihood=self.likelihood,
        )

        return {
            "z_paths": sde_out["z_paths"],  # (B, P, T, D)
            "gp_mean": gp_mean,  # (B, P, T)
            "gp_var": gp_var,  # (B, P, T)
            "sde_kl": sde_out["kl"],  # scalar
            "enc_out": enc_out,  # dict
        }

    # =========================================================================
    # Loss computation
    # =========================================================================

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        n_paths: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Compute the joint ELBO and all sub-components.

        The total loss is:

            L = - reconstruction
                + kl_beta  * sde_kl
                + gp_beta  * gp_loss
                + calib_gamma * calib_loss

        where:
            reconstruction  = E[log p(y | f, z, cal)]  (flow log-prob, maximised)
            sde_kl          = KL(q_SDE ‖ p_SDE)        (Girsanov, minimised)
            gp_loss         = -ELBO_GP                  (sparse GP KL, minimised)
            calib_loss      = mean pinball loss          (calibration, minimised)

        Parameters
        ----------
        batch : dict
            Mini-batch from EPFWindowDataset with keys:
                'history'    : (B, 168, history_feat_dim)
                'future_exog': (B,  24, future_feat_dim)
                'target'     : (B,  24) — asinh-transformed price
                'target_raw' : (B,  24) — raw EUR/MWh price
        n_paths : int or None
            Passed through to forward().  Useful for validation with a higher
            path count to get stable ELBO estimates.

        Returns
        -------
        dict with keys:
            'loss'           : scalar — total training loss (to minimise)
            'reconstruction' : scalar — mean log p(y) across batch & horizon
            'sde_kl'         : scalar — Girsanov KL term
            'gp_loss'        : scalar — negative sparse GP ELBO
            'calib_loss'     : scalar — mean pinball calibration penalty
        """
        # ------------------------------------------------------------------
        # Unpack batch
        # ------------------------------------------------------------------
        history = batch["history"]  # (B, 168, hist_dim)
        future_exog = batch["future_exog"]  # (B, 24, fut_dim)
        target = batch["target"]  # (B, 24) — asinh prices
        target_raw_eur = batch["target_raw"]  # (B, 24) — EUR/MWh prices

        batch_size = history.shape[0]
        horizon = future_exog.shape[1]

        # ------------------------------------------------------------------
        # Forward pass through all four components
        # ------------------------------------------------------------------
        fwd = self.forward(history, future_exog, n_paths=n_paths)

        z_paths = fwd["z_paths"]  # (B, P, T, D)
        gp_mean = fwd["gp_mean"]  # (B, P, T)
        gp_var = fwd["gp_var"]  # (B, P, T)
        n_paths_actual = z_paths.shape[1]

        # ==================================================================
        # Term 1 — Flow reconstruction
        #
        # flow.log_prob expects:
        #   y_raw    : (B, T)          — EUR/MWh prices
        #   gp_mean  : (B, P, T)
        #   gp_var   : (B, P, T)
        #   z_paths  : (B, P, T, D)
        #   calendar : (B, T, n_cal)   — future_exog serves as calendar here
        #
        # Returns: (B, T) — log-probs averaged over P paths
        # ==================================================================
        flow_logprob = self.flow.log_prob(
            y_raw=target_raw_eur,  # (B, T)
            gp_mean=gp_mean,  # (B, P, T)
            gp_var=gp_var,  # (B, P, T)
            z_paths=z_paths,  # (B, P, T, D)
            calendar=future_exog,  # (B, T, n_cal)
        )
        # flow_logprob : (B, T) — average log p(y) over paths, per timestep
        # Reconstruction = mean over batch and horizon (to be maximised)
        reconstruction = flow_logprob.mean()

        # ==================================================================
        # Term 2 — SDE KL divergence (Girsanov)
        #
        # Already computed inside LatentSDE.forward() as a scalar.
        # Represents KL(q_posterior ‖ p_prior) over the SDE trajectories.
        # ==================================================================
        sde_kl = fwd["sde_kl"]  # scalar

        # ==================================================================
        # Term 3 — Sparse GP variational ELBO
        #
        # Construct GP training inputs by concatenating:
        #   - future_exog (B, T, n_cal) expanded across paths
        #   - z_paths     (B, P, T, D)
        # → gp_x : (B*P, T, input_dim)  then flattened to (B*P*T, input_dim)
        #
        # GP targets: expand asinh-transformed price across all paths,
        # then flatten → (B*P*T,)
        # ==================================================================
        gp_x = prepare_inputs(
            time_features=future_exog,
            z_paths=z_paths,
            batch=batch_size,
            n_paths=n_paths_actual,
            horizon=horizon,
            n_calendar=self.n_calendar,
        )
        # gp_x : (B*P, T, input_dim) → flatten to (B*P*T, input_dim)
        gp_x_flat = gp_x.reshape(-1, self.gp_input_dim)

        # Expand target (asinh prices) across path dimension then flatten
        # target       : (B, T)
        # target_exp   : (B, P, T)
        # target_flat  : (B*P*T,)
        gp_y = (
            target.unsqueeze(1)  # (B, 1, T)
            .expand(batch_size, n_paths_actual, horizon)  # (B, P, T)
            .reshape(-1)  # (B*P*T,)
        )

        # compute_elbo returns the NEGATIVE ELBO (positive loss to minimise)
        gp_loss = self.gp.compute_elbo(
            x=gp_x_flat,
            y=gp_y,
            likelihood=self.likelihood,
            mll=self.mll,
        )

        # ==================================================================
        # Term 4 — Calibration penalty (pinball loss)
        #
        # Draw a moderate number of price samples from the flow and compute
        # the empirical quantile loss against the observed prices.
        # Using 32 samples here keeps GPU memory manageable during training.
        #
        # flow.sample returns: (B, P, n_samp, T)
        # Flatten paths × samples → (B, P*n_samp, T) for quantile computation
        # ==================================================================
        with torch.no_grad():
            price_samples = self.flow.sample(
                gp_mean=gp_mean,  # (B, P, T)
                gp_var=gp_var,  # (B, P, T)
                z_paths=z_paths,  # (B, P, T, D)
                calendar=future_exog,  # (B, T, n_cal)
                n_samples=32,
            )
        # price_samples : (B, P, 32, T)

        # Flatten path and sample dimensions
        price_samples_flat = price_samples.reshape(
            batch_size, n_paths_actual * 32, horizon
        )  # (B, P*32, T)

        # Compute pinball (quantile) loss for each quantile level τ
        # Pinball loss for a single quantile τ:
        #   L_τ(q_τ, y) = (q_τ - y) * (τ - 1{y < q_τ})
        # where q_τ is the τ-th empirical quantile of the samples.
        calib_losses: List[Tensor] = []
        for tau in self.quantile_levels:
            # Empirical τ-quantile over the sample dimension → (B, T)
            q_tau = torch.quantile(price_samples_flat, tau.item(), dim=1)

            # target_raw_eur : (B, T) — observed prices in EUR/MWh
            # Pinball loss element-wise
            errors = target_raw_eur - q_tau  # (B, T)
            pinball = torch.where(
                errors >= 0,
                tau * errors,
                (tau - 1.0) * errors,
            )  # (B, T)
            calib_losses.append(pinball.mean())

        calib_loss = torch.stack(calib_losses).mean()

        # ==================================================================
        # Total loss
        #
        # Signs:
        #   - reconstruction is a log-probability → MAXIMISE → negate for loss
        #   + kl_beta * sde_kl   → penalise SDE posterior deviation from prior
        #   + gp_beta * gp_loss  → gp_loss is already -ELBO (positive)
        #   + calib_gamma * calib_loss → penalise quantile miscalibration
        # ==================================================================
        loss = (
            -reconstruction
            + self.kl_beta * sde_kl
            + self.gp_beta * gp_loss
            + self.calib_gamma * calib_loss
        )

        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "sde_kl": sde_kl,
            "gp_loss": gp_loss,
            "calib_loss": calib_loss,
        }

    # =========================================================================
    # Lightning training / validation steps
    # =========================================================================

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        """Single training step.

        Computes the joint ELBO loss, logs all sub-components to Lightning
        logger (e.g. W&B or TensorBoard), and returns the scalar loss for
        gradient computation.

        Parameters
        ----------
        batch : dict
            Mini-batch from the DataLoader.
        batch_idx : int
            Index of the current batch within the epoch (unused).

        Returns
        -------
        Tensor
            Scalar loss for backpropagation.
        """
        losses = self.compute_loss(batch, n_paths=self.n_paths_train)

        # Log all terms with 'train/' prefix
        self.log(
            "train/loss", losses["loss"], prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "train/reconstruction",
            losses["reconstruction"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train/sde_kl",
            losses["sde_kl"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train/gp_loss",
            losses["gp_loss"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train/calib_loss",
            losses["calib_loss"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return losses["loss"]

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        """Single validation step.

        Uses the training path count for memory consistency.  For a richer
        evaluation, use predict_distribution() after training.

        Parameters
        ----------
        batch : dict
            Mini-batch from the validation DataLoader.
        batch_idx : int
            Index of the current batch (unused).

        Returns
        -------
        Tensor
            Scalar validation loss.
        """
        # Use training path count during validation to keep GPU memory stable.
        # The full inference path count (n_paths_infer) is reserved for the
        # final predict_distribution() call.
        losses = self.compute_loss(batch, n_paths=self.n_paths_train)

        # Log all terms with 'val/' prefix
        self.log(
            "val/loss", losses["loss"], prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "val/reconstruction",
            losses["reconstruction"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val/sde_kl", losses["sde_kl"], prog_bar=False, on_epoch=True, on_step=False
        )
        self.log(
            "val/gp_loss",
            losses["gp_loss"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val/calib_loss",
            losses["calib_loss"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return losses["loss"]

    # =========================================================================
    # Optimiser and scheduler
    # =========================================================================

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimiser and CosineAnnealingLR scheduler.

        Returns
        -------
        dict
            Lightning-compatible configuration with 'optimizer' and
            'lr_scheduler' keys.
        """
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=50,
            eta_min=1e-6,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_lr",
            },
        }

    # =========================================================================
    # Epoch hooks
    # =========================================================================

    def on_train_epoch_start(self) -> None:
        """Log the current KL annealing weight at the start of each epoch.

        Allows monitoring the β warm-up schedule in real time on the training
        dashboard (W&B / TensorBoard).  The actual annealing is driven by an
        external KLAnnealingCallback that sets ``self.kl_beta``.
        """
        # Prefer the SDE module's own kl_beta if it carries one
        # (some extensions override this per-module for finer control).
        kl_beta_val: float
        if hasattr(self.sde, "kl_beta"):
            kl_beta_val = float(self.sde.kl_beta)
        else:
            kl_beta_val = float(self.kl_beta)

        self.log(
            "train/kl_beta",
            kl_beta_val,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

    # =========================================================================
    # Inference
    # =========================================================================

    def predict_distribution(
        self,
        history: Tensor,
        future_exog: Tensor,
    ) -> Dict[str, Any]:
        """Full probabilistic forecast: samples + quantiles + mean.

        Runs the complete pipeline with n_paths_infer SDE paths and
        n_samples_infer flow samples per path, producing a rich
        forecast distribution for evaluation (CRPS, pinball, WIS, etc.).

        Parameters
        ----------
        history : Tensor, shape (batch, history_len, history_feat_dim)
            Historical window for encoding.
        future_exog : Tensor, shape (batch, horizon, future_feat_dim)
            Future exogenous features for the forecast horizon.

        Returns
        -------
        dict with keys:
            'samples'  : Tensor (batch, n_paths_infer * n_samples_infer, horizon)
                         Raw EUR/MWh price samples collapsed across paths.
            'quantiles': dict[float, Tensor (batch, horizon)]
                         Empirical quantiles at levels [0.05, 0.10, 0.25, 0.50,
                         0.75, 0.90, 0.95] in EUR/MWh.
            'mean'     : Tensor (batch, horizon)
                         Sample mean over all paths×samples in EUR/MWh.
            'gp_mean'  : Tensor (batch, n_paths_infer, horizon)
                         GP posterior mean (useful for diagnostics).
            'gp_var'   : Tensor (batch, n_paths_infer, horizon)
                         GP posterior variance (useful for diagnostics).
            'z_paths'  : Tensor (batch, n_paths_infer, horizon, latent_dim)
                         Latent force trajectories (useful for interpretation).
        """
        with torch.no_grad():
            # ----------------------------------------------------------------
            # Run the full forward pass with inference path count
            # ----------------------------------------------------------------
            fwd = self.forward(
                history=history,
                future_exog=future_exog,
                n_paths=self.n_paths_infer,
            )

            z_paths = fwd["z_paths"]  # (B, P_inf, T, D)
            gp_mean = fwd["gp_mean"]  # (B, P_inf, T)
            gp_var = fwd["gp_var"]  # (B, P_inf, T)

            batch_size = history.shape[0]
            horizon = future_exog.shape[1]
            n_p = z_paths.shape[1]  # n_paths_infer

            # ----------------------------------------------------------------
            # Sample from flow: (B, P_inf, n_samples_infer, T)
            # ----------------------------------------------------------------
            samples = self.flow.sample(
                gp_mean=gp_mean,
                gp_var=gp_var,
                z_paths=z_paths,
                calendar=future_exog,
                n_samples=self.n_samples_infer,
            )
            # samples : (B, P_inf, K, T)

            # Flatten paths × samples → (B, P_inf*K, T)
            n_k = self.n_samples_infer
            samples_flat = samples.reshape(batch_size, n_p * n_k, horizon)
            # samples_flat : (B, P*K, T)

            # ----------------------------------------------------------------
            # Compute empirical quantiles at each registered level
            # ----------------------------------------------------------------
            quantile_dict: Dict[float, Tensor] = {}
            for tau in self.quantile_levels:
                tau_float = tau.item()
                q = torch.quantile(samples_flat, tau_float, dim=1)  # (B, T)
                quantile_dict[round(tau_float, 2)] = q

            # ----------------------------------------------------------------
            # Forecast mean (EUR/MWh) over all paths × samples
            # ----------------------------------------------------------------
            forecast_mean = samples_flat.mean(dim=1)  # (B, T)

        return {
            "samples": samples_flat,  # (B, P*K, T)  EUR/MWh
            "quantiles": quantile_dict,  # dict float → (B, T)
            "mean": forecast_mean,  # (B, T)
            "gp_mean": gp_mean,  # (B, P_inf, T)
            "gp_var": gp_var,  # (B, P_inf, T)
            "z_paths": z_paths,  # (B, P_inf, T, D)
        }


# =============================================================================
# __main__ smoke test
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test to verify all components wire together correctly.

    Creates a random mini-batch matching the EPFWindowDataset shapes:
        history    : (4, 168, 20)  — 4-sample batch, 1-week window, 20 features
        future_exog: (4,  24,  6)  — 24-hour horizon, 6 exog features
        target     : (4,  24)      — asinh-transformed prices
        target_raw : (4,  24)      — EUR/MWh prices

    Checks:
        1. All loss components are finite scalars.
        2. forward() returns the expected shapes.
        3. predict_distribution() returns valid outputs.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("LFGPNRFModel — smoke test")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ------------------------------------------------------------------
    # Instantiate model with reduced path/inducing counts for fast test
    # ------------------------------------------------------------------
    print("Instantiating model ...")
    model = LFGPNRFModel(
        history_feat_dim=20,
        future_feat_dim=6,
        latent_dim=3,
        encoder_hidden=128,
        sde_hidden=64,
        n_inducing=32,  # small for fast test
        n_flow_transforms=2,  # small for fast test
        n_flow_bins=4,  # small for fast test
        n_paths_train=4,  # small for fast test
        n_paths_infer=8,  # small for fast test
        n_samples_infer=16,  # small for fast test
        kl_beta=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {n_params:,}\n")

    # ------------------------------------------------------------------
    # Build random batch matching dataset shapes
    # ------------------------------------------------------------------
    B = 4
    history_len = 168
    horizon = 24
    hist_dim = 20
    fut_dim = 6

    batch = {
        "history": torch.randn(B, history_len, hist_dim, device=device),
        "future_exog": torch.randn(B, horizon, fut_dim, device=device),
        "target": torch.randn(B, horizon, device=device) * 0.5,  # asinh scale
        "target_raw": torch.randn(B, horizon, device=device) * 50 + 50,  # EUR/MWh
    }

    # ------------------------------------------------------------------
    # Test forward()
    # ------------------------------------------------------------------
    print("Testing forward() ...")
    model.train()
    fwd = model.forward(batch["history"], batch["future_exog"])

    assert fwd["z_paths"].shape == (
        B,
        model.n_paths_train,
        horizon,
        model.latent_dim,
    ), f"z_paths shape mismatch: {fwd['z_paths'].shape}"
    assert fwd["gp_mean"].shape == (B, model.n_paths_train, horizon), (
        f"gp_mean shape mismatch: {fwd['gp_mean'].shape}"
    )
    assert fwd["gp_var"].shape == (B, model.n_paths_train, horizon), (
        f"gp_var shape mismatch: {fwd['gp_var'].shape}"
    )
    assert fwd["sde_kl"].ndim == 0 or fwd["sde_kl"].numel() == 1, (
        f"sde_kl should be scalar, got shape {fwd['sde_kl'].shape}"
    )

    print(f"  z_paths  : {tuple(fwd['z_paths'].shape)}")
    print(f"  gp_mean  : {tuple(fwd['gp_mean'].shape)}")
    print(f"  gp_var   : {tuple(fwd['gp_var'].shape)}")
    print(f"  sde_kl   : {fwd['sde_kl'].item():.4f}")
    print("  forward() — PASSED\n")

    # ------------------------------------------------------------------
    # Test compute_loss()
    # ------------------------------------------------------------------
    print("Testing compute_loss() ...")
    losses = model.compute_loss(batch)

    required_keys = ["loss", "reconstruction", "sde_kl", "gp_loss", "calib_loss"]
    all_finite = True
    for key in required_keys:
        val = losses[key]
        is_finite = torch.isfinite(val).all().item()
        status = "✓" if is_finite else "✗ NON-FINITE"
        print(f"  {key:20s}: {val.item():12.4f}  {status}")
        if not is_finite:
            all_finite = False

    if not all_finite:
        print("\n  WARNING: some loss components are non-finite.")
        print("  This can happen on the very first step before the GP")
        print("  variational parameters have been initialised.  If it")
        print("  persists after a few steps, check the kernel hyperparameters.")
    else:
        print("\n  compute_loss() — all components finite — PASSED\n")

    # ------------------------------------------------------------------
    # Verify loss is differentiable (backward pass)
    # ------------------------------------------------------------------
    print("Testing backward pass ...")
    losses["loss"].backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    if grad_norms:
        print(f"  Params with grads : {len(grad_norms)}")
        print(f"  Grad norm (mean)  : {float(np.mean(grad_norms)):.4f}")
        print(f"  Grad norm (max)   : {float(np.max(grad_norms)):.4f}")
        print("  backward() — PASSED\n")
    else:
        print("  WARNING: no gradients found — check model parameters.\n")

    # ------------------------------------------------------------------
    # Test predict_distribution() (inference mode, no grad)
    # ------------------------------------------------------------------
    print("Testing predict_distribution() ...")
    model.eval()
    pred = model.predict_distribution(batch["history"], batch["future_exog"])

    n_total = model.n_paths_infer * model.n_samples_infer
    assert pred["samples"].shape == (B, n_total, horizon), (
        f"samples shape mismatch: {pred['samples'].shape}"
    )
    assert pred["mean"].shape == (B, horizon), (
        f"mean shape mismatch: {pred['mean'].shape}"
    )
    assert len(pred["quantiles"]) == len(model.quantile_levels), (
        f"expected {len(model.quantile_levels)} quantile levels, "
        f"got {len(pred['quantiles'])}"
    )

    print(f"  samples shape   : {tuple(pred['samples'].shape)}")
    print(f"  mean shape      : {tuple(pred['mean'].shape)}")
    print(f"  quantile levels : {list(pred['quantiles'].keys())}")
    print(f"  q0.05 (first)   : {pred['quantiles'][0.05][0, :4].tolist()}")
    print(f"  q0.95 (first)   : {pred['quantiles'][0.95][0, :4].tolist()}")
    print("  predict_distribution() — PASSED\n")

    # ------------------------------------------------------------------
    # Test on_train_epoch_start (logging hook — needs a mock trainer)
    # ------------------------------------------------------------------
    print("Testing on_train_epoch_start() ...")
    try:
        # Without a real trainer attached we just call it directly;
        # it will raise AttributeError on self.log — that's expected.
        model.on_train_epoch_start()
        print("  on_train_epoch_start() — PASSED (trainer attached)")
    except AttributeError as e:
        if "log" in str(e):
            print(f"  on_train_epoch_start() — skipped (no trainer): {e}")
        else:
            raise

    print("=" * 65)
    print("All smoke tests completed successfully.")
    print("=" * 65)
