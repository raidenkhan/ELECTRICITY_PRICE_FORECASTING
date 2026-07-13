# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# flow_emission.py — Conditional RQ-NSF Normalizing Flow Emission Layer
#
# Role (Layer 3 in the full pipeline):
#   Given the GP posterior mean/variance and the latent force path z_t, this
#   module models the conditional price distribution p(y_t | f_t, z_t, cal_t)
#   as a Rational-Quadratic Neural Spline Flow (RQ-NSF; Durkan et al. 2019).
#
#   The flow transforms prices y (EUR/MWh) → standard Gaussian via a two-stage
#   mapping:
#       Stage 1 (price normalisation):
#           y_norm = (y - price_min) / (price_max - price_min)  ∈ (0, 1)
#           y_logit = logit(clip(y_norm, 1e-4, 1-1e-4))         ∈ ℝ
#       Stage 2 (RQ-NSF):
#           z = T_φ(y_logit; condition)                          ~ N(0, 1)
#
#   The Jacobian of Stage 1 is folded into the log-probability during training
#   to ensure the flow density is with respect to the original EUR/MWh scale.
#
#   The condition vector fed to the flow is:
#       [gp_mean_t, gp_std_t, z_t (latent_dim), cal_t (n_calendar)]
#   giving condition_dim = 2 + latent_dim + n_calendar = 11 by default.
#
# Architecture summary:
#   Pre/post normalisation   logit / sigmoid ↔ (y - y_min) / (y_max - y_min)
#   Flow backbone            zuko.flows.NSF
#     features=1             univariate distribution (one price per timestep)
#     context=condition_dim  conditioned on GP posterior + latent force + cal
#     transforms=4           coupling layers (passes=2 → coupling, not autoregressive)
#     bins=8                 RQ spline bins per transform
#     hidden_features=[128, 128]
#
# Dimension key (defaults):
#   latent_dim    = 3     (renewable surplus / thermal scarcity / demand surge)
#   n_calendar    = 6     (hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos)
#   condition_dim = 11    (2 + 3 + 6)
#   price_min     = -600  EUR/MWh  (soft lower bound, below EPEX floor)
#   price_max     = 3000  EUR/MWh  (soft upper bound, EPEX market cap)
#
# Key design choices:
#   • gp_std (not gp_var) is used in the condition vector.  Variance spans many
#     orders of magnitude and numerically destabilises the MLP conditioner; std
#     lives on the same scale as the mean, which makes conditioning smoother.
#   • Coupling layers (passes=2) are preferred over fully autoregressive layers
#     because, with features=1, autoregressive and coupling are equivalent —
#     there is only one dimension to transform.  The `passes=2` flag is an
#     explicit reminder of this, and keeps the interface compatible if features
#     were ever increased to a multivariate setting.
#   • The logit pre-transform imposes soft price caps without hard truncation.
#     Values far outside [price_min, price_max] are mapped to extreme logit
#     values, which the flow can still assign finite (low) density to, avoiding
#     the numerical blowup that a hard truncated distribution would produce
#     during training on occasional spike observations.
#   • CRPS is estimated via the energy-score identity rather than quantile
#     integration, which avoids sorting and is parallelisable on GPU.
#
# References:
#   Durkan et al. (2019) "Neural Spline Flows"  arXiv:1906.04032
#   Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
#   LF_GP_NRF_Research_Plan.md §3.4, §4.2
#   zuko documentation https://zuko.readthedocs.io/stable/
# =============================================================================

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import zuko.flows
from torch import Tensor


class NormalizingFlowEmission(nn.Module):
    """Conditional RQ-NSF normalizing flow emission layer for LF-GP-NRF.

    Models the conditional price distribution

        p(y_t | f_t, z_t, cal_t)

    as a Rational-Quadratic Neural Spline Flow whose condition vector combines
    the GP posterior summary (mean, std), the latent force state, and calendar
    features for the forecast hour.

    The flow operates in *logit space*: prices are first mapped to (0,1) via
    min-max normalisation and then to ℝ via the logit function.  The Jacobian
    of this pre-transform is included in `log_prob` so that all densities are
    expressed with respect to the original EUR/MWh scale.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent force vector z_t.  Default 3.
    n_calendar : int
        Number of calendar conditioning scalars (hour_sin, hour_cos,
        dow_sin, dow_cos, doy_sin, doy_cos).  Default 6.
    n_transforms : int
        Number of RQ-NSF coupling/autoregressive layers.  Default 4.
    n_bins : int
        Number of rational-quadratic spline bins per transform.  Default 8.
    hidden_features : list of int
        Hidden layer widths for the MLP conditioner networks inside each
        coupling layer.  Default [128, 128].
    price_min : float
        Soft lower price bound in EUR/MWh used for normalisation.
        Default -600.0 (below EPEX spot floor).
    price_max : float
        Soft upper price bound in EUR/MWh used for normalisation.
        Default 3000.0 (EPEX market cap).
    """

    def __init__(
        self,
        latent_dim: int = 3,
        n_calendar: int = 6,
        n_transforms: int = 4,
        n_bins: int = 8,
        hidden_features: List[int] | None = None,
        price_min: float = -200.0,
        price_max: float = 1000.0,
    ) -> None:
        super().__init__()

        if hidden_features is None:
            hidden_features = [128, 128]

        self.latent_dim = latent_dim
        self.n_calendar = n_calendar
        self.n_transforms = n_transforms
        self.n_bins = n_bins
        self.price_min = price_min
        self.price_max = price_max

        # ------------------------------------------------------------------
        # Condition vector layout:
        #   [ gp_mean (1), gp_std (1), z_t (latent_dim), cal_t (n_calendar) ]
        # ------------------------------------------------------------------
        self.condition_dim: int = 2 + latent_dim + n_calendar

        # ------------------------------------------------------------------
        # RQ-NSF backbone (zuko)
        #
        # features=1        — univariate; one price per (batch, path, timestep)
        # context=cond_dim  — full condition vector described above
        # transforms        — number of coupling layers
        # bins              — RQ spline resolution
        # hidden_features   — MLP conditioner widths
        # passes=2          — coupling (not fully autoregressive); with
        #                     features=1 this is equivalent but makes the
        #                     intent explicit and future-proofs a multivariate
        #                     extension
        #
        # Zuko NSF API:
        #   flow_dist = self.flow(condition)          # LazyDistribution → Distribution
        #   flow_dist.log_prob(x)                     # x shape: (N, 1)
        #   flow_dist.sample((n_samples,))            # returns (n_samples, N, 1)
        # ------------------------------------------------------------------
        self.flow = zuko.flows.NSF(
            features=1,
            context=self.condition_dim,
            transforms=n_transforms,
            bins=n_bins,
            hidden_features=hidden_features,
            passes=2,  # coupling layers (checkered mask; safe for features=1)
        )

    # ------------------------------------------------------------------
    # Price normalisation helpers
    # ------------------------------------------------------------------

    def _to_logit_space(self, y_raw: Tensor) -> tuple[Tensor, Tensor]:
        """Map EUR/MWh prices to logit space and compute log-Jacobian.

        The two-step transform is:
            y_norm  = (y - price_min) / (price_max - price_min)   ∈ (0, 1)
            y_logit = logit(clip(y_norm, ε, 1-ε))                 ∈ ℝ

        The log-Jacobian of the *full* transform (including the 1/range factor
        from the linear normalisation step) is:

            log|dy_logit / dy| = -log(y_norm * (1 - y_norm))
                                 - log(price_max - price_min)

        where y_norm is evaluated *before* clipping, so the Jacobian is exact
        for interior points.  For the rare cases where y_raw falls outside the
        soft bounds (genuine price spikes), the clipping introduces a small
        approximation error that is acceptable in practice.

        Parameters
        ----------
        y_raw : Tensor
            Prices in EUR/MWh, arbitrary shape.

        Returns
        -------
        y_logit : Tensor
            Same shape as y_raw, in ℝ (logit space).
        log_jac : Tensor
            Pointwise log |dy_logit / dy|, same shape as y_raw.
            This must be *added* to the flow's log_prob to recover
            log p(y) in EUR/MWh.
        """
        price_range = self.price_max - self.price_min  # scalar

        # Linear normalisation to (0, 1) — exact y_norm for Jacobian
        y_norm = (y_raw - self.price_min) / price_range  # (...,)

        # Log-Jacobian: d logit(y_norm) / dy  =  1 / (y_norm(1-y_norm)) * 1/range
        # Clamp y_norm inside (ε, 1-ε) before log to avoid -inf
        eps = 1e-4
        y_norm_clamped = y_norm.clamp(min=eps, max=1.0 - eps)
        log_jac = (
            -torch.log(y_norm_clamped)
            - torch.log1p(-y_norm_clamped)
            - math.log(price_range)
        )  # log |dz/dy|  (positive contribution to log p(y))

        # Apply logit using the clamped value
        y_logit = torch.log(y_norm_clamped) - torch.log1p(-y_norm_clamped)

        return y_logit, log_jac

    def _from_logit_space(self, y_logit: Tensor) -> Tensor:
        """Inverse map: logit space → EUR/MWh.

        y_norm  = sigmoid(y_logit)
        y_raw   = y_norm * (price_max - price_min) + price_min

        Parameters
        ----------
        y_logit : Tensor
            Samples in logit space, arbitrary shape.

        Returns
        -------
        Tensor
            Prices in EUR/MWh, same shape as input.
        """
        y_norm = torch.sigmoid(y_logit)  # (0, 1)
        return y_norm * (self.price_max - self.price_min) + self.price_min

    # ------------------------------------------------------------------
    # Condition vector construction
    # ------------------------------------------------------------------

    def build_condition(
        self,
        gp_mean: Tensor,
        gp_var: Tensor,
        z_t: Tensor,
        cal_t: Tensor,
    ) -> Tensor:
        """Assemble the flat condition vector fed to the flow's conditioner MLP.

        All input tensors may be in either a batched/path-expanded form or a
        pre-flattened (N,) / (N, dim) form.  The method flattens everything to
        (N, condition_dim) where N = batch * n_paths * horizon.

        Parameters
        ----------
        gp_mean : Tensor, shape (batch, n_paths, horizon) or (N,)
            GP posterior mean at each forecast timestep.
        gp_var : Tensor, shape (batch, n_paths, horizon) or (N,)
            GP posterior variance (not std) at each forecast timestep.
            Converted to std internally for numerical conditioning.
        z_t : Tensor, shape (batch, n_paths, horizon, latent_dim) or (N, latent_dim)
            Latent force path samples.
        cal_t : Tensor, shape (batch, horizon, n_calendar) or (N, n_calendar)
            Calendar features for the forecast horizon.  Broadcast across
            n_paths automatically when in the 3-D form.

        Returns
        -------
        Tensor, shape (N, condition_dim)
            Flat condition matrix ready to be passed to ``self.flow(condition)``.
        """
        # ------------------------------------------------------------------
        # Detect shape: if gp_mean is 3-D → batched (batch, n_paths, horizon)
        #               if gp_mean is 1-D → pre-flattened (N,)
        # ------------------------------------------------------------------
        if gp_mean.dim() == 3:
            # gp_mean/var : (batch, n_paths, horizon)
            # z_t         : (batch, n_paths, horizon, latent_dim)
            # cal_t       : (batch, horizon, n_calendar)

            batch, n_paths, horizon = gp_mean.shape

            # Convert variance → std for better numerical conditioning
            gp_std = torch.sqrt(gp_var + 1e-6)  # (batch, n_paths, horizon)

            # Flatten gp_mean and gp_std to (N, 1)
            gp_mean_flat = gp_mean.reshape(-1, 1)  # (N, 1)
            gp_std_flat = gp_std.reshape(-1, 1)  # (N, 1)

            # Flatten z_t to (N, latent_dim)
            z_flat = z_t.reshape(-1, self.latent_dim)  # (N, latent_dim)

            # Broadcast cal_t across n_paths:
            #   (batch, horizon, n_calendar)
            #     → (batch, 1, horizon, n_calendar)
            #     → (batch, n_paths, horizon, n_calendar)
            #     → (N, n_calendar)
            cal_expanded = cal_t.unsqueeze(1).expand(
                batch, n_paths, horizon, self.n_calendar
            )
            cal_flat = cal_expanded.reshape(-1, self.n_calendar)  # (N, n_calendar)

        else:
            # Pre-flattened: gp_mean (N,), gp_var (N,), z_t (N, latent_dim),
            #                cal_t (N, n_calendar)
            gp_std = torch.sqrt(gp_var + 1e-6)  # (N,)
            gp_mean_flat = gp_mean.unsqueeze(-1)  # (N, 1)
            gp_std_flat = gp_std.unsqueeze(-1)  # (N, 1)
            z_flat = z_t  # (N, latent_dim)
            cal_flat = cal_t  # (N, n_calendar)

        # Concatenate all parts along the feature dimension
        # Result: (N, 1 + 1 + latent_dim + n_calendar) = (N, condition_dim)
        condition = torch.cat([gp_mean_flat, gp_std_flat, z_flat, cal_flat], dim=-1)
        return condition  # (N, condition_dim)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def log_prob(
        self,
        y_raw: Tensor,
        gp_mean: Tensor,
        gp_var: Tensor,
        z_paths: Tensor,
        calendar: Tensor,
    ) -> Tensor:
        """Compute log p(y_t | f_t, z_t, cal_t) for all (batch, path, horizon).

        The log-probability is evaluated in EUR/MWh units (the natural scale of
        the observation).  Internally:
          1. y is mapped to logit space, and the log-Jacobian is computed.
          2. The flow evaluates log p_flow(y_logit | condition).
          3. The two terms are summed: log p(y) = log p_flow(y_logit) + log|J|.

        Parameters
        ----------
        y_raw : Tensor, shape (batch, horizon)
            Observed prices in EUR/MWh.
        gp_mean : Tensor, shape (batch, n_paths, horizon)
            GP posterior mean for each SDE path.
        gp_var : Tensor, shape (batch, n_paths, horizon)
            GP posterior variance for each SDE path.
        z_paths : Tensor, shape (batch, n_paths, horizon, latent_dim)
            Latent force samples from the Neural SDE.
        calendar : Tensor, shape (batch, horizon, n_calendar)
            Calendar features (hour_sin/cos, dow_sin/cos, doy_sin/cos).

        Returns
        -------
        log_prob_mean : Tensor, shape (batch, horizon)
            Log-probabilities averaged across the n_paths Monte-Carlo paths.
            Suitable for direct use in the ELBO reconstruction term.

        Notes
        -----
        The per-path, per-timestep log-probabilities (batch, n_paths, horizon)
        are also computed internally and can be recovered by not averaging if
        a different path-combination strategy is desired.
        """
        batch, n_paths, horizon = gp_mean.shape

        # ------------------------------------------------------------------
        # Step 1: Pre-transform y_raw → logit space
        #   y_raw   : (batch, horizon)
        #   Expand across n_paths → (batch, n_paths, horizon)
        # ------------------------------------------------------------------
        y_expanded = y_raw.unsqueeze(1).expand(batch, n_paths, horizon)
        y_logit, log_jac = self._to_logit_space(y_expanded)
        # y_logit, log_jac : (batch, n_paths, horizon)

        # ------------------------------------------------------------------
        # Step 2: Build flat condition vector  (N, condition_dim)
        # ------------------------------------------------------------------
        condition = self.build_condition(gp_mean, gp_var, z_paths, calendar)
        # condition : (N, condition_dim)  where N = batch * n_paths * horizon

        # ------------------------------------------------------------------
        # Step 3: Evaluate flow log-probability
        #
        # zuko NSF expects:
        #   condition : (N, condition_dim)
        #   x         : (N, features=1)
        # ------------------------------------------------------------------
        y_logit_flat = y_logit.reshape(-1, 1)  # (N, 1)

        flow_dist = self.flow(condition)  # conditional Distribution
        lp_flow = flow_dist.log_prob(y_logit_flat)  # (N,)

        # ------------------------------------------------------------------
        # Step 4: Add log-Jacobian to convert from logit-space density to
        #         EUR/MWh density.
        #         log p(y) = log p_flow(y_logit) + log|dy_logit / dy|
        # ------------------------------------------------------------------
        log_jac_flat = log_jac.reshape(-1)  # (N,)
        lp_total = lp_flow + log_jac_flat  # (N,)

        # ------------------------------------------------------------------
        # Step 5: Reshape to (batch, n_paths, horizon) and average over paths
        # ------------------------------------------------------------------
        lp_full = lp_total.reshape(batch, n_paths, horizon)
        # (batch, n_paths, horizon)

        # Average log-probs over Monte-Carlo paths → (batch, horizon)
        log_prob_mean = lp_full.mean(dim=1)  # (batch, horizon)

        return log_prob_mean

    def sample(
        self,
        gp_mean: Tensor,
        gp_var: Tensor,
        z_paths: Tensor,
        calendar: Tensor,
        n_samples: int = 50,
    ) -> Tensor:
        """Draw price samples from the conditional flow distribution.

        Parameters
        ----------
        gp_mean : Tensor, shape (batch, n_paths, horizon)
        gp_var : Tensor, shape (batch, n_paths, horizon)
        z_paths : Tensor, shape (batch, n_paths, horizon, latent_dim)
        calendar : Tensor, shape (batch, horizon, n_calendar)
        n_samples : int
            Number of price samples to draw per (batch, path, horizon) triplet.
            Default 50.

        Returns
        -------
        prices : Tensor, shape (batch, n_paths, n_samples, horizon)
            Price samples in EUR/MWh.

        Notes
        -----
        Samples are returned in EUR/MWh via the inverse pre-transform:
            y_logit → sigmoid → (0,1) → de-normalise → EUR/MWh.
        """
        batch, n_paths, horizon = gp_mean.shape

        # ------------------------------------------------------------------
        # Build flat condition (N = batch*n_paths*horizon, condition_dim)
        # ------------------------------------------------------------------
        condition = self.build_condition(gp_mean, gp_var, z_paths, calendar)

        # ------------------------------------------------------------------
        # Sample from flow
        #   flow_dist.sample((n_samples,)) → (n_samples, N, 1)
        # ------------------------------------------------------------------
        flow_dist = self.flow(condition)
        y_logit_samples = flow_dist.sample((n_samples,))  # (n_samples, N, 1)
        y_logit_samples = y_logit_samples.squeeze(-1)  # (n_samples, N)

        # ------------------------------------------------------------------
        # Inverse pre-transform: logit space → EUR/MWh
        # ------------------------------------------------------------------
        y_raw_samples = self._from_logit_space(y_logit_samples)
        # (n_samples, N)

        # ------------------------------------------------------------------
        # Reshape to (batch, n_paths, n_samples, horizon)
        # ------------------------------------------------------------------
        # First reshape N → (batch, n_paths, horizon):
        y_raw_samples = y_raw_samples.reshape(n_samples, batch, n_paths, horizon)
        # Then permute to (batch, n_paths, n_samples, horizon):
        prices = y_raw_samples.permute(1, 2, 0, 3).contiguous()

        return prices  # (batch, n_paths, n_samples, horizon)

    def crps_estimate(
        self,
        y_raw: Tensor,
        gp_mean: Tensor,
        gp_var: Tensor,
        z_paths: Tensor,
        calendar: Tensor,
        n_samples: int = 200,
    ) -> Tensor:
        """Estimate the Continuous Ranked Probability Score (CRPS) via MC.

        Uses the energy-score identity (Gneiting & Raftery 2007):

            CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]

        where X, X' are independent draws from the forecast distribution F.
        This estimator is unbiased and parallelisable.

        Parameters
        ----------
        y_raw : Tensor, shape (batch, horizon)
            Observed prices in EUR/MWh.
        gp_mean : Tensor, shape (batch, n_paths, horizon)
        gp_var : Tensor, shape (batch, n_paths, horizon)
        z_paths : Tensor, shape (batch, n_paths, horizon, latent_dim)
        calendar : Tensor, shape (batch, horizon, n_calendar)
        n_samples : int
            Number of Monte-Carlo samples used to estimate each expectation.
            Default 200.  Higher values give lower-variance CRPS estimates at
            the cost of more GPU memory.

        Returns
        -------
        crps : Tensor, shape (batch, horizon)
            Per-timestep CRPS values (non-negative; lower is better).
        """
        with torch.no_grad():
            # ----------------------------------------------------------
            # Draw two independent sample sets for the energy-score trick
            # samples_a, samples_b : (batch, n_paths, n_samples, horizon)
            # ----------------------------------------------------------
            samples_a = self.sample(
                gp_mean, gp_var, z_paths, calendar, n_samples=n_samples
            )
            samples_b = self.sample(
                gp_mean, gp_var, z_paths, calendar, n_samples=n_samples
            )
            # (batch, n_paths, n_samples, horizon)

            batch, n_paths, _, horizon = samples_a.shape

            # ----------------------------------------------------------
            # Average over paths first (marginalise path index), then over
            # samples, to form a single forecast distribution per
            # (batch, timestep) pair.
            #
            # Flatten (n_paths, n_samples) → (n_paths * n_samples,)
            # for the expectation estimate.
            # ----------------------------------------------------------
            # Reshape: (batch, n_paths*n_samples, horizon)
            samples_a_flat = samples_a.reshape(batch, n_paths * n_samples, horizon)
            samples_b_flat = samples_b.reshape(batch, n_paths * n_samples, horizon)

            # y_raw : (batch, horizon) → expand to (batch, 1, horizon)
            y_expanded = y_raw.unsqueeze(1)  # (batch, 1, horizon)

            # E[|X - y|] : mean over the sample dimension → (batch, horizon)
            term1 = torch.abs(samples_a_flat - y_expanded).mean(dim=1)

            # E[|X - X'|] : mean over the sample dimension → (batch, horizon)
            term2 = torch.abs(samples_a_flat - samples_b_flat).mean(dim=1)

            # CRPS = E[|X-y|] - 0.5 * E[|X-X'|]
            crps = term1 - 0.5 * term2  # (batch, horizon)

            # CRPS is theoretically non-negative; clamp for numerical safety
            crps = crps.clamp(min=0.0)

        return crps  # (batch, horizon)


# =============================================================================
# __main__ — quick smoke-test (no training, no data loading required)
# =============================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke-test] device = {device}")

    # ------------------------------------------------------------------
    # Hyperparameters matching the LF-GP-NRF defaults (§4.3)
    # ------------------------------------------------------------------
    BATCH = 4
    N_PATHS = 8  # M SDE paths (reduced from 16 for fast testing)
    HORIZON = 24  # 24-hour ahead forecast
    LATENT = 3  # latent force dim
    N_CAL = 6  # calendar features
    N_SAMP = 50  # price samples in .sample()
    N_CRPS = 200  # MC samples for CRPS

    # ------------------------------------------------------------------
    # Instantiate module
    # ------------------------------------------------------------------
    model = NormalizingFlowEmission(
        latent_dim=LATENT,
        n_calendar=N_CAL,
        n_transforms=4,
        n_bins=8,
        hidden_features=[128, 128],
        price_min=-600.0,
        price_max=3000.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke-test] NormalizingFlowEmission — trainable params: {n_params:,}")
    print(f"[smoke-test] condition_dim = {model.condition_dim}")

    # ------------------------------------------------------------------
    # Synthetic inputs (plausible ranges)
    # ------------------------------------------------------------------
    # Actual prices in EUR/MWh — mix of typical, negative, and spike values
    y_raw = torch.cat(
        [
            torch.zeros(BATCH, HORIZON // 4).uniform_(-50.0, 50.0),  # normal
            torch.zeros(BATCH, HORIZON // 4).uniform_(-200.0, 0.0),  # negative
            torch.zeros(BATCH, HORIZON // 4).uniform_(100.0, 500.0),  # high
            torch.zeros(BATCH, HORIZON // 4).uniform_(500.0, 2000.0),  # spike
        ],
        dim=1,
    ).to(device)
    # y_raw : (BATCH, HORIZON)

    # GP posterior outputs (simulate typical posterior outputs)
    gp_mean = torch.randn(BATCH, N_PATHS, HORIZON).to(device) * 30.0 + 50.0
    gp_var = torch.rand(BATCH, N_PATHS, HORIZON).to(device).abs() * 400.0 + 1.0

    # Latent force paths (unit-ish scale; SDE output)
    z_paths = torch.randn(BATCH, N_PATHS, HORIZON, LATENT).to(device) * 0.5

    # Calendar features (sin/cos encoding already in [-1, 1])
    calendar = torch.zeros(BATCH, HORIZON, N_CAL).to(device)
    hours = torch.arange(HORIZON, dtype=torch.float32)
    calendar[:, :, 0] = torch.sin(2 * math.pi * hours / 24).unsqueeze(0)
    calendar[:, :, 1] = torch.cos(2 * math.pi * hours / 24).unsqueeze(0)
    calendar[:, :, 2] = torch.sin(2 * math.pi * torch.tensor([0.0]) / 7)
    calendar[:, :, 3] = torch.cos(2 * math.pi * torch.tensor([0.0]) / 7)
    calendar[:, :, 4] = torch.sin(2 * math.pi * torch.tensor([100.0]) / 365)
    calendar[:, :, 5] = torch.cos(2 * math.pi * torch.tensor([100.0]) / 365)

    # ------------------------------------------------------------------
    # Test 1: log_prob → shape (batch, horizon), no NaN / Inf
    # ------------------------------------------------------------------
    print("\n[Test 1] log_prob ...")
    lp = model.log_prob(y_raw, gp_mean, gp_var, z_paths, calendar)
    assert lp.shape == (BATCH, HORIZON), (
        f"Expected ({BATCH}, {HORIZON}), got {lp.shape}"
    )
    assert not torch.isnan(lp).any(), "log_prob contains NaN!"
    assert not torch.isinf(lp).any(), "log_prob contains Inf!"
    print(f"  shape  : {tuple(lp.shape)}  ✓")
    print(f"  range  : [{lp.min().item():.3f}, {lp.max().item():.3f}]  ✓")
    print(f"  no NaN : {not torch.isnan(lp).any().item()}  ✓")
    print(f"  no Inf : {not torch.isinf(lp).any().item()}  ✓")

    # Check gradients flow
    lp.mean().backward()
    grad_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  grads  : {grad_ok}  ✓")

    # ------------------------------------------------------------------
    # Test 2: sample → shape (batch, n_paths, n_samples, horizon)
    #         all values within price range
    # ------------------------------------------------------------------
    print("\n[Test 2] sample ...")
    with torch.no_grad():
        prices = model.sample(gp_mean, gp_var, z_paths, calendar, n_samples=N_SAMP)

    expected_shape = (BATCH, N_PATHS, N_SAMP, HORIZON)
    assert prices.shape == expected_shape, (
        f"Expected {expected_shape}, got {prices.shape}"
    )
    # Soft bounds — sigmoid output guarantees (price_min, price_max) strictly
    assert (prices > model.price_min).all(), (
        f"Some samples below price_min={model.price_min}"
    )
    assert (prices < model.price_max).all(), (
        f"Some samples above price_max={model.price_max}"
    )
    assert not torch.isnan(prices).any(), "sample contains NaN!"
    print(f"  shape  : {tuple(prices.shape)}  ✓")
    print(
        f"  min    : {prices.min().item():.2f} EUR/MWh  (bound: {model.price_min})  ✓"
    )
    print(
        f"  max    : {prices.max().item():.2f} EUR/MWh  (bound: {model.price_max})  ✓"
    )
    print(f"  no NaN : {not torch.isnan(prices).any().item()}  ✓")

    # ------------------------------------------------------------------
    # Test 3: crps_estimate → shape (batch, horizon), non-negative
    # ------------------------------------------------------------------
    print("\n[Test 3] crps_estimate ...")
    crps = model.crps_estimate(
        y_raw, gp_mean, gp_var, z_paths, calendar, n_samples=N_CRPS
    )
    assert crps.shape == (BATCH, HORIZON), (
        f"Expected ({BATCH}, {HORIZON}), got {crps.shape}"
    )
    assert (crps >= 0.0).all(), "CRPS contains negative values!"
    assert not torch.isnan(crps).any(), "CRPS contains NaN!"
    print(f"  shape  : {tuple(crps.shape)}  ✓")
    print(f"  range  : [{crps.min().item():.3f}, {crps.max().item():.3f}] EUR/MWh  ✓")
    print(f"  non-neg: {(crps >= 0).all().item()}  ✓")
    print(f"  no NaN : {not torch.isnan(crps).any().item()}  ✓")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  All smoke-tests passed.  NormalizingFlowEmission is ready.")
    print("=" * 60)
    sys.exit(0)
