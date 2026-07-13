# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# force_gp.py — Sparse Variational GP with Force-Conditioned Warped Kernel
#
# Role (Layer 3b in the full pipeline):
#   Wraps WarpedForceKernel inside an ApproximateGP (SVGP) framework so that
#   the full ~79,000-point German electricity dataset can be handled without
#   O(n³) exact GP inference.
#
#   The GP receives the joint (calendar_features, z_t) input tensor — one row
#   per (sample, horizon-step) pair — and returns a posterior predictive
#   distribution over GP function values f_t.  These f_t values are then fed
#   to the NormalizingFlowEmission as the "smooth skeleton" of the price.
#
# Inducing point strategy:
#   Inducing locations Z_m ∈ ℝ^{M × input_dim} are stored as a free
#   nn.Parameter and jointly optimised with all other model parameters via
#   the variational ELBO.  M=256 is chosen for the RTX 4000 (8.6 GB VRAM)
#   constraint while still providing good posterior approximation quality.
#
# Variational family:
#   CholeskyVariationalDistribution — stores the lower-triangular Cholesky
#   factor of the M×M variational covariance S directly; this avoids the
#   positive-definite constraint of directly parameterising S and provides
#   numerically stable KL computation.
#
#   UnwhitenedVariationalStrategy — works directly in the prior covariance
#   space (not whitened/natural parameterisation).  This is preferred here
#   because the WarpedForceKernel's non-stationarity means that the whitened
#   parameterisation can create poorly-conditioned natural gradients.
#
# Key functions exposed to the outer LFGPNRFModel:
#   ForceConditionedGP.forward(x)             — GP prior/posterior distribution
#   ForceConditionedGP.compute_elbo(x, y, …)  — SVGP ELBO term for joint training
#   ForceConditionedGP.predict_on_paths(…)    — batched prediction over SDE paths
#   prepare_inputs(time_features, z_paths, …) — input tensor construction helper
#
# Dimension key:
#   batch      — number of training/inference windows in the mini-batch
#   n_paths    — number of SDE sample paths (M=16 train, 256 inference)
#   horizon    — forecast horizon steps (24 for day-ahead)
#   n_calendar — number of calendar/time features
#   latent_dim — dimensionality of z_t  (default 3)
#   input_dim  — n_calendar + latent_dim  (total GP input width)
#   n_inducing — number of inducing points (default 256)
#
# References:
#   Hensman et al. (2015) "Scalable Variational Gaussian Process Classification"
#   Titsias (2009) "Variational Learning of Inducing Variables in Sparse GPs"
#   Wilson et al. (AISTATS 2016) "Deep Kernel Learning"
#   LF_GP_NRF_Research_Plan.md §3.3, §4.2–4.3
# =============================================================================

from __future__ import annotations

from typing import Tuple

import gpytorch
import torch
import torch.nn as nn
from torch import Tensor

from .force_kernel import WarpedForceKernel

# ---------------------------------------------------------------------------
# Standalone input-preparation helper (no self dependency → easily testable)
# ---------------------------------------------------------------------------


def prepare_inputs(
    time_features: Tensor,
    z_paths: Tensor,
    batch: int,
    n_paths: int,
    horizon: int,
    n_calendar: int,
) -> Tensor:
    """Construct the GP input tensor by concatenating calendar features and z paths.

    For each SDE path m, the calendar features are the same (they are
    deterministic functions of the forecast horizon timestamps), while the
    latent force trajectory z^(m) is path-specific.  This function tiles the
    calendar block accordingly and concatenates it with the z block.

    Parameters
    ----------
    time_features : Tensor, shape (batch, horizon, n_calendar)
        Calendar/time features for each forecast horizon step.  Includes the
        raw time index as dimension 0 (required by WarpedForceKernel).
    z_paths : Tensor, shape (batch, n_paths, horizon, latent_dim)
        Latent force trajectories sampled from the Neural SDE.
    batch : int
        Mini-batch size.  Must equal ``time_features.shape[0]``.
    n_paths : int
        Number of SDE sample paths.  Must equal ``z_paths.shape[1]``.
    horizon : int
        Forecast horizon length.  Must equal ``time_features.shape[1]``.
    n_calendar : int
        Number of calendar feature dimensions.  Must equal
        ``time_features.shape[2]``.

    Returns
    -------
    Tensor, shape (batch * n_paths, horizon, input_dim)
        Stacked GP input tensor where ``input_dim = n_calendar + latent_dim``.
        The batch and path dimensions are folded together so the GP sees a
        single (batch*n_paths * horizon, input_dim) sequence when further
        reshaped for evaluation.

    Notes
    -----
    The returned tensor keeps the horizon dimension intact so that the caller
    can optionally process each horizon step independently.  The GP evaluation
    in ``predict_on_paths`` flattens it further to (batch*n_paths*horizon,
    input_dim).
    """
    # time_features: (batch, horizon, n_calendar)
    # Expand to (batch, n_paths, horizon, n_calendar) by repeating along path dim
    # then reshape to (batch*n_paths, horizon, n_calendar)
    cal_expanded = (
        time_features.unsqueeze(1)  # (batch, 1, horizon, n_calendar)
        .expand(
            batch, n_paths, horizon, n_calendar
        )  # (batch, n_paths, horizon, n_calendar)
        .reshape(batch * n_paths, horizon, n_calendar)  # (B*M, T, n_cal)
    )

    # z_paths: (batch, n_paths, horizon, latent_dim)
    # Reshape to (batch*n_paths, horizon, latent_dim)
    latent_dim = z_paths.shape[-1]
    z_flat = z_paths.reshape(batch * n_paths, horizon, latent_dim)  # (B*M, T, d)

    # Concatenate along feature dim → (B*M, T, n_calendar + latent_dim)
    gp_inputs = torch.cat([cal_expanded, z_flat], dim=-1)  # (B*M, T, input_dim)

    return gp_inputs


# ---------------------------------------------------------------------------
# ForceConditionedGP — ApproximateGP with SVGP variational strategy
# ---------------------------------------------------------------------------


class ForceConditionedGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP conditioned on the latent force trajectory z_t.

    This is the core probabilistic layer that maps joint (calendar, z_t) inputs
    to a distribution over electricity price function values.  It uses the
    WarpedForceKernel so that correlation structure adapts to market regimes
    encoded in z_t.

    Parameters
    ----------
    input_dim : int
        Total GP input dimensionality = n_calendar + latent_dim.
    latent_dim : int
        Latent force dimensionality.  Default 3.
    n_inducing : int
        Number of inducing points.  Default 256 (reduced from the plan's 512
        to fit within RTX 4000 8.6 GB VRAM constraints).  Increase to 512 when
        training on an A100.
    n_calendar : int
        Number of calendar feature dimensions.  Must equal
        input_dim - latent_dim.

    Attributes
    ----------
    inducing_points : nn.Parameter, shape (n_inducing, input_dim)
        Learned inducing point locations in the joint (cal, z) input space.
        Initialised from N(0, 1); in practice, k-means++ initialisation on the
        first training batch is recommended before the first optimiser step.
    mean_module : gpytorch.means.ConstantMean
        Constant mean function; the single scalar offset is learnable.
    covar_module : WarpedForceKernel
        Non-stationary warped kernel (see force_kernel.py).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        n_inducing: int = 256,
        n_calendar: int | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Resolve n_calendar
        # ------------------------------------------------------------------
        if n_calendar is None:
            n_calendar = input_dim - latent_dim

        if n_calendar != input_dim - latent_dim:
            raise ValueError(
                f"n_calendar ({n_calendar}) must equal "
                f"input_dim - latent_dim ({input_dim} - {latent_dim} = "
                f"{input_dim - latent_dim})."
            )

        # ------------------------------------------------------------------
        # GPyTorch SVGP correct initialisation pattern:
        #
        # The variational strategy requires a reference to the model, but
        # `self` does not exist until after super().__init__ completes.
        # The canonical solution used throughout GPyTorch's own examples is:
        #
        #   1. Use a plain torch.Tensor (NOT nn.Parameter) for inducing_points
        #      when constructing the variational strategy before super().__init__.
        #   2. Call super().__init__(variational_strategy) — GPyTorch's
        #      ApproximateGP.__init__ registers the strategy and sets
        #      self.variational_strategy, which in turn stores the model ref.
        #   3. After super().__init__, the inducing points inside the strategy
        #      are already a registered Parameter (GPyTorch does this
        #      internally); no further nn.Parameter wrapping is needed.
        #
        # Passing model=None and then back-patching creates a circular
        # reference in nn.Module.named_children() that causes Python to hit
        # its recursion limit when PyTorch tries to move the model to GPU.
        # ------------------------------------------------------------------

        # Step 1 — plain tensor; GPyTorch will register it as a Parameter
        inducing_points = torch.randn(n_inducing, input_dim)

        # Step 2 — variational distribution (no model reference needed)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=n_inducing,
        )

        # Step 3 — variational strategy with learn_inducing_locations=True
        # GPyTorch wraps inducing_points as an nn.Parameter internally when
        # learn_inducing_locations=True, so we do NOT wrap it ourselves.
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True,
        )

        # ApproximateGP.__init__ registers the strategy as a sub-module and
        # stores it as self.variational_strategy.  Because we passed `self`
        # above, the model reference is correct from the start — no circular
        # reference is created.
        super().__init__(variational_strategy)

        # Store config for use in predict_on_paths / prepare_inputs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_calendar = n_calendar
        self.n_inducing = n_inducing

        # ------------------------------------------------------------------
        # Mean function — learnable constant offset
        # ------------------------------------------------------------------
        self.mean_module = gpytorch.means.ConstantMean()

        # ------------------------------------------------------------------
        # Covariance function — non-stationary warped kernel
        # ------------------------------------------------------------------
        self.covar_module = WarpedForceKernel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_calendar=n_calendar,
        )

        # ------------------------------------------------------------------
        # Patch _cholesky_factor on the variational_strategy *instance*.
        #
        # GPyTorch's VariationalStrategy._cholesky_factor is memoised and
        # called on `self.variational_strategy`, not on `self` (the model).
        # Overriding it on the GP class therefore has no effect.  The only
        # reliable approach is to replace the bound method on the strategy
        # instance directly, after super().__init__ has created it.
        #
        # The replacement adds a 1e-2 diagonal nugget and raises the jitter
        # ceiling to 1e-1, which is sufficient for the random-initialisation
        # near-singular regime of WarpedForceKernel.
        # ------------------------------------------------------------------
        import types

        from linear_operator import to_dense as _to_dense
        from linear_operator.utils.cholesky import psd_safe_cholesky as _psd_chol

        def _robust_cholesky_factor(strategy_self, induc_induc_covar):
            mat = _to_dense(induc_induc_covar).float()
            eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
            mat = mat + 1e-2 * eye
            return _psd_chol(mat, jitter=1e-3, max_tries=8)

        self.variational_strategy._cholesky_factor = types.MethodType(
            _robust_cholesky_factor, self.variational_strategy
        )

    # ------------------------------------------------------------------
    # GP prior / variational posterior
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the GP *prior* distribution at input locations x.

        GPyTorch's variational strategy intercepts calls to self(x) and
        converts this prior into the variational posterior q(f(x)) via the
        inducing point equations.  This method defines only the prior; the
        posterior is obtained by calling ``self(x)`` (not ``self.forward(x)``).

        Parameters
        ----------
        x : Tensor, shape (n, input_dim)
            GP input points.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Prior distribution p(f(x)) = N(m(x), K(x, x)).
        """
        mean = self.mean_module(x)  # (n,)
        # Use a larger jitter tolerance to handle the non-stationary warped
        # kernel which can produce near-singular K(Z_m, Z_m) at initialisation.
        with gpytorch.settings.cholesky_jitter(1e-3):
            covar = self.covar_module(x, x)  # LazyEvaluatedKernelTensor (n, n)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # ------------------------------------------------------------------
    # ELBO computation
    # ------------------------------------------------------------------

    def compute_elbo(
        self,
        x: Tensor,
        y: Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        mll: gpytorch.mlls.VariationalELBO,
    ) -> Tensor:
        """Compute the negative SVGP ELBO for a mini-batch.

        The ELBO is::

            L = Σ_i E_{q(f_i)}[log p(y_i | f_i)] - KL[q(u) || p(u)]

        and this function returns ``-L`` (the loss to minimise).

        Parameters
        ----------
        x : Tensor, shape (batch * horizon, input_dim)
            Flattened GP inputs for the current mini-batch.  Construct via
            ``prepare_inputs(...).reshape(-1, input_dim)``.
        y : Tensor, shape (batch * horizon,)
            Corresponding observed log-prices (or raw prices depending on the
            training pipeline's normalisation choice).
        likelihood : gpytorch.likelihoods.Likelihood
            The observation likelihood (typically GaussianLikelihood or a
            custom heteroscedastic variant coupled to the flow).
        mll : gpytorch.mlls.VariationalELBO
            Pre-constructed ELBO objective.  Should be initialised once in the
            outer training loop as::

                mll = gpytorch.mlls.VariationalELBO(
                    likelihood, gp_model, num_data=n_train
                )

        Returns
        -------
        Tensor
            Scalar loss = -ELBO.  Differentiable w.r.t. all model parameters.
        """
        # Obtain variational posterior q(f(x)) via the variational strategy.
        # Use elevated jitter tolerance during ELBO computation to prevent
        # NotPSDError when the warped kernel matrix K(Z_m, Z_m) is near-singular
        # early in training (random inducing point initialisation).
        with gpytorch.settings.cholesky_jitter(1e-3):
            output = self(x)  # MultivariateNormal, shape-checked internally
            loss = -mll(output, y)  # scalar: negative ELBO
        return loss

    # ------------------------------------------------------------------
    # Batched prediction over SDE paths
    # ------------------------------------------------------------------

    def predict_on_paths(
        self,
        time_features: Tensor,
        z_paths: Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the GP posterior predictive mean and variance over SDE paths.

        For each of the ``n_paths`` latent force trajectories z^(m), the GP
        posterior is evaluated at the joint (time_features, z^(m)) inputs.
        The inducing points and kernel parameters are shared across paths,
        keeping memory O(M × M_inducing²) rather than O(M × n²).

        Parameters
        ----------
        time_features : Tensor, shape (batch, horizon, n_calendar)
            Calendar/time features for each forecast horizon step.
        z_paths : Tensor, shape (batch, n_paths, horizon, latent_dim)
            SDE-sampled latent force trajectories.
        likelihood : gpytorch.likelihoods.Likelihood
            Observation likelihood; used to add observation noise to the
            predictive variance (f_* → y_* marginalisation).

        Returns
        -------
        gp_mean : Tensor, shape (batch, n_paths, horizon)
            Posterior predictive mean at each (path, horizon-step).
        gp_var : Tensor, shape (batch, n_paths, horizon)
            Posterior predictive variance (diagonal of the predictive
            covariance), including observation noise from the likelihood.

        Notes
        -----
        The full (horizon × horizon) posterior covariance is not returned to
        avoid O(horizon²) memory per path.  If the downstream flow needs
        cross-horizon correlations, call ``self(x_flat)`` directly and keep
        the full MultivariateNormal object.
        """
        batch = time_features.shape[0]
        n_paths = z_paths.shape[1]
        horizon = time_features.shape[1]

        # Build GP input tensor: (batch*n_paths, horizon, input_dim)
        gp_inputs = prepare_inputs(
            time_features=time_features,
            z_paths=z_paths,
            batch=batch,
            n_paths=n_paths,
            horizon=horizon,
            n_calendar=self.n_calendar,
        )  # (B*M, T, input_dim)

        # Flatten to (B*M*T, input_dim) for a single GP forward pass —
        # the inducing point approximation handles all points jointly.
        B_M = batch * n_paths
        x_flat = gp_inputs.reshape(B_M * horizon, self.input_dim)  # (B*M*T, D)

        # Evaluate GP posterior (uses variational strategy internally).
        # cholesky_jitter: elevated tolerance for non-stationary warped kernel.
        # fast_pred_var: uses LOVE approximation for O(1) predictive variance.
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
            f_dist = self(x_flat)  # MultivariateNormal (B*M*T,)
            y_dist = likelihood(f_dist)  # adds obs noise to variance

        # Extract mean and variance — diagonal only for memory efficiency
        pred_mean = y_dist.mean  # (B*M*T,)
        pred_var = y_dist.variance  # (B*M*T,)

        # Reshape back to (batch, n_paths, horizon)
        gp_mean = pred_mean.reshape(batch, n_paths, horizon)  # (B, M, T)
        gp_var = pred_var.reshape(batch, n_paths, horizon)  # (B, M, T)

        return gp_mean, gp_var


# =============================================================================
# __main__ — smoke test: verify predict_on_paths returns correct shapes
# =============================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    # ----- Dimension configuration -----
    latent_dim = 3
    n_calendar = 5  # [time_idx, hour_sin, hour_cos, dow_sin, dow_cos]
    input_dim = n_calendar + latent_dim  # 8
    n_inducing = 16  # tiny for fast smoke test (use 256 in production)

    batch = 4
    n_paths = 2
    horizon = 24

    print("=" * 60)
    print("ForceConditionedGP smoke test")
    print(
        f"  input_dim  = {input_dim}  (n_calendar={n_calendar}, latent_dim={latent_dim})"
    )
    print(f"  n_inducing = {n_inducing}")
    print(f"  batch={batch}, n_paths={n_paths}, horizon={horizon}")
    print("=" * 60)

    # ----- Construct model + likelihood -----
    gp = ForceConditionedGP(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_inducing=n_inducing,
        n_calendar=n_calendar,
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    print(f"\nGP parameter count  : {sum(p.numel() for p in gp.parameters())}")
    print(f"Likelihood params   : {sum(p.numel() for p in likelihood.parameters())}")

    # ----- Fake inputs -----
    # time_features: (batch, horizon, n_calendar)
    #   First column = monotonic time index
    time_idx = torch.arange(horizon, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    time_idx = time_idx.expand(batch, horizon).unsqueeze(-1)  # (B, T, 1)
    cal_rest = torch.randn(batch, horizon, n_calendar - 1)
    time_features = torch.cat([time_idx, cal_rest], dim=-1)  # (B, T, n_cal)

    # z_paths: (batch, n_paths, horizon, latent_dim)
    z_paths = torch.randn(batch, n_paths, horizon, latent_dim)

    print(f"\ntime_features shape : {time_features.shape}")
    print(f"z_paths shape       : {z_paths.shape}")

    # ----- Test prepare_inputs -----
    gp_inputs = prepare_inputs(
        time_features=time_features,
        z_paths=z_paths,
        batch=batch,
        n_paths=n_paths,
        horizon=horizon,
        n_calendar=n_calendar,
    )
    expected_shape = (batch * n_paths, horizon, input_dim)
    print(f"\nprepare_inputs output shape : {gp_inputs.shape}")
    assert gp_inputs.shape == expected_shape, (
        f"prepare_inputs: expected {expected_shape}, got {gp_inputs.shape}"
    )
    print("  [PASS] prepare_inputs shape correct")

    # ----- Test predict_on_paths -----
    gp.eval()
    likelihood.eval()

    with torch.no_grad():
        gp_mean, gp_var = gp.predict_on_paths(
            time_features=time_features,
            z_paths=z_paths,
            likelihood=likelihood,
        )

    print("\npredict_on_paths outputs:")
    print(
        f"  gp_mean shape : {gp_mean.shape}  (expected ({batch}, {n_paths}, {horizon}))"
    )
    print(
        f"  gp_var  shape : {gp_var.shape}  (expected ({batch}, {n_paths}, {horizon}))"
    )

    assert gp_mean.shape == (batch, n_paths, horizon), (
        f"gp_mean shape mismatch: expected ({batch},{n_paths},{horizon}), "
        f"got {gp_mean.shape}"
    )
    assert gp_var.shape == (batch, n_paths, horizon), (
        f"gp_var shape mismatch: expected ({batch},{n_paths},{horizon}), "
        f"got {gp_var.shape}"
    )
    print("  [PASS] predict_on_paths shapes correct")

    # Variance must be non-negative everywhere
    assert (gp_var >= 0).all(), "gp_var contains negative values"
    print("  [PASS] gp_var >= 0 everywhere")

    # ----- Test compute_elbo -----
    gp.train()
    likelihood.train()

    n_train = batch * horizon  # notional dataset size for ELBO scaling
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=n_train)

    # Flatten inputs for ELBO (use first path for simplicity)
    x_elbo = gp_inputs[:batch, :, :].reshape(batch * horizon, input_dim)  # (B*T, D)
    y_elbo = torch.randn(batch * horizon)  # (B*T,)

    loss = gp.compute_elbo(x=x_elbo, y=y_elbo, likelihood=likelihood, mll=mll)
    print(f"\ncompute_elbo loss   : {loss.item():.4f}  (scalar={loss.shape == ()})")
    assert loss.shape == (), f"ELBO loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"ELBO loss is not finite: {loss.item()}"
    print("  [PASS] compute_elbo returns finite scalar")

    # Check gradients flow through
    loss.backward()
    grad_check_params = [
        ("mean_module.constant", gp.mean_module.constant),
        ("log_A", gp.covar_module.log_A),
    ]
    for name, param in grad_check_params:
        assert param.grad is not None, f"No gradient for {name}"
        print(
            f"  [PASS] grad flows through {name}  (|g|={param.grad.abs().item():.4e})"
        )

    # ----- Final summary -----
    print("\n[PASS] All ForceConditionedGP checks passed.")
    sys.exit(0)
