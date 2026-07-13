# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# force_kernel.py — Force-Conditioned Warped GP Kernel (GPyTorch Custom Kernel)
#
# Role (Layer 3a in the full pipeline):
#   Implements the WarpedForceKernel, a non-stationary covariance function that
#   adapts its correlation structure based on the latent force state z_t sampled
#   from the Neural SDE.  The kernel blends two components via a learned gate:
#
#     K_warped(t, t', z_t, z_{t'}) =
#         K_base(h(x), h(x')) * gate(z_t, z_{t'})       [smooth periodic]
#       + K_regime(t, t', δ)    * (1 - gate(z_t, z_{t'})) [sharp transitions]
#
#   where:
#     K_base    — SpectralMixtureKernel operating on DKL-learned features h(x)
#                 captures 24h / 168h / annual periodicity (Wilson-Adams 2013)
#     K_regime  — Matérn-1/2 kernel with force-adaptive lengthscale
#                 ℓ(δ) = regime_lengthscale_net(δ) + 0.1, δ = ‖z_t - z_{t'}‖₂
#     gate      — sigmoid(MLP([z_t, z_{t'}])) ∈ (0, 1)
#                 near 1 in calm markets (K_base dominates)
#                 near 0 during price spikes (K_regime dominates)
#     h_θ       — Deep Kernel Learning MLP: (cal + z) → cal-dim feature space
#
# Input tensor layout:
#   X : (n, input_dim)  where input_dim = n_calendar + latent_dim
#       X[:, :n_calendar]  — calendar/time features (time-index, hour sin/cos, …)
#       X[:, n_calendar:]  — latent force z_t (latent_dim = 3)
#
# Key design decisions:
#   * All intermediate kernel matrices are evaluated eagerly via .evaluate() to
#     prevent lazy-tensor shape mismatches inside the custom forward().
#   * gate is computed with full (n1, n2) broadcasting — no Python loops.
#   * regime_lengthscale_net uses Softplus output so ℓ is strictly positive;
#     an additional +0.1 floor prevents numerical collapse at δ = 0.
#   * log_A (amplitude of K_regime) is a free scalar nn.Parameter initialised
#     at 0 → A = 1, with gradient flowing freely during training.
#   * is_stationary returns False so GPyTorch skips stationarity assumptions
#     (e.g., it will not attempt to symmetrise the kernel matrix from cache).
#
# References:
#   Wilson & Adams (ICML 2013)  "Gaussian Process Kernels for Pattern Discovery"
#   Wilson et al. (AISTATS 2016) "Deep Kernel Learning"
#   LF_GP_NRF_Research_Plan.md §3.3
# =============================================================================
from __future__ import annotations  # noqa: E402  (must be first real statement)

from typing import Optional

import gpytorch
import torch
import torch.nn as nn
from linear_operator.operators import DenseLinearOperator
from torch import Tensor

# ---------------------------------------------------------------------------
# Internal MLP builder — avoids repetition across sub-module definitions
# ---------------------------------------------------------------------------


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    hidden_activation: nn.Module,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Construct a two-layer MLP with configurable activations.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Width of the single hidden layer.
    out_dim : int
        Output feature dimension.
    hidden_activation : nn.Module
        Activation applied after the first linear layer.
    output_activation : nn.Module or None
        Activation applied after the final linear layer.  If None, the output
        is left as a raw linear projection.

    Returns
    -------
    nn.Sequential
        The assembled MLP.
    """
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        hidden_activation,
        nn.Linear(hidden_dim, out_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# WarpedForceKernel
# ---------------------------------------------------------------------------


class WarpedForceKernel(gpytorch.kernels.Kernel):
    """Non-stationary kernel warped by the latent force state z_t.

    The kernel operates on concatenated input tensors of shape (n, input_dim)
    where the layout is::

        X = [ cal_features | z_t ]
              <─ n_calendar ─>  <latent_dim>

    The covariance is computed as::

        K(x1, x2) = K_base(h(x1), h(x2)) * gate(z1, z2)
                  + K_regime(t1, t2, δ)   * (1 - gate(z1, z2))

    Parameters
    ----------
    input_dim : int
        Total input dimension = n_calendar + latent_dim.
    latent_dim : int
        Dimensionality of the latent force z_t.  Default 3.
    n_calendar : int
        Number of calendar/time features.  Derived as input_dim - latent_dim
        when not provided explicitly; must equal input_dim - latent_dim.
    dkl_hidden : int
        Hidden layer width of the deep kernel feature-extractor MLP.  Default 64.
    gate_hidden : int
        Hidden layer width of the gating MLP.  Default 32.

    Sub-modules
    -----------
    base_kernel
        SpectralMixtureKernel(num_mixtures=4, ard_num_dims=n_calendar).
        Captures multi-scale periodicity in the DKL feature space.
    regime_kernel
        MaternKernel(nu=0.5, ard_num_dims=n_calendar).
        Produces non-differentiable (C^0) sample paths — appropriate for
        electricity prices that can jump within a single hour.
    feature_net
        MLP(n_calendar + latent_dim → dkl_hidden → n_calendar, Tanh).
        Deep kernel feature extractor; maps (cal, z) → n_calendar-dim features
        on which base_kernel and regime_kernel operate.
    gate_net
        MLP(latent_dim * 2 → gate_hidden → 1, Tanh → Sigmoid).
        Scalar gate g ∈ (0, 1) for each pair (z1_i, z2_j).
    regime_lengthscale_net
        MLP(1 → 16 → 1, Tanh → Softplus).
        Maps force distance δ_ij ∈ ℝ≥0 to adaptive lengthscale ℓ_ij.
    log_A
        Scalar nn.Parameter, initialised at 0.  A = exp(log_A) is the
        amplitude of the regime kernel component.
    """

    # Tell GPyTorch this kernel is non-stationary — disables symmetry caching.
    @property
    def is_stationary(self) -> bool:  # type: ignore[override]
        return False

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        n_calendar: Optional[int] = None,
        dkl_hidden: int = 64,
        gate_hidden: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Dimension book-keeping
        # ------------------------------------------------------------------
        if n_calendar is None:
            n_calendar = input_dim - latent_dim

        if n_calendar != input_dim - latent_dim:
            raise ValueError(
                f"n_calendar ({n_calendar}) must equal "
                f"input_dim - latent_dim ({input_dim} - {latent_dim} = "
                f"{input_dim - latent_dim})."
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_calendar = n_calendar
        self.dkl_hidden = dkl_hidden
        self.gate_hidden = gate_hidden

        # ------------------------------------------------------------------
        # 1. Base kernel — lightweight RBF + Periodic sum operating on DKL
        #    features.  SpectralMixtureKernel was removed because it
        #    materialises a full (n1 x n2) dense matrix even for the diag path,
        #    causing OOM on 6144-point batches.  RBF + Periodic is O(n)
        #    for the diagonal and O(n^2) only when the full matrix is needed.
        #
        #    Two periodic components (24 h and 168 h) capture the dominant
        #    electricity-market seasonalities; the RBF captures smooth
        #    non-periodic variation.
        # ------------------------------------------------------------------
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=n_calendar)
        self.periodic_24h = gpytorch.kernels.PeriodicKernel()  # hour-of-day
        self.periodic_168h = gpytorch.kernels.PeriodicKernel()  # day-of-week
        # Scale wrappers so each component has a learnable output variance
        self.base_kernel = gpytorch.kernels.ScaleKernel(
            self.rbf_kernel + self.periodic_24h + self.periodic_168h
        )

        # ------------------------------------------------------------------
        # 2. Regime kernel — Matérn-1/2 for sharp price transitions
        #    ard_num_dims=n_calendar: the time-index dimension gets its own ℓ
        # ------------------------------------------------------------------
        self.regime_kernel = gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=n_calendar,
        )

        # ------------------------------------------------------------------
        # 3. Deep Kernel Learning feature extractor
        #    Maps the full (cal + z) input → n_calendar-dim feature space.
        #    Tanh keeps features in a bounded range suitable for the SM kernel.
        # ------------------------------------------------------------------
        self.feature_net = _build_mlp(
            in_dim=n_calendar + latent_dim,  # = input_dim
            hidden_dim=dkl_hidden,
            out_dim=n_calendar,
            hidden_activation=nn.Tanh(),
            output_activation=nn.Tanh(),
        )

        # ------------------------------------------------------------------
        # 4. Gating network
        #    Input : concatenation of two latent vectors [z1_i, z2_j] (2*latent_dim)
        #    Output: scalar in (0, 1) via Tanh-hidden + Sigmoid
        # ------------------------------------------------------------------
        self.gate_net = _build_mlp(
            in_dim=latent_dim * 2,
            hidden_dim=gate_hidden,
            out_dim=1,
            hidden_activation=nn.Tanh(),
            output_activation=nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # 5. Regime lengthscale network
        #    Input : scalar δ_ij = ‖z1_i - z2_j‖₂  (shape (n1, n2, 1))
        #    Output: adaptive lengthscale ℓ_ij > 0.1
        #    Softplus ensures ℓ > 0 before the +0.1 floor is added.
        # ------------------------------------------------------------------
        self.regime_lengthscale_net = _build_mlp(
            in_dim=1,
            hidden_dim=16,
            out_dim=1,
            hidden_activation=nn.Tanh(),
            output_activation=nn.Softplus(),
        )

        # ------------------------------------------------------------------
        # 6. Regime kernel amplitude (log scale for unconstrained optimisation)
        #    A = exp(log_A).  Initialised at 0 → A = 1.
        # ------------------------------------------------------------------
        self.log_A = nn.Parameter(torch.tensor(0.0))

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers; zero biases."""
        for module in [self.feature_net, self.gate_net, self.regime_lengthscale_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward — covariance matrix computation
    # ------------------------------------------------------------------

    def _base_kernel_diag(self, h1: Tensor) -> Tensor:
        """Compute only the diagonal of the base kernel K_base(h, h).

        Avoids constructing the full (n x n) matrix, which is the dominant
        OOM cause when n = batch * n_paths * horizon ~ 6000+.
        """
        # RBF diagonal is always 1 * output_scale (identical inputs → max corr)
        n = h1.shape[0]
        device = h1.device
        dtype = h1.dtype
        # For a stationary kernel k(x, x) = k(0) = output_scale.
        # We obtain output_scale from the ScaleKernel's outputscale parameter.
        out_scale = self.base_kernel.outputscale  # scalar
        return out_scale.expand(n).to(device=device, dtype=dtype)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        """Compute the (n1, n2) warped covariance matrix.

        Parameters
        ----------
        x1 : Tensor, shape (n1, input_dim)
            First set of input points [cal_features | z_t].
        x2 : Tensor, shape (n2, input_dim)
            Second set of input points.

        Returns
        -------
        Tensor, shape (n1, n2)
            Warped covariance matrix K(x1, x2).
        """
        # ------------------------------------------------------------------
        # 0. Split inputs into calendar and latent-force parts
        # ------------------------------------------------------------------
        # x1: (n1, input_dim)  →  cal1: (n1, n_calendar), z1: (n1, latent_dim)
        # x2: (n2, input_dim)  →  cal2: (n2, n_calendar), z2: (n2, latent_dim)
        cal1 = x1[..., : self.n_calendar]  # (n1, n_calendar)
        z1 = x1[..., self.n_calendar :]  # (n1, latent_dim)
        cal2 = x2[..., : self.n_calendar]  # (n2, n_calendar)
        z2 = x2[..., self.n_calendar :]  # (n2, latent_dim)

        n1 = x1.shape[0]
        n2 = x2.shape[0]

        # ------------------------------------------------------------------
        # 1. Deep Kernel Learning feature extraction
        #    h(x) = feature_net(x_full) ∈ ℝ^{n_calendar}
        #    Operating on the full (cal, z) input lets the MLP learn how
        #    calendar features interact with market-stress regime.
        # ------------------------------------------------------------------
        h1 = self.feature_net(x1)  # (n1, n_calendar)
        h2 = self.feature_net(x2)  # (n2, n_calendar)

        # ------------------------------------------------------------------
        # 2. Base covariance in DKL feature space
        # ------------------------------------------------------------------
        if diag:
            # Fast diagonal path — no full matrix needed.
            # For stationary kernels k(x,x) = outputscale everywhere.
            K_base_d: Tensor = self._base_kernel_diag(h1)  # (n1,)
            # Gate diagonal: z_pairs_d shape (n1, 2*latent_dim)
            z_pairs_d = torch.cat([z1, z1], dim=-1)  # (n1, 2d) — x1==x2 on diag
            gate_d: Tensor = self.gate_net(z_pairs_d).squeeze(-1)  # (n1,)
            # Regime kernel diagonal: distance=0 → ℓ=base, exp(0)=1
            A = torch.exp(self.log_A)
            K_regime_d = A * torch.ones(n1, device=x1.device, dtype=x1.dtype)
            diag_vals = K_base_d * gate_d + K_regime_d * (1.0 - gate_d)  # (n1,)
            return diag_vals

        # Full matrix path — only reached when diag=False
        K_base: Tensor = self.base_kernel(h1, h2).to_dense()  # (n1, n2)

        # ------------------------------------------------------------------
        # 3. Gating function — computed for all (i, j) pairs via broadcasting
        #
        #    z1_exp : (n1, 1, latent_dim) → broadcast → (n1, n2, latent_dim)
        #    z2_exp : (1, n2, latent_dim) → broadcast → (n1, n2, latent_dim)
        #    z_pairs: (n1, n2, 2*latent_dim)  after cat on last dim
        #    gate   : (n1, n2, 1) → squeeze → (n1, n2)
        # ------------------------------------------------------------------
        z1_exp = z1.unsqueeze(1).expand(n1, n2, self.latent_dim)  # (n1, n2, d)
        z2_exp = z2.unsqueeze(0).expand(n1, n2, self.latent_dim)  # (n1, n2, d)
        z_pairs = torch.cat([z1_exp, z2_exp], dim=-1)  # (n1, n2, 2d)

        # Apply gate_net on the last dim — reshape to (n1*n2, 2d) for the MLP,
        # then reshape back to (n1, n2).
        z_pairs_flat = z_pairs.reshape(n1 * n2, self.latent_dim * 2)  # (n1*n2, 2d)
        gate_flat = self.gate_net(z_pairs_flat)  # (n1*n2, 1)
        gate: Tensor = gate_flat.reshape(n1, n2)  # (n1, n2)

        # ------------------------------------------------------------------
        # 4. Force distance δ_ij = ‖z1_i - z2_j‖₂
        #    diff : (n1, n2, latent_dim)
        #    delta: (n1, n2)
        # ------------------------------------------------------------------
        diff = z1_exp - z2_exp  # (n1, n2, d)
        delta = torch.norm(diff, p=2, dim=-1)  # (n1, n2)

        # ------------------------------------------------------------------
        # 5. Adaptive regime lengthscale ℓ_ij
        #    regime_lengthscale_net maps δ_ij (scalar) → ℓ_ij via a small MLP
        #    with Softplus output (strictly positive), plus a 0.1 floor.
        #
        #    delta_flat : (n1*n2, 1)
        #    ell_flat   : (n1*n2, 1)
        #    ell        : (n1, n2)
        # ------------------------------------------------------------------
        delta_flat = delta.reshape(n1 * n2, 1)  # (n1*n2, 1)
        ell_flat = self.regime_lengthscale_net(delta_flat)  # (n1*n2, 1)
        ell: Tensor = ell_flat.reshape(n1, n2) + 0.1  # (n1, n2) > 0.1

        # ------------------------------------------------------------------
        # 6. Regime covariance K_regime = A * exp(-|t1_i - t2_j| / ℓ_ij)
        #
        #    The "time index" is the FIRST calendar feature (dimension 0), which
        #    is expected to be a monotonic hour/timestep counter that the
        #    data-loading pipeline should place first in the calendar block.
        #
        #    t1: (n1,)  →  t1_exp: (n1, 1) → broadcast → (n1, n2)
        #    t2: (n2,)  →  t2_exp: (1, n2) → broadcast → (n1, n2)
        # ------------------------------------------------------------------
        t1 = cal1[:, 0]  # (n1,)  — raw time index
        t2 = cal2[:, 0]  # (n2,)
        t1_exp = t1.unsqueeze(1).expand(n1, n2)  # (n1, n2)
        t2_exp = t2.unsqueeze(0).expand(n1, n2)  # (n1, n2)

        time_dist = torch.abs(t1_exp - t2_exp)  # (n1, n2)  ≥ 0
        A = torch.exp(self.log_A)  # scalar
        K_regime: Tensor = A * torch.exp(-time_dist / ell)  # (n1, n2)

        # ------------------------------------------------------------------
        # 7. Blend: K_warped = K_base * gate + K_regime * (1 - gate)
        # ------------------------------------------------------------------
        K_warped = K_base * gate + K_regime * (1.0 - gate)  # (n1, n2)

        # 8. Diagonal nugget — guarantees positive definiteness at all times.
        #    Added only when x1 and x2 have the same shape (square evaluation,
        #    i.e. the K(Z_m, Z_m) inducing-point matrix).  Off-diagonal
        #    cross-covariance K(X, Z_m) does not need the nugget.
        if n1 == n2:
            nugget = 1e-3 * torch.eye(n1, device=K_warped.device, dtype=K_warped.dtype)
            K_warped = K_warped + nugget

        return DenseLinearOperator(K_warped)


# =============================================================================
# __main__ — smoke test: verify (8, 8) output and positive semi-definiteness
# =============================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)

    # ----- Kernel configuration -----
    latent_dim = 3
    n_calendar = 5  # [time_index, hour_sin, hour_cos, dow_sin, dow_cos]
    input_dim = n_calendar + latent_dim  # = 8

    print("=" * 60)
    print("WarpedForceKernel smoke test")
    print(f"  input_dim  = {input_dim}")
    print(f"  n_calendar = {n_calendar}")
    print(f"  latent_dim = {latent_dim}")
    print("=" * 60)

    kernel = WarpedForceKernel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_calendar=n_calendar,
        dkl_hidden=64,
        gate_hidden=32,
    )
    kernel.eval()

    # ----- Construct 8 fake input points -----
    # Calendar features: time_index = 0..7, random sin/cos/dow features
    n_pts = 8
    time_idx = torch.arange(n_pts, dtype=torch.float32).unsqueeze(1)  # (8, 1)
    cal_rest = torch.randn(n_pts, n_calendar - 1)  # (8, 4)
    cal = torch.cat([time_idx, cal_rest], dim=1)  # (8, 5)
    z = torch.randn(n_pts, latent_dim)  # (8, 3)
    X = torch.cat([cal, z], dim=1)  # (8, 8)

    print(f"\nInput X shape : {X.shape}")

    # ----- Forward pass -----
    with torch.no_grad():
        K = kernel(X, X).evaluate()  # use .evaluate() to get plain tensor

    print(f"K shape       : {K.shape}")
    assert K.shape == (n_pts, n_pts), f"Expected ({n_pts}, {n_pts}), got {K.shape}"
    print(f"K dtype       : {K.dtype}")

    # ----- Symmetry check -----
    sym_err = (K - K.T).abs().max().item()
    print(f"\nSymmetry error (max |K - Kᵀ|) : {sym_err:.2e}")
    assert sym_err < 1e-5, f"Kernel is not symmetric: max error = {sym_err}"

    # ----- Positive semi-definiteness check -----
    # Add a small nugget for numerical stability before eigendecomposition
    nugget = 1e-4
    K_pd = K + nugget * torch.eye(n_pts)
    eigvals = torch.linalg.eigvalsh(K_pd)
    min_eig = eigvals.min().item()
    print(f"Min eigenvalue (with nugget {nugget}) : {min_eig:.4f}")
    assert min_eig > 0.0, f"Kernel matrix is not PSD — min eigenvalue = {min_eig:.4f}"

    # ----- Value range sanity -----
    print(f"\nDiagonal values  : {K.diag().numpy().round(4)}")
    print(f"Off-diag sample  : K[0,1] = {K[0, 1].item():.4f}")
    print(f"K min / max      : {K.min().item():.4f} / {K.max().item():.4f}")

    # ----- is_stationary check -----
    assert not kernel.is_stationary, "Kernel should be non-stationary"
    print(f"\nis_stationary    : {kernel.is_stationary}  (expected False)")

    # ----- Parameter count -----
    n_params = sum(p.numel() for p in kernel.parameters())
    print(f"Trainable params : {n_params}")

    # ----- Non-square evaluation (n1 ≠ n2) -----
    X2 = torch.cat(
        [
            torch.cat(
                [
                    torch.arange(3, dtype=torch.float32).unsqueeze(1),
                    torch.randn(3, n_calendar - 1),
                ],
                dim=1,
            ),
            torch.randn(3, latent_dim),
        ],
        dim=1,
    )  # (3, 8)

    with torch.no_grad():
        K_rect = kernel(X, X2).evaluate()

    print(f"\nRectangular K(8, 3) shape : {K_rect.shape}")
    assert K_rect.shape == (n_pts, 3), f"Expected (8, 3), got {K_rect.shape}"

    print("\n[PASS] All WarpedForceKernel checks passed.")
    sys.exit(0)
