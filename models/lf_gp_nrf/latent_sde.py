# =============================================================================
# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# latent_sde.py — Neural SDE Latent Force Field (Layer 1)
#
# Role in pipeline:
#   Receives the initial condition distribution q(z_0 | x) and the time-varying
#   context from LatentForceEncoder, then integrates a Neural SDE forward over
#   the 24-hour forecast horizon to produce an ensemble of latent trajectories:
#
#       z_t  ~  dz = f_θ(z, ctx, exog) dt + g_φ(z) dW_t
#
#   The posterior drift f_θ is conditioned on the encoder context and future
#   exogenous features.  A fixed OU prior f_prior(z) = -0.1·z enables a
#   tractable Girsanov KL divergence estimate used in the ELBO.
#
# Architecture:
#   drift_net       MLP (latent+context+exog → hidden → hidden → latent), Tanh
#   diffusion_net   MLP (latent → hidden//2 → latent), Tanh → Softplus + 0.05
#   prior_drift     fixed: f_prior(z) = -0.1·z  (OU mean reversion, frozen)
#
# SDE details:
#   noise_type  : 'diagonal'       (one noise channel per latent dim)
#   sde_type    : 'stratonovich'   (required for Reversible Heun solver)
#   solver      : 'reversible_heun'
#   dt          : 0.5 hours        (two sub-steps per 1-hour data interval)
#   horizon     : 0 → 24 hours     (25 timepoints, 24 intervals)
#
# KL divergence:
#   Girsanov formula, approximated by discrete sum:
#       KL ≈ (dt/2) · Σ_t ‖u_t‖²,  u_t = (f_post - f_prior) / (g + ε)
#
# Dimension key (defaults):
#   latent_dim   = 3
#   context_dim  = 128   (from LatentForceEncoder hidden_dim)
#   exog_dim     = 6     (future exogenous features per timestep)
#   hidden_dim   = 64
#
# References:
#   Li et al. (2020) "Scalable Gradients for Stochastic Differential Equations"
#   Kidger et al. (2021) "Efficient and Accurate Gradients for Neural SDEs"
#   LF_GP_NRF_Research_Plan.md §4.4–4.6
# =============================================================================

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchsde
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchsde is required for LatentSDE.  Install with: pip install torchsde==0.2.6"
    ) from e


# ---------------------------------------------------------------------------
# Helper: build a simple MLP
# ---------------------------------------------------------------------------


def _mlp(
    dims: list[int], activation: nn.Module, final_activation: nn.Module | None = None
) -> nn.Sequential:
    """Construct a fully-connected network from a list of layer widths.

    Parameters
    ----------
    dims:
        List of integers [in, h1, h2, ..., out] specifying each layer width.
    activation:
        Activation inserted between every pair of consecutive linear layers.
    final_activation:
        Optional activation appended after the last linear layer.  If None,
        no activation is applied to the output.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation)
        else:
            # last linear layer
            if final_activation is not None:
                layers.append(final_activation)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# LatentSDE
# ---------------------------------------------------------------------------


class LatentSDE(nn.Module):
    """Neural SDE latent force field compatible with torchsde's ``sdeint`` API.

    The module stores a *trajectory* of context and exogenous tensors set via
    :meth:`set_trajectory` before each ``sdeint`` call.  Inside ``f`` and
    ``g``, continuous-time context is obtained by linear interpolation between
    the integer-indexed horizon steps.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent state z(t).  Default 3.
    context_dim : int
        Dimensionality of the context vector produced by the encoder at each
        forecast timestep.  Default 128.
    exog_dim : int
        Number of future exogenous features per timestep.  Default 6.
    hidden_dim : int
        Hidden width for the drift MLP.  Diffusion MLP uses ``hidden_dim//2``.
        Default 64.
    dt : float
        SDE integration step in hours.  Default 0.5 (two sub-steps per hour).
    noise_type : str
        torchsde noise type.  Default ``'diagonal'``.
    sde_type : str
        torchsde SDE type.  Must be ``'stratonovich'`` for Reversible Heun.
        Default ``'stratonovich'``.
    """

    # torchsde reads these class attributes to configure the solver
    noise_type: str
    sde_type: str

    def __init__(
        self,
        latent_dim: int = 3,
        context_dim: int = 128,
        exog_dim: int = 6,
        hidden_dim: int = 64,
        dt: float = 0.5,
        noise_type: str = "diagonal",
        sde_type: str = "stratonovich",
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.exog_dim = exog_dim
        self.hidden_dim = hidden_dim
        self.dt = dt

        # torchsde reads these as class-level attributes; set on instance too
        self.noise_type = noise_type
        self.sde_type = sde_type

        # ------------------------------------------------------------------
        # Sub-network 1: posterior drift
        #   input : (latent_dim + context_dim + exog_dim)
        #   hidden: hidden_dim → hidden_dim
        #   output: latent_dim   (no final activation)
        # ------------------------------------------------------------------
        drift_in = latent_dim + context_dim + exog_dim
        self.drift_net = nn.Sequential(
            nn.Linear(drift_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            # no output activation — raw drift
        )

        # ------------------- INNOVATION: Merit-Order Potential -------------------
        # Physical "restoring force" derived from a scalar potential H(z, exog).
        # Ensures latent forces are grounded in Supply/Demand equilibrium.
        # -------------------------------------------------------------------------
        self.potential_net = nn.Sequential(
            nn.Linear(latent_dim + exog_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Scalar potential
        )
        self.potential_weight = nn.Parameter(torch.tensor(0.02))
        self.zlb_weight = nn.Parameter(torch.tensor(0.02))

        # ------------------------------------------------------------------
        # Sub-network 2: diagonal diffusion
        #   input : latent_dim
        #   hidden: hidden_dim // 2
        #   output: latent_dim  via Tanh → Softplus + 0.05
        #   Range : (0.05, ~1.55) — ensures strictly positive diffusion
        # ------------------------------------------------------------------
        diff_hidden = max(hidden_dim // 2, 1)
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim, diff_hidden),
            nn.Tanh(),
            nn.Linear(diff_hidden, latent_dim),
            nn.Softplus(),
        )
        self._diff_floor = 0.05  # minimum diffusion coefficient

        # ------------------------------------------------------------------
        # Sub-network 3: prior drift (frozen OU mean reversion)
        #   f_prior(z) = -0.1 * z   — implemented analytically, no parameters
        # ------------------------------------------------------------------
        self._prior_mean_reversion: float = 0.1

        # ------------------------------------------------------------------
        # Trajectory buffers — populated by set_trajectory() before sdeint
        # ------------------------------------------------------------------
        # Shape: (batch, horizon, context_dim) and (batch, horizon, exog_dim)
        self.context_trajectory: Tensor | None = None
        self.exog_trajectory: Tensor | None = None
        self._horizon: int = 0

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights with small orthogonal initialisation for drift,
        and near-zero initialisation for diffusion to start training stable."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "diffusion" in name:
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.orthogonal_(module.weight, gain=0.5)
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Trajectory management
    # ------------------------------------------------------------------

    def set_trajectory(self, context: Tensor, exog: Tensor) -> None:
        """Cache encoder context and exogenous features for use inside f/g.

        Must be called immediately before ``torchsde.sdeint``.

        Parameters
        ----------
        context : Tensor
            Shape ``(batch, horizon, context_dim)``.
        exog : Tensor
            Shape ``(batch, horizon, exog_dim)``.
        """
        self.context_trajectory = context  # (B, H, context_dim)
        self.exog_trajectory = exog  # (B, H, exog_dim)
        self._horizon = context.shape[1]

    # ------------------------------------------------------------------
    # Continuous-time linear interpolation helper
    # ------------------------------------------------------------------

    def _interpolate_at_t(self, t: Tensor, trajectory: Tensor) -> Tensor:
        """Linearly interpolate a (batch, horizon, feat) trajectory at time t.

        Parameters
        ----------
        t : Tensor
            Scalar (0-dim) or 1-dim tensor giving the current SDE time.
        trajectory : Tensor
            Shape ``(batch_expanded, horizon, feat_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch_expanded, feat_dim)``.
        """
        H = trajectory.shape[1]

        # Convert t to a Python float for indexing arithmetic
        t_float = t.item() if t.numel() == 1 else float(t)

        # Clamp to valid index range
        t_lo = max(0, min(int(math.floor(t_float)), H - 1))
        t_hi = max(0, min(t_lo + 1, H - 1))
        alpha = t_float - math.floor(t_float)
        alpha = max(0.0, min(alpha, 1.0))

        lo = trajectory[:, t_lo, :]  # (B_expanded, feat_dim)
        hi = trajectory[:, t_hi, :]  # (B_expanded, feat_dim)

        return lo + alpha * (hi - lo)

    # ------------------------------------------------------------------
    # torchsde interface: f (drift) and g (diffusion)
    # ------------------------------------------------------------------

    def f(self, t: Tensor, z: Tensor) -> Tensor:
        """Posterior drift f_θ(t, z) with Physics-Informed Merit-Order Potential.

        Includes four components:
            1. Learned Neural Drift (unstructured black-box)
            2. Potential Force (gradient of Merit-Order Potential H)
            3. ZLB Barrier (prevents unphysical negative prices without RE)
            4. Stability Wall (prevents numerical divergence)
        """
        # Interpolate time-varying conditioning signals
        context_t = self._interpolate_at_t(t, self.context_trajectory)
        exog_t = self._interpolate_at_t(t, self.exog_trajectory)

        # 1. Learned Neural Drift
        inp = torch.cat([z, context_t, exog_t], dim=-1)
        nn_drift = self.drift_net(inp)

        # 2. Merit-Order Potential Force (Innovation)
        # We compute -grad_z H(z, exog) to find the equilibrium restoring force.
        # This aligns the latent force with the market supply stack structure.
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            pot_inp = torch.cat([z_in, exog_t], dim=-1)
            potential = self.potential_net(pot_inp)
            # Higher-order gradient allows backprop through the SDE solver
            grad_z = torch.autograd.grad(
                potential.sum(), z_in, create_graph=True, retain_graph=True
            )[0]
        mo_force = -self.potential_weight * grad_z

        # 3. ZLB Physics Barrier (Innovation)
        # exog_t indices: [0:load, 1:solar, 2:wind_on, 3:wind_off, ...]
        solar_gen = exog_t[:, 1]
        # Restoring force: if z[0] (price proxy) is negative, check if RE justifies it.
        # If solar is low, push price back toward positive.
        z_price = z[:, 0]
        zlb_active = (z_price < -1.0).float()
        # Barrier is stronger when solar is weaker (ZLB violation)
        barrier_strength = torch.relu(1.0 - solar_gen)
        zlb_force_val = zlb_active * barrier_strength * torch.abs(z_price + 1.0)
        
        # Apply specifically to the first latent dimension (assumed price-dominant)
        zlb_force = torch.zeros_like(z)
        zlb_force[:, 0] = self.zlb_weight * zlb_force_val

        # 4. Standard Soft Boundary Wall (Stability)
        wall_mask = (z.abs() > 5.0).float()
        soft_wall = -0.2 * z * wall_mask

        return nn_drift + mo_force + zlb_force + soft_wall

    def g(self, t: Tensor, z: Tensor) -> Tensor:
        """Diagonal diffusion  g_φ(z).

        Parameters
        ----------
        t : Tensor
            Current time (unused — diffusion is time-homogeneous).
        z : Tensor
            Current latent state, shape ``(batch_expanded, latent_dim)``.

        Returns
        -------
        Tensor
            Diffusion coefficients, shape ``(batch_expanded, latent_dim)``.
            Guaranteed ≥ 0.05.
        """
        return self.diffusion_net(z) + self._diff_floor  # (B_exp, latent_dim)

    # ------------------------------------------------------------------
    # Prior drift (OU process — not trainable)
    # ------------------------------------------------------------------

    def f_prior(self, t: Tensor, z: Tensor) -> Tensor:
        """Prior OU drift  f_prior(z) = -0.1 · z.

        Parameters
        ----------
        t : Tensor
            Current time (unused — prior is time-homogeneous).
        z : Tensor
            Shape ``(batch_expanded, latent_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch_expanded, latent_dim)``.
        """
        return -self._prior_mean_reversion * z

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        z0: Tensor,
        context: Tensor,
        exog: Tensor,
        n_steps: int = 24,
        n_paths: int = 16,
    ) -> Tensor:
        """Sample latent trajectories from the posterior SDE.

        Parameters
        ----------
        z0 : Tensor
            Initial latent state, shape ``(batch, latent_dim)``.
        context : Tensor
            Encoder context, shape ``(batch, horizon, context_dim)``.
        exog : Tensor
            Future exogenous features, shape ``(batch, horizon, exog_dim)``.
        n_steps : int
            Number of hourly forecast steps (default 24).
        n_paths : int
            Number of Monte Carlo sample paths per batch element (default 16).

        Returns
        -------
        Tensor
            Sampled latent trajectories, shape
            ``(batch, n_paths, n_steps, latent_dim)``.
            Note: the returned tensor excludes the initial condition timestep
            (index 0) so that each of the n_steps entries corresponds to a
            forecast hour 1 … n_steps.
        """
        batch = z0.shape[0]
        device = z0.device
        dtype = z0.dtype

        # ------------------------------------------------------------------
        # Expand tensors to (batch * n_paths) for parallel path simulation
        # ------------------------------------------------------------------

        # z0: (batch, latent_dim) → (batch * n_paths, latent_dim)
        z0_expanded = z0.unsqueeze(1).expand(batch, n_paths, self.latent_dim)
        z0_expanded = z0_expanded.reshape(batch * n_paths, self.latent_dim).contiguous()

        # context: (batch, H, C) → (batch * n_paths, H, C)
        ctx_expanded = (
            context.unsqueeze(1)
            .expand(batch, n_paths, context.shape[1], self.context_dim)
            .reshape(batch * n_paths, context.shape[1], self.context_dim)
            .contiguous()
        )

        # exog: (batch, H, E) → (batch * n_paths, H, E)
        exog_expanded = (
            exog.unsqueeze(1)
            .expand(batch, n_paths, exog.shape[1], self.exog_dim)
            .reshape(batch * n_paths, exog.shape[1], self.exog_dim)
            .contiguous()
        )

        # Cache trajectories for use in f/g
        self.set_trajectory(ctx_expanded, exog_expanded)

        # ------------------------------------------------------------------
        # Integration timepoints: 0, 1, ..., n_steps  (hourly)
        # ------------------------------------------------------------------
        ts = torch.linspace(
            0.0, float(n_steps), n_steps + 1, device=device, dtype=dtype
        )

        # ------------------------------------------------------------------
        # Integrate the SDE
        # sdeint returns shape (n_steps+1, batch*n_paths, latent_dim)
        # ------------------------------------------------------------------
        z_seq = torchsde.sdeint(
            sde=self,
            y0=z0_expanded,
            ts=ts,
            method="reversible_heun",
            dt=self.dt,
            names={"drift": "f", "diffusion": "g"},
        )
        # z_seq: (T+1, B*P, latent_dim)  where T = n_steps

        # Drop initial condition (t=0), keep t=1…T
        z_seq = z_seq[1:]  # (n_steps, B*P, latent_dim)

        # Reshape to (batch, n_paths, n_steps, latent_dim)
        z_paths = (
            z_seq.permute(1, 0, 2).reshape(  # (B*P, n_steps, latent_dim)
                batch, n_paths, n_steps, self.latent_dim
            )  # (B, P, n_steps, latent_dim)
        )

        return z_paths

    # ------------------------------------------------------------------
    # KL divergence (Girsanov)
    # ------------------------------------------------------------------

    def kl_divergence(
        self,
        z_paths: Tensor,
        context: Tensor,
        exog: Tensor,
    ) -> Tensor:
        """Estimate the Girsanov KL  KL(q ‖ p)  over the sampled trajectories.

        Uses the discrete approximation:

            KL ≈ (dt / 2) · Σ_t E_q[ ‖u_t‖² ]

        where  u_t = (f_posterior(t, z_t) − f_prior(t, z_t)) / (g(t, z_t) + ε)

        Parameters
        ----------
        z_paths : Tensor
            Sampled trajectories, shape ``(batch, n_paths, n_steps, latent_dim)``.
        context : Tensor
            Encoder context, shape ``(batch, horizon, context_dim)``.
        exog : Tensor
            Future exogenous features, shape ``(batch, horizon, exog_dim)``.

        Returns
        -------
        Tensor
            Scalar KL estimate.
        """
        batch, n_paths, n_steps, latent_dim = z_paths.shape
        device = z_paths.device
        dtype = z_paths.dtype
        eps = 1e-5

        # ------------------------------------------------------------------
        # Expand context/exog to (B*P, H, D) for consistent f/g evaluation
        # ------------------------------------------------------------------
        ctx_expanded = (
            context.unsqueeze(1)
            .expand(batch, n_paths, context.shape[1], self.context_dim)
            .reshape(batch * n_paths, context.shape[1], self.context_dim)
            .contiguous()
        )
        exog_expanded = (
            exog.unsqueeze(1)
            .expand(batch, n_paths, exog.shape[1], self.exog_dim)
            .reshape(batch * n_paths, exog.shape[1], self.exog_dim)
            .contiguous()
        )
        self.set_trajectory(ctx_expanded, exog_expanded)

        # Flatten batch and path dims for vectorised evaluation
        # z_flat: (B*P, n_steps, latent_dim)
        z_flat = z_paths.reshape(batch * n_paths, n_steps, latent_dim)

        kl_sum = torch.zeros(1, device=device, dtype=dtype)

        for step_idx in range(n_steps):
            # Time corresponding to this step (hours, 1-indexed since we
            # dropped t=0 from z_paths)
            t_val = torch.tensor(float(step_idx + 1), device=device, dtype=dtype)

            z_t = z_flat[:, step_idx, :]  # (B*P, latent_dim)

            f_post = self.f(t_val, z_t)  # (B*P, latent_dim)
            f_prior = self.f_prior(t_val, z_t)  # (B*P, latent_dim)
            g_t = self.g(t_val, z_t)  # (B*P, latent_dim)

            # Control variate u_t = (f_post - f_prior) / (g_t + eps)
            u_t = (f_post - f_prior) / (g_t + eps)  # (B*P, latent_dim)

            # ‖u_t‖² summed over latent dims, then averaged over B*P
            kl_sum = kl_sum + u_t.pow(2).sum(dim=-1).mean()

        # Multiply by dt/2 (trapezoidal / Euler–Maruyama quadrature weight)
        kl = (self.dt / 2.0) * kl_sum.squeeze()

        return kl

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        z0_mean: Tensor,
        z0_logvar: Tensor,
        context: Tensor,
        exog: Tensor,
        n_paths: int = 16,
    ) -> Dict[str, Tensor]:
        """Full forward pass: reparameterise z0, integrate SDE, estimate KL.

        Parameters
        ----------
        z0_mean : Tensor
            Posterior mean of the initial condition, shape
            ``(batch, latent_dim)``.
        z0_logvar : Tensor
            Posterior log-variance of the initial condition, shape
            ``(batch, latent_dim)``.
        context : Tensor
            Encoder context trajectory, shape
            ``(batch, horizon, context_dim)``.
        exog : Tensor
            Future exogenous features, shape ``(batch, horizon, exog_dim)``.
        n_paths : int
            Number of Monte Carlo sample paths.  Default 16.

        Returns
        -------
        dict with keys:
            ``'z_paths'`` : ``(batch, n_paths, n_steps, latent_dim)``
            ``'kl'``      : scalar Tensor — Girsanov KL estimate
            ``'z0'``      : ``(batch, latent_dim)`` — the reparameterised z0
        """
        # ------------------------------------------------------------------
        # Reparameterise z0 ~ q(z_0 | x)
        # ------------------------------------------------------------------
        eps = torch.randn_like(z0_mean)
        z0 = z0_mean + eps * torch.exp(0.5 * z0_logvar)  # (batch, latent_dim)

        # ------------------------------------------------------------------
        # Sample posterior trajectories
        # ------------------------------------------------------------------
        n_steps = context.shape[1]  # use horizon length from context
        z_paths = self.sample_posterior(
            z0=z0,
            context=context,
            exog=exog,
            n_steps=n_steps,
            n_paths=n_paths,
        )  # (batch, n_paths, n_steps, latent_dim)

        # ------------------------------------------------------------------
        # Estimate KL divergence
        # ------------------------------------------------------------------
        kl = self.kl_divergence(z_paths, context, exog)

        return {
            "z_paths": z_paths,  # (B, P, T, latent_dim)
            "kl": kl,  # scalar
            "z0": z0,  # (B, latent_dim)
        }


# =============================================================================
# __main__ — smoke test
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LatentSDE smoke test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    BATCH = 2
    N_PATHS = 4
    HORIZON = 24
    LATENT = 3
    CONTEXT = 128
    EXOG = 6
    HIDDEN = 64

    # Instantiate model
    model = LatentSDE(
        latent_dim=LATENT,
        context_dim=CONTEXT,
        exog_dim=EXOG,
        hidden_dim=HIDDEN,
        dt=0.5,
        noise_type="diagonal",
        sde_type="stratonovich",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    # Create synthetic inputs
    z0_mean = torch.randn(BATCH, LATENT, device=device)
    z0_logvar = torch.zeros(BATCH, LATENT, device=device)
    context = torch.randn(BATCH, HORIZON, CONTEXT, device=device)
    exog = torch.randn(BATCH, HORIZON, EXOG, device=device)

    print("\nRunning forward pass …")
    out = model.forward(
        z0_mean=z0_mean,
        z0_logvar=z0_logvar,
        context=context,
        exog=exog,
        n_paths=N_PATHS,
    )

    z_paths = out["z_paths"]
    kl = out["kl"]
    z0 = out["z0"]

    # ------------------------------------------------------------------
    # Test 1: z_paths shape
    # ------------------------------------------------------------------
    expected_shape = (BATCH, N_PATHS, HORIZON, LATENT)
    assert z_paths.shape == expected_shape, (
        f"FAIL  z_paths shape: expected {expected_shape}, got {z_paths.shape}"
    )
    print(f"PASS  z_paths.shape == {tuple(z_paths.shape)}")

    # ------------------------------------------------------------------
    # Test 2: kl is a positive scalar
    # ------------------------------------------------------------------
    assert kl.ndim == 0, f"FAIL  kl should be a scalar, got shape {kl.shape}"
    assert kl.item() > 0.0, f"FAIL  kl should be positive, got {kl.item():.6f}"
    print(f"PASS  kl is positive scalar: {kl.item():.6f}")

    # ------------------------------------------------------------------
    # Test 3: backward pass from kl
    # ------------------------------------------------------------------
    kl.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert len(grad_norms) > 0, "FAIL  No gradients computed in backward pass"
    print(
        f"PASS  Backward pass OK — {len(grad_norms)} parameter tensors received gradients"
    )
    for name, gnorm in grad_norms.items():
        print(f"        grad norm  {name}: {gnorm:.4f}")

    print("\nz0 shape:", z0.shape)
    print(
        "z_paths stats — min: {:.3f}  max: {:.3f}  mean: {:.3f}".format(
            z_paths.min().item(), z_paths.max().item(), z_paths.mean().item()
        )
    )

    print("\nAll tests passed.")
    sys.exit(0)
