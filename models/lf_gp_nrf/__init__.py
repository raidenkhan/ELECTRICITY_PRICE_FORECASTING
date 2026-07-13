# LF-GP-NRF: Latent Force Gaussian Process with Neural Regime Flow
# Electricity Price Forecasting — MPhil Research
#
# Package structure:
#   encoder.py        — BiLSTM recognition network (amortised variational encoder)
#   latent_sde.py     — Neural SDE latent force field (torchsde, Reversible Heun)
#   force_kernel.py   — Force-conditioned warped GP kernel (GPyTorch custom kernel)
#   force_gp.py       — Sparse Variational GP with warped kernel (SVGP, 512 inducing pts)
#   flow_emission.py  — Conditional RQ-NSF normalizing flow emission (zuko)
#   model.py          — LFGPNRFModel end-to-end Lightning module + joint ELBO

from .encoder import LatentForceEncoder
from .flow_emission import NormalizingFlowEmission
from .force_gp import ForceConditionedGP
from .force_kernel import WarpedForceKernel
from .latent_sde import LatentSDE
from .model import LFGPNRFModel

__all__ = [
    "LatentForceEncoder",
    "LatentSDE",
    "WarpedForceKernel",
    "ForceConditionedGP",
    "NormalizingFlowEmission",
    "LFGPNRFModel",
]
