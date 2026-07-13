#!/usr/bin/env python
# train_lfgpnrf.py — LF-GP-NRF phased training
# Usage: python src/experiments/train_lfgpnrf.py [--fast_dev_run] [--phase PHASE]
#
# Implements the four-phase training schedule from §5.2 of the Research Plan:
#   Phase 0 — warm-up:      encoder + GP only   (8 epochs,  lr=5e-3)
#   Phase 1 — SDE unlock:   KL annealing 0→1    (12 epochs, lr=1e-3)
#   Phase 2 — flow unlock:  dual LR             (15 epochs, lr=5e-4/1e-4)
#   Phase 3 — joint tuning: cosine annealing    (30 epochs, lr=1e-4)
#
# Run from the epf_clean_restart/ root:
#   python src/experiments/train_lfgpnrf.py
#   python src/experiments/train_lfgpnrf.py --phase 2
#   python src/experiments/train_lfgpnrf.py --fast_dev_run

import argparse
import glob
import json
import os
import pickle
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# ---------------------------------------------------------------------------
# Make the repo root importable regardless of CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset import build_dataloaders
from src.models.lf_gp_nrf.model import LFGPNRFModel

# ===========================================================================
# Custom callbacks
# ===========================================================================


class KLAnnealingCallback(pl.Callback):
    """Linearly anneals ``model.kl_beta`` from 0 to 1.

    The anneal runs over ``warmup_epochs`` epochs, starting at epoch
    ``start_epoch``.  Before ``start_epoch`` the beta is kept at 0.0
    (free-bits / no KL pressure); after ``start_epoch + warmup_epochs`` it
    is clamped to 1.0 (full KL weight).

    This mirrors the β-VAE style annealing recommended in §5.3 of the
    Research Plan to prevent posterior collapse of z_t during Phase 1.

    Parameters
    ----------
    start_epoch : int
        Epoch at which the linear ramp begins.  Default 0.
    warmup_epochs : int
        Number of epochs over which the ramp climbs from 0 to 1.  Default 6.
    """

    def __init__(self, start_epoch: int = 0, warmup_epochs: int = 6) -> None:
        super().__init__()
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            pl_module.kl_beta = 0.0
        elif epoch < self.start_epoch + self.warmup_epochs:
            pl_module.kl_beta = (epoch - self.start_epoch) / self.warmup_epochs
        else:
            pl_module.kl_beta = 1.0
        pl_module.log("kl_beta", pl_module.kl_beta, prog_bar=True)


class FreezeModulesCallback(pl.Callback):
    """Freezes and unfreezes named sub-modules at scheduled epochs.

    The ``schedule`` dict maps an epoch number to a dict with optional
    ``'freeze'`` and ``'unfreeze'`` keys, each containing a list of
    attribute names on the ``pl_module``.

    Example schedule::

        {
            0:  {'freeze':   ['sde', 'flow']},
            8:  {'unfreeze': ['sde']},
            20: {'unfreeze': ['flow']},
        }

    Freezing sets ``requires_grad=False`` on all parameters of the named
    module; unfreezing restores ``requires_grad=True``.  The action fires
    at the *start* of the specified epoch, before any forward pass.

    Parameters
    ----------
    schedule : dict
        Mapping of ``{epoch_int: {'freeze': [...], 'unfreeze': [...]}}``
    """

    def __init__(self, schedule: dict | None = None) -> None:
        super().__init__()
        self.schedule: dict = schedule or {}

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if epoch not in self.schedule:
            return

        for name in self.schedule[epoch].get("freeze", []):
            module = getattr(pl_module, name)
            for p in module.parameters():
                p.requires_grad_(False)
            print(f'[FreezeModulesCallback] Epoch {epoch}: froze "{name}"')

        for name in self.schedule[epoch].get("unfreeze", []):
            module = getattr(pl_module, name)
            for p in module.parameters():
                p.requires_grad_(True)
            print(f'[FreezeModulesCallback] Epoch {epoch}: unfroze "{name}"')


# ===========================================================================
# Phase configuration
# ===========================================================================


def get_phase_config(phase: int) -> dict:
    """Return the training hyper-parameters for a given training phase.

    Phase 0 — Warm-up (8 epochs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Freeze the SDE and flow; only the BiLSTM encoder and GP train.
    This pre-conditions the GP to capture seasonal structure *before* the
    Neural SDE is introduced, preventing the SDE from immediately
    overwhelming the GP's kernel structure.

    KL weight is zero (SDE is frozen so the KL is undefined anyway).
    High LR (5e-3) with batch size 32 for fast convergence on the easy
    encoder objective.

    Phase 1 — SDE unlock (12 epochs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Unfreeze the Neural SDE.  Anneal the KL term from 0 → 1 over the
    first 6 epochs to prevent posterior collapse of z_t (§5.3 Scenario A).
    Reduced LR (1e-3) for more careful optimisation of the joint objective.

    Phase 2 — Flow unlock (15 epochs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Unfreeze the normalising flow.  The SDE and GP should by now have a
    stable latent representation; the flow learns the residual
    non-Gaussianity.  Dual LR: 5e-4 for flow parameters, 1e-4 for the
    rest (encoded in the LightningModule's ``configure_optimizers`` when
    ``lr_flow`` is present).

    Phase 3 — Joint fine-tuning (30 epochs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All modules active.  LR=1e-4 with cosine annealing (T_max=30,
    η_min=1e-6) as recommended in §5.2.  The calibration penalty
    (``calib_gamma``) is activated at this phase.

    Parameters
    ----------
    phase : int
        Training phase index in {0, 1, 2, 3}.

    Returns
    -------
    dict
        Training configuration consumed by ``train_phase`` and
        ``build_callbacks``.

    Raises
    ------
    ValueError
        If ``phase`` is not in {0, 1, 2, 3}.
    """
    configs = {
        # ------------------------------------------------------------------
        # Phase 0: warm-up — encoder + GP only
        # ------------------------------------------------------------------
        0: {
            "epochs": 8,
            "lr": 5e-3,
            "batch_size": 32,
            "kl_beta": 0.0,
            "kl_start": 0,
            "kl_warmup": 0,  # no ramp; stays at 0
            "calib_active": False,
            "scheduler": "none",
            # At epoch 0: freeze sde and flow before first batch
            "freeze_schedule": {
                0: {"freeze": ["sde", "flow"]},
            },
            "description": (
                "Warm-up: BiLSTM encoder + GP only.  SDE and flow frozen. "
                "Teaches the GP to capture seasonal structure."
            ),
        },
        # ------------------------------------------------------------------
        # Phase 1: SDE unlock + KL annealing
        # ------------------------------------------------------------------
        1: {
            "epochs": 12,
            "lr": 1e-3,
            "batch_size": 32,
            "kl_beta": 0.0,  # starts at 0, ramped by callback
            "kl_start": 0,
            "kl_warmup": 6,  # anneal over first 6 epochs of this phase
            "calib_active": False,
            "scheduler": "none",
            # At epoch 0 of this phase: unfreeze sde; flow stays frozen
            "freeze_schedule": {
                0: {"unfreeze": ["sde"]},
            },
            "description": (
                "SDE unlock: Neural SDE unfrozen.  KL term annealed 0→1 "
                "over 6 epochs to prevent posterior collapse."
            ),
        },
        # ------------------------------------------------------------------
        # Phase 2: flow unlock — dual learning rate
        # ------------------------------------------------------------------
        2: {
            "epochs": 40,
            "lr": 1e-4,  # GP / SDE / encoder LR
            "lr_flow": 5e-4,  # flow-specific LR (higher — fresh start)
            "batch_size": 32,
            "kl_beta": 1.0,  # KL fully on
            "kl_start": 0,
            "kl_warmup": 0,
            "calib_active": False,
            "scheduler": "none",
            # At epoch 0 of this phase: unfreeze flow
            "freeze_schedule": {
                0: {"unfreeze": ["flow"]},
            },
            "description": (
                "Flow unlock: normalising flow unfrozen with higher LR (5e-4). "
                "GP/SDE/encoder train at 1e-4.  40 epochs so the flow has "
                "enough time to learn regime-conditioned tail shapes."
            ),
        },
        # ------------------------------------------------------------------
        # Phase 3: joint fine-tuning
        # ------------------------------------------------------------------
        3: {
            "epochs": 50,
            "lr": 1e-4,
            "batch_size": 32,
            "kl_beta": 1.0,
            "kl_start": 0,
            "kl_warmup": 0,
            "calib_active": True,  # calibration penalty active
            "scheduler": "cosine",
            "cosine_t_max": 50,
            "cosine_eta_min": 1e-6,
            "freeze_schedule": {},  # nothing to freeze/unfreeze
            "description": (
                "Joint fine-tuning: all modules active, cosine annealing, "
                "calibration penalty on.  Gradient clipping max_norm=1.0.  "
                "50 epochs with cosine decay from 1e-4 to 1e-6."
            ),
        },
    }

    if phase not in configs:
        raise ValueError(f"phase must be in {{0, 1, 2, 3}}, got {phase}.")
    return configs[phase]


# ===========================================================================
# Callback factory
# ===========================================================================


def build_callbacks(phase: int, out_dir: str, phase_config: dict) -> list:
    """Construct the callback list for a training phase.

    Callbacks included:
    - ``ModelCheckpoint`` — saves top-3 + last by ``val/loss``.
    - ``EarlyStopping``   — patience=8, min_delta=1e-4.
    - ``LearningRateMonitor`` — logs LR per epoch to TensorBoard.
    - ``TQDMProgressBar`` — lighter refresh (every 10 steps).
    - ``KLAnnealingCallback`` — ramps ``kl_beta`` linearly.
    - ``FreezeModulesCallback`` — applies the phase freeze schedule.

    Parameters
    ----------
    phase : int
        Current training phase (used in checkpoint filename).
    out_dir : str
        Root output directory (checkpoints written to ``out_dir/checkpoints``).
    phase_config : dict
        Config dict from ``get_phase_config``.

    Returns
    -------
    list of pl.Callback
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Each phase gets its own sub-directory so last.ckpt filenames never
    # collide across phases and the cross-phase handoff glob is unambiguous.
    phase_ckpt_dir = os.path.join(ckpt_dir, f"phase{phase}")
    os.makedirs(phase_ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=phase_ckpt_dir,
        filename=f"lfgpnrf-phase{phase}-{{epoch:02d}}-{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Patience scales with phase length: short phases need tighter patience,
    # long phases (2/3) need room to escape local plateaus.
    patience_by_phase = {0: 6, 1: 8, 2: 15, 3: 20}
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=patience_by_phase.get(phase, 12),
        mode="min",
        min_delta=1e-4,
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    progress_cb = TQDMProgressBar(refresh_rate=10)

    kl_anneal_cb = KLAnnealingCallback(
        start_epoch=phase_config.get("kl_start", 0),
        warmup_epochs=phase_config.get("kl_warmup", 6),
    )

    freeze_cb = FreezeModulesCallback(
        schedule=phase_config.get("freeze_schedule", {}),
    )

    return [
        checkpoint_cb,
        early_stop_cb,
        lr_monitor_cb,
        progress_cb,
        kl_anneal_cb,
        freeze_cb,
    ]


# ===========================================================================
# Single-phase training
# ===========================================================================


def train_phase(
    phase: int,
    model: LFGPNRFModel,
    train_loader,
    val_loader,
    phase_config: dict,
    out_dir: str,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """Train one phase of the LF-GP-NRF model.

    Steps
    -----
    1. Build a ``TensorBoardLogger`` namespaced to this phase.
    2. Construct a ``pl.Trainer`` with AMP (``precision='16-mixed'``),
       gradient clipping, and the phase callbacks.
    3. If ``phase > 0``, attempt to load the *last* checkpoint from the
       previous phase so that training resumes from the correct weights.
    4. Call ``trainer.fit``.
    5. Save the final state-dict to
       ``out_dir/checkpoints/lfgpnrf_phase{phase}_final.pt``.

    Parameters
    ----------
    phase : int
    model : LFGPNRFModel
    train_loader : DataLoader
    val_loader : DataLoader
    phase_config : dict
        From ``get_phase_config(phase)``.
    out_dir : str
    fast_dev_run : bool
        If ``True``, run only 1 batch per split (sanity check).

    Returns
    -------
    pl.Trainer
        Fitted trainer (useful for inspecting logged metrics in tests).
    """
    # ------------------------------------------------------------------
    # 1. Logger
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir=os.path.join(out_dir, "logs"),
        name=f"phase{phase}",
    )

    # ------------------------------------------------------------------
    # 2. Trainer
    #
    # precision='32-true' uses full float32.  GPyTorch's variational strategy
    # backward pass mixes float32 Cholesky factors with the AMP float16 graph,
    # producing a dtype mismatch RuntimeError.  Float32 is the safe default
    # until GPyTorch adds native AMP support.  On the RTX 4000 the throughput
    # difference is negligible for this 777 K-parameter model.
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=phase_config["epochs"],
        accelerator="gpu",
        devices=1,
        precision="32-true",  # float32; GPyTorch backward is AMP-incompatible
        gradient_clip_val=1.0,  # prevent SDE diffusion blow-up (§5.2)
        log_every_n_steps=10,
        callbacks=build_callbacks(phase, out_dir, phase_config),
        logger=logger,
        fast_dev_run=fast_dev_run,
        enable_model_summary=(phase == 0),  # print summary only on first phase
    )

    # ------------------------------------------------------------------
    # 3. Resume from previous phase's last checkpoint
    # ------------------------------------------------------------------
    ckpt_path = None
    if phase > 0:
        # Each phase saves its last.ckpt into its own sub-directory
        # checkpoints/phase{N}/last.ckpt — look there first.
        prev_phase_dir = os.path.join(out_dir, "checkpoints", f"phase{phase - 1}")
        preferred = os.path.join(prev_phase_dir, "last.ckpt")
        if os.path.exists(preferred):
            ckpt_path = preferred
            print(f"[train_phase] Phase {phase}: resuming from {ckpt_path}")
        else:
            # Fallback 1: any last*.ckpt in the previous phase sub-dir
            sub_matches = glob.glob(os.path.join(prev_phase_dir, "last*.ckpt"))
            if sub_matches:
                ckpt_path = max(sub_matches, key=os.path.getmtime)
                print(f"[train_phase] Phase {phase}: resuming from {ckpt_path}")
            else:
                # Fallback 2: scan flat checkpoints/ for any last.ckpt
                flat_matches = glob.glob(
                    os.path.join(out_dir, "checkpoints", "**", "last*.ckpt"),
                    recursive=True,
                )
                if flat_matches:
                    ckpt_path = max(flat_matches, key=os.path.getmtime)
                    print(
                        f"[train_phase] Phase {phase}: fallback checkpoint → {ckpt_path}"
                    )
                else:
                    print(
                        f"[train_phase] Phase {phase}: no checkpoint found; "
                        f"starting from current model weights."
                    )

        # Safety net: ensure kl_beta is correct for this phase regardless
        # of what value was stored in the loaded checkpoint.
        _persist_kl_beta(model, phase)

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # ------------------------------------------------------------------
    # 5. Save final state-dict for downstream evaluation / ablation
    # ------------------------------------------------------------------
    if not fast_dev_run:
        # Save to both the phase sub-dir AND the flat checkpoints/ root so
        # downstream scripts can always find lfgpnrf_phase{N}_final.pt in one place.
        final_pt_path = os.path.join(
            out_dir, "checkpoints", f"lfgpnrf_phase{phase}_final.pt"
        )
        torch.save(model.state_dict(), final_pt_path)
        print(f"[train_phase] Saved final state-dict → {final_pt_path}")

    return trainer


# ===========================================================================
# Main entry-point
# ===========================================================================


def _persist_kl_beta(model: LFGPNRFModel, phase: int) -> None:
    """Ensure kl_beta stays at 1.0 when loading weights for phases 1+.

    Lightning checkpoints restore the full model state, including kl_beta
    which is a plain float attribute (not a Parameter).  When phase 1's
    KL annealing ramps to 1.0, saving and reloading the checkpoint correctly
    preserves that value.  This function is a safety net for the edge case
    where an older checkpoint has kl_beta=0.0 and we're starting phase 2+.
    """
    if phase >= 2:
        model.kl_beta = 1.0


def main(args: argparse.Namespace) -> None:
    """End-to-end phased training of LF-GP-NRF.

    Flow
    ----
    1. Build train / val / test ``DataLoader`` s via ``build_dataloaders``.
    2. Persist the fitted scalers to ``outputs/scalers.pkl`` for later use
       in the evaluation / benchmark scripts.
    3. Infer ``history_feat_dim`` and ``future_feat_dim`` from the first
       training batch (avoids hard-coding feature counts).
    4. Instantiate ``LFGPNRFModel`` with the inferred dimensions.
    5. Loop over phases ``[args.phase, 4)`` calling ``train_phase`` for each.
    """
    DATA_DIR = "data/raw"
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build data loaders
    # ------------------------------------------------------------------
    print("Building data loaders...")
    train_loader, val_loader, test_loader, scalers = build_dataloaders(
        data_path=os.path.join(DATA_DIR, "Germany_master_entsoe_2015_2026.csv"),
        comm_path=os.path.join(DATA_DIR, "commodities.csv"),
        flow_path=os.path.join(DATA_DIR, "cross_border_flows.csv"),
        train_end="2022-12-31",
        val_end="2023-12-31",
        test_end="2024-12-31",
        batch_size=32,
        num_workers=0,
    )
    print(
        f"Splits ready — "
        f"train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}, "
        f"test batches: {len(test_loader)}"
    )

    # ------------------------------------------------------------------
    # 2. Persist scalers
    # ------------------------------------------------------------------
    scalers_path = os.path.join(OUT_DIR, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved → {scalers_path}")

    # ------------------------------------------------------------------
    # 3. Infer feature dimensions from first training batch
    # ------------------------------------------------------------------
    sample_batch = next(iter(train_loader))
    history_feat_dim = sample_batch["history"].shape[-1]
    future_feat_dim = sample_batch["future_exog"].shape[-1]
    print(f"history_feat_dim={history_feat_dim}, future_feat_dim={future_feat_dim}")

    # Persist dimension info alongside the scalers for reproducibility
    dims_path = os.path.join(OUT_DIR, "feature_dims.json")
    with open(dims_path, "w") as f:
        json.dump(
            {"history_feat_dim": history_feat_dim, "future_feat_dim": future_feat_dim},
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 4. Instantiate model
    #
    # Hyper-parameters follow §4.3 of the Research Plan (Table 4.1).
    # kl_beta=0.0 — Phase 0 starts with no KL pressure; KLAnnealingCallback
    #               ramps it up during Phase 1.
    # calib_gamma=0.1 — Calibration penalty weight; activated in Phase 3 by
    #                   the model's configure_optimizers / training_step.
    # ------------------------------------------------------------------
    print("\nInstantiating LFGPNRFModel...")
    model = LFGPNRFModel(
        history_feat_dim=history_feat_dim,
        future_feat_dim=future_feat_dim,
        latent_dim=3,
        encoder_hidden=128,
        sde_hidden=64,
        n_inducing=256,
        n_flow_transforms=4,
        n_flow_bins=8,
        n_paths_train=8,
        n_paths_infer=64,
        n_samples_infer=100,
        lr=1e-3,
        weight_decay=1e-4,
        kl_beta=0.0,
        gp_beta=1.0,
        calib_gamma=0.1,
    )

    # ------------------------------------------------------------------
    # 5. Phased training loop
    # ------------------------------------------------------------------
    start_phase = args.phase
    for phase in range(start_phase, 4):
        print(f"\n{'=' * 60}")
        print(f"  TRAINING PHASE {phase}")
        print(f"{'=' * 60}")

        phase_config = get_phase_config(phase)
        print(f"  {phase_config['description']}")
        print(
            f"  epochs={phase_config['epochs']}, "
            f"lr={phase_config['lr']}, "
            f"kl_warmup={phase_config.get('kl_warmup', 0)}"
        )

        # Apply phase-level kl_beta to the model before training starts
        # (the KLAnnealingCallback will override this each epoch, but
        # setting it here ensures the correct value for Phase 2/3 where
        # kl_beta should begin at 1.0)
        model.kl_beta = phase_config["kl_beta"]

        # Activate / deactivate calibration penalty
        if hasattr(model, "calib_active"):
            model.calib_active = phase_config.get("calib_active", False)

        trainer = train_phase(
            phase=phase,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            phase_config=phase_config,
            out_dir=OUT_DIR,
            fast_dev_run=args.fast_dev_run,
        )

        if args.fast_dev_run:
            print("fast_dev_run complete — stopping after first phase.")
            break

    print("\nTraining complete.")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LF-GP-NRF phased training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help=(
            "Run only 1 batch per split per phase — for CI / smoke testing. "
            "No checkpoints are written in this mode."
        ),
    )

    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            "Start from this phase.  Useful for resuming interrupted training. "
            "Phase 0=warm-up, 1=SDE unlock, 2=flow unlock, 3=joint fine-tune."
        ),
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
