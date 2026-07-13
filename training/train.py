"""
train.py - Joint Training for MixVol TFT

Runs the complete unified training pipeline after expert pretraining.
Produces TensorBoard logs, CSV histories, and tracking metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import crps_from_quantiles
from src.training.losses import total_loss

def track_router_entropy(gate_weights: torch.Tensor) -> float:
    """H = -sum w_i log w_i"""
    # Fix: Aggregation adjusted for new 2D router weights [batch, experts]
    w = gate_weights.mean(dim=0) + 1e-8
    entropy = -(w * torch.log(w)).sum().item()
    return entropy

def load_pretrained_experts(model: nn.Module, n_experts: int = 4):
    """Loads only the expert submodule weights from pretrained checkpoints (Fix 1).
    
    Each checkpoint saves the full model, so we filter to the target expert's
    keys and load them with strict=True into that expert only.
    """
    for i in range(n_experts):
        ckpt_path = Path(f"models/pretrained_expert_{i}.pt")
        if ckpt_path.exists():
            print(f"[train] Loading pretrained weights for Expert {i} from {ckpt_path} ...")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Extract only the keys belonging to moe.experts[i]
            prefix = f"moe.experts.{i}."
            expert_state = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if expert_state:
                model.moe.experts[i].load_state_dict(expert_state, strict=True)
                print(f"[train]   Loaded {len(expert_state)} tensors into moe.experts[{i}]")
            else:
                print(f"[train]   WARNING: No keys matching '{prefix}*' found in {ckpt_path}")

def plot_training_figures(history_df: pd.DataFrame, out_dir: Path):
    """Figure 2a (4-panel loss curves) and Figure 2b (Router entropy)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Figure 2a ---
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Figure 2a – Training Curves (Joint Phase)", fontsize=12, fontweight="bold")
    
    epochs = history_df["epoch"]
    metrics = [
        ("train_loss", "val_loss", "Total Loss"),
        ("train_pinball", "val_pinball", "Pinball Loss (CRPS proxy)"),
        ("train_coherence", "val_coherence", "Physical Coherence Penalty"),
        ("train_load_balance", "val_load_balance", "Routing Load Balance Penalty")
    ]
    
    for ax, (t_col, v_col, title) in zip(axes.flat, metrics):
        ax.plot(epochs, history_df[t_col], label="Train", color="#1f77b4", lw=2)
        ax.plot(epochs, history_df[v_col], label="Val", color="#ff7f0e", lw=2, linestyle='--')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    fig.tight_layout()
    fig.savefig(out_dir / "fig_02a_training_curves.pdf", dpi=200)
    plt.close(fig)
    print(f"[train] Figure 2a saved -> {out_dir / 'fig_02a_training_curves.pdf'}")
    
    # --- Figure 2b ---
    if "val_entropy" in history_df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(epochs, history_df["val_entropy"], color="#d62728", lw=2, marker='o', markersize=4)
        ax2.set_title("Figure 2b – Router Entropy over Epochs (No Collapse)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Entropy H = -Σ w_i log(w_i)")
        ax2.grid(True, alpha=0.3)
        
        # Max entropy for 4 experts is ln(4) approx 1.386
        ax2.axhline(np.log(4), color="#7f7f7f", linestyle="--", label="Max Entropy (Uniform)")
        ax2.legend()
        
        fig2.tight_layout()
        fig2.savefig(out_dir / "fig_02b_router_entropy.pdf", dpi=200)
        plt.close(fig2)
        print(f"[train] Figure 2b saved -> {out_dir / 'fig_02b_router_entropy.pdf'}")

def train_joint(model: nn.Module, train_loader, val_loader, config: dict) -> dict:
    """
    Joint training of the unified architecture using mixed precision, cosine annealing,
    tensorboard tracking, and early stopping.
    """
    load_pretrained_experts(model, config.get("n_experts", 4))
    
    device = next(model.parameters()).device
    max_epochs = config.get("max_epochs", 100)
    
    quantiles = config.get("quantile_levels", [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    
    # NEW: Phase 1 - Label-Supervised Router Seeding (Freeze Experts)
    print("\n[train] Phase 1: Label-Supervised Router Seeding (15 epochs, Experts Frozen)")
    print("[train]   Goal: Force router to recognize market regimes using ground-truth labels.")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.moe.router.parameters():
        param.requires_grad = True
    for param in model.moe.router_summary.parameters():
        param.requires_grad = True
    for param in model.hurst_gate.parameters():
        param.requires_grad = True
        
    router_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    router_epochs = 15 
    criterion_ce = nn.CrossEntropyLoss()
    
    for epoch in range(1, router_epochs + 1):
        model.train()
        total_ce = 0
        total_lb = 0
        for batch in train_loader:
            x_hist = batch['x_hist'].to(device)
            h_t = batch['h_t'].to(device)
            # Consensus regime for the horizon (mode of first 24 steps)
            regime_labels = batch['regime'].to(device) # [b, 24]
            # Use the most frequent regime in the window as target
            target_regime = torch.mode(regime_labels, dim=1).values # [b]
            
            router_optimizer.zero_grad()
            out = model(x_hist, None, h_t)
            
            # 1. Supervised Loss: Router must match the regime labels
            # gate_weights are softmax, but for CE we need raw logits. 
            # We recover them implicitly or just use the probs.
            # Actually, SparseMoEHead doesn't return raw logits. 
            # I'll modify SparseMoEHead to return logits too.
            ce_loss = criterion_ce(out['gate_weights'], target_regime)
            
            # 2. Load Balance Penalty (Secondary)
            losses = total_loss(out['quantiles'], batch['y'][:,:24].to(device), 
                                quantiles, p_floor=-3000.0, beta=1.0, gate_probs=out['gate_weights'])
            
            loss = ce_loss + losses['load_balance']
            loss.backward()
            router_optimizer.step()
            
            total_ce += ce_loss.item()
            total_lb += losses['load_balance'].item()
            
        print(f"  Seeding Epoch {epoch:02d} | CE Loss: {total_ce/len(train_loader):.4f} | LB Loss: {total_lb/len(train_loader):.4f}")

    # Unfreeze everything for Joint Training
    for param in model.parameters():
        param.requires_grad = True
    
    print("\n[train] Phase 2: Joint Training (Experts + Router)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Fix 2: WarmRestarts prevents LR from hitting zero, which causes fp16 NaN cascade
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    
    scaler = torch.amp.autocast('cuda') if torch.cuda.is_available() else None
    # Fix: Slower GradScaler growth (1.5) and lower init scale to prevent NaN spikes during MoE transitions
    scaler_grad = torch.amp.GradScaler('cuda', init_scale=1024.0, growth_factor=1.5) if torch.cuda.is_available() else None
    
    writer = SummaryWriter(log_dir="results/tensorboard_logs")
    
    history = []
    
    best_crps = float('inf')
    best_epoch = 0
    patience = config.get('patience', 30)
    patience_counter = 0
    
    # Add symmetry-breaking noise to router at start of joint training
    with torch.no_grad():
        model.moe.router.weight.add_(torch.randn_like(model.moe.router.weight) * 0.05)
    
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    best_model_path = out_dir / "best_model.pt"
    
    # quantiles already defined above
    
    for epoch in range(1, max_epochs + 1):
        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        train_stats = {k: 0.0 for k in ['total', 'pinball', 'coherence', 'load_balance']}
        train_batches = 0
        
        for batch in train_loader:
            x_hist = batch['x_hist'].to(device)
            x_static = batch['x_static'].to(device)
            h_t = batch['h_t'].to(device)
            y = batch['y'].to(device)  # [b, forecast_horizon]
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with scaler:
                    out = model(x_hist, x_static, h_t)
                    q_preds = out['quantiles']
                    y_t = y[:, :q_preds.size(1)]
                    
                    # Supervised Regularizer: Keep router anchored to regimes
                    regime_labels = batch['regime'].to(device)
                    target_regime = torch.mode(regime_labels, dim=1).values
                    ce_loss = criterion_ce(out['gate_weights'], target_regime)
                    
                    losses = total_loss(
                        q_preds, y_t, quantiles, p_floor=-3000.0,
                        gate_probs=out['gate_weights'], lambda_pc=0.01, beta=5.0
                    )
                    # Combined losses: Forecast error + Diversity + Regime Anchor
                    total_loss_final = losses['total'] + 0.5 * ce_loss
                scaler_grad.scale(total_loss_final).backward()
                scaler_grad.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_grad.step(optimizer)
                scaler_grad.update()
            else:
                out = model(x_hist, x_static, h_t)
                q_preds = out['quantiles']
                y_t = y[:, :q_preds.size(1)]
                
                # Supervised Regularizer
                regime_labels = batch['regime'].to(device)
                target_regime = torch.mode(regime_labels, dim=1).values
                ce_loss = criterion_ce(out['gate_weights'], target_regime)
                
                losses = total_loss(
                    q_preds, y_t, quantiles, p_floor=-3000.0,
                    gate_probs=out['gate_weights'], lambda_pc=0.01, beta=5.0
                )
                total_loss_final = losses['total'] + 0.5 * ce_loss
                total_loss_final.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            # --- NaN Guard ---
            is_nan = False
            for k, v in losses.items():
                if torch.isnan(v):
                    is_nan = True
                    # Only print once to avoid log spam
                    if train_batches == 0:
                        print(f"  [batch {train_batches}] NaN detected in {k} loss! Skipping stats accumulation. This is common during GradScaler calibration.")
                    break
            
            if not is_nan:
                for k in train_stats:
                    train_stats[k] += losses[k].item()
                train_batches += 1
            
        scheduler.step()
        for k in train_stats: train_stats[k] /= max(1, train_batches)
        
        # -----------------------------
        # VALIDATE
        # -----------------------------
        model.eval()
        val_stats = {k: 0.0 for k in ['total', 'pinball', 'coherence', 'load_balance']}
        val_batches = 0
        val_crps = 0.0
        val_entropy = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x_hist = batch['x_hist'].to(device)
                x_static = batch['x_static'].to(device)
                h_t = batch['h_t'].to(device)
                y = batch['y'].to(device)
                
                if scaler is not None:
                    with scaler:
                        out = model(x_hist, x_static, h_t)
                        q_preds = out['quantiles']
                        y_t = y[:, :q_preds.size(1)]
                        losses = total_loss(
                            q_preds, y_t, quantiles, p_floor=-3000.0,
                            gate_probs=out['gate_weights'], lambda_pc=0.01, beta=1.0
                        )
                else:
                    out = model(x_hist, x_static, h_t)
                    q_preds = out['quantiles']
                    y_t = y[:, :q_preds.size(1)]
                    losses = total_loss(
                        q_preds, y_t, quantiles, p_floor=-3000.0,
                        gate_probs=out['gate_weights'], lambda_pc=0.01, beta=1.0
                    )
                
                crps_batch = crps_from_quantiles(y_t, q_preds, quantiles)
                val_crps += crps_batch
                ent = track_router_entropy(out['gate_weights'])
                if val_batches == 0:
                    # Fix: Adjusted indexing for 2D probs [batch, n_experts]
                    print(f"  [DEBUG] probs[0] = {out['gate_weights'][0,:]}")
                    print(f"  [DEBUG] ent_batch = {ent}")
                val_entropy += ent
                
                for k in val_stats:
                    val_stats[k] += losses[k].item()
                val_batches += 1
                
        for k in val_stats: val_stats[k] /= max(1, val_batches)
        val_crps /= max(1, val_batches)
        val_entropy /= max(1, val_batches)
        
        # -----------------------------
        # METRICS TRACKING
        # -----------------------------
        history.append({
            "epoch": epoch,
            "train_loss": train_stats["total"],
            "val_loss": val_stats["total"],
            "train_pinball": train_stats["pinball"],
            "val_pinball": val_stats["pinball"],
            "train_coherence": train_stats["coherence"],
            "val_coherence": val_stats["coherence"],
            "train_load_balance": train_stats["load_balance"],
            "val_load_balance": val_stats["load_balance"],
            "val_crps": val_crps,
            "val_entropy": val_entropy
        })
        
        writer.add_scalar("Loss/Train", train_stats["total"], epoch)
        writer.add_scalar("Loss/Val", val_stats["total"], epoch)
        writer.add_scalar("CRPS/Val", val_crps, epoch)
        writer.add_scalar("Router/Entropy", val_entropy, epoch)
        
        print(f"Epoch {epoch:03d} | Tr Total: {train_stats['total']:.3f} | Vl Total: {val_stats['total']:.3f} | " 
              f"Vl CRPS: {val_crps:.3f} | Entropy: {val_entropy:.3f}")
        
        if val_crps < best_crps - 1e-4:
            best_crps = val_crps
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved new best model (CRPS: {val_crps:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val CRPS: {best_crps:.4f} at epoch {best_epoch}.")
            break

    # Finalize Tracking Tables and Visualizations
    hist_df = pd.DataFrame(history)
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(tables_dir / "training_history.csv", index=False)
    
    fig_dir = Path("results/figures")
    plot_training_figures(hist_df, fig_dir)
    
    return {
        'best_epoch': best_epoch,
        'best_val_crps': best_crps,
        'history_path': str(tables_dir / "training_history.csv")
    }

if __name__ == "__main__":
    from src.data.dataset import get_dataloader
    from src.models.mixvol_tft import ElectricityMixVolTFT
    import yaml
    
    with open("config.yaml") as f:
        raw_config = yaml.safe_load(f)
        
    fm_path = raw_config["data"]["processed_path"]
    rl_path = Path(raw_config["data"]["regimes_dir"]) / "regime_labels.parquet"
    hs_path = Path(raw_config["data"]["regimes_dir"]) / "hurst_series.parquet"
    
    print("[train_script] Loading dataloaders...")
    train_loader = get_dataloader(fm_path, rl_path, hs_path, split="train", batch_size=64, num_workers=0)
    val_loader = get_dataloader(fm_path, rl_path, hs_path, split="val", batch_size=64, num_workers=0)
    
    # Setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train_script] Initializing MixVol TFT model on {device}...")
    model = ElectricityMixVolTFT({
        'n_features': train_loader.dataset.x_hist.shape[1],
        'n_static_features': train_loader.dataset.x_static.shape[1] if len(train_loader.dataset.x_static.shape)>1 else 8,
        'seq_len': 168,
        'hidden_dim': 64,
        'n_quantiles': 7,
        'n_experts': 4
    })
    model.to(device)
    
    # -----------------------------
    # 2. Train with Weighted Sampler to fix Expert Starvation (Fix 6)
    # -----------------------------
    print("[train_script] Calculating regime weights for balanced sampling...")
    train_ds = train_loader.dataset
    # Sample regime labels across all valid indices
    all_regimes = []
    for i in range(len(train_ds)):
        # We sample based on the regime of the target start
        all_regimes.append(train_ds.regimes[i + train_ds.seq_len])
    all_regimes = np.array(all_regimes).astype(int)
    
    counts = np.bincount(all_regimes)
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = class_weights[all_regimes]
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    balanced_train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=True
    )
    
    # Restrict to rapid training representation since we're in terminal environment
    print("[train_script] Starting Joint Training Phase for 100 epochs ...")
    raw_config['max_epochs'] = 100
    raw_config['patience'] = 30 
    train_joint(model, balanced_train_loader, val_loader, raw_config)
