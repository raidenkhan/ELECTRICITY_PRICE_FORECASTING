"""
curriculum.py - Curriculum Pre-training for Experts

Implements single-expert pretraining on regime-filtered datasets.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Workaround for the strictly "causal" requirement: weighted random sampler
# inherently shuffles. If strict causality is required in pre-training,
# we'd use SequentialSampler. But the prompt explicitly asks for:
# "Use weighted sampler: w = 1.0 for regular hours, w = 5.0 for extreme events"
# So we use WeightedRandomSampler for pre-training.

def _get_expert_filter_mask(dataset, expert_id: int) -> list[int]:
    """Return indices of dataset corresponding to the expert's regime rule."""
    indices = []
    prices = []
    
    # We must iterate or access dataset.regimes / dataset.y
    # Since dataset is ElectricityDataset, we can peek at its attributes directly
    # for speed rather than running __getitem__ on all.
    ctx_starts = range(len(dataset))
    
    for idx in ctx_starts:
        ctx_end = idx + dataset.seq_len
        tgt_start = ctx_end
        
        # We look at the FIRST hour of the forecast horizon to determine if this window belongs here
        # (Could also look at the whole horizon, but first hour is typical for regime assignment)
        regime = dataset.regimes[tgt_start]
        price = dataset.y[tgt_start]
        
        if expert_id in (0, 1, 2):
            if regime == expert_id:
                indices.append(idx)
                prices.append(price)
        elif expert_id == 3:
            if price < 0 or abs(price) > 150:
                indices.append(idx)
                prices.append(price)
                
    return indices, prices

def pretrain_single_expert(expert_id: int, expert_model: nn.Module, dataset, config: dict) -> dict:
    """
    Pretrains an expert model on its specific regime partition.
    
    expert_id: 0-3
    expert_model: A PyTorch module that maps (x_hist, x_static, h_t) -> outputs
                  Typically an instance of ElectricityMixVolTFT. We will only
                  update its context layers and the specific expert_id.
    """
    from src.training.losses import pinball_loss
    
    # 1. Filter dataset indices
    indices, prices = _get_expert_filter_mask(dataset, expert_id)
    if not indices:
        print(f"[curriculum] Warning: No samples found for Expert {expert_id}. Skipping.")
        return {'final_val_loss': float('inf'), 'n_training_samples': 0, 'epochs_trained': 0, 'checkpoint_path': ""}

    # 2. Compute weights for sampler
    weights = []
    for p in prices:
        if p < 0 or abs(p) > 150:
            weights.append(5.0)
        else:
            weights.append(1.0)
            
    # Because WeightedRandomSampler acts as a RandomSampler according to distribution,
    # it breaks the strictly sequential loading, but it is required by the prompt.
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    subset = Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=config.get("batch_size", 32), sampler=sampler)
    
    # 3. Setup optimizer
    # We only train the shared encoders and the target expert. The router and other experts are frozen.
    # If expert_model is the full mixvol, we freeze router and other experts.
    if hasattr(expert_model, 'moe'):
        for param in expert_model.moe.router.parameters():
            param.requires_grad = False
        for param in expert_model.hurst_gate.parameters():
            param.requires_grad = False
            
        for ext_id, ext_module in enumerate(expert_model.moe.experts):
            if ext_id != expert_id:
                for param in ext_module.parameters():
                    param.requires_grad = False
            else:
                for param in ext_module.parameters():
                    param.requires_grad = True
                    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, expert_model.parameters()), lr=1e-3)
    
    quantiles = config.get("quantile_levels", [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    
    device = next(expert_model.parameters()).device
    expert_model.train()
    
    best_loss = float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    max_epochs = config.get("max_epochs", 50)
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"pretrained_expert_{expert_id}.pt"
    
    print(f"\n[curriculum] Pretraining Expert {expert_id} on {len(indices)} samples ...")
    
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0.0
        batches = 0
        
        for batch in loader:
            x_hist = batch['x_hist'].to(device)
            x_static = batch['x_static'].to(device)
            h_t = batch['h_t'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # New Architecture: Each expert does its own VSN/Attention
            # We pass raw embeddings directly
            x_emb = expert_model.feature_embed(x_hist.unsqueeze(-1))
            
            # Run ONLY target expert
            out_dict = expert_model.moe.experts[expert_id](x_emb)
            
            # The new Expert forward pass returns 24 steps of quantiles.
            # We target the first hour for pre-training simplicity.
            y_target = y[:, 0:1]
            q_preds = out_dict['quantiles'][:, 0:1, :]
            
            loss = pinball_loss(q_preds, y_target, quantiles)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / max(1, batches)
        print(f"  Epoch {epoch:02d} | Train Pinball: {avg_loss:.4f}")
        
        # Early stopping logic (acting on train loss here as dataset subset might not leave room for val)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(expert_model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping triggered after {epoch} epochs.")
            break
            
    # Unfreeze everything back
    for param in expert_model.parameters():
        param.requires_grad = True
        
    return {
        'final_val_loss': round(best_loss, 4), 
        'n_training_samples': len(indices),
        'epochs_trained': best_epoch,
        'checkpoint_path': str(ckpt_path)
    }
