"""
train_cnn_bilstm.py — Training pipeline for CNN-BiLSTM-AR
===========================================================
Trains the benchmark model using Pinball Loss to optimize for CRPS.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'src'))

from data.sequence_dataset import get_dataloaders
from models.cnn_bilstm import CNNBiLSTM_AR

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, preds, target):
        """
        preds: [Batch, n_quantiles]
        target: [Batch]
        """
        # Move quantiles to correct device
        self.quantiles = self.quantiles.to(preds.device)
        
        target = target.unsqueeze(1).expand_as(preds)
        error = target - preds
        
        loss = torch.max(self.quantiles * error, (self.quantiles - 1) * error)
        return loss.mean()

def train_cnn_bilstm(epochs=20, window_size=168, batch_size=64, lr=1e-3):
    print("[train_cnn_bilstm] Loading feature matrix...")
    df = pd.read_parquet(ROOT / 'data/processed/feature_matrix.parquet')
    
    # Exclude datetime index to treat it as data if needed, but it's index.
    # Count features
    num_features = df.select_dtypes(include=[np.number]).drop(columns=['price'], errors='ignore').shape[1]
    print(f"[train_cnn_bilstm] num_features: {num_features}")

    loaders, scaler = get_dataloaders(df, window_size=window_size, batch_size=batch_size)
    print(f"[train_cnn_bilstm] Dataset sizes: Train={len(loaders['train'].dataset)}, Val={len(loaders['val'].dataset)}")
    
    if len(loaders['val'].dataset) == 0:
        print("Warning: Validation set is empty. Are you sure 'val' is in df['split']?")
        print("Splits found:", df['split'].unique())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_cnn_bilstm] Training on device: {device}")

    quantiles = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    model = CNNBiLSTM_AR(input_dim=num_features, n_quantiles=len(quantiles)).to(device)
    
    criterion = PinballLoss(quantiles)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    models_dir = ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / 'cnn_bilstm_best.pt'

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            loss = criterion(preds, y)
            loss.backward()
            
            # Gradient clipping to prevent explosion in BiLSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss / train_batches
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in loaders['val']:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss / val_batches
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  -> New best validation loss! Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

if __name__ == '__main__':
    train_cnn_bilstm()
