"""
dataset.py - Dataset and DataLoaders for Electricity MixVol TFT

Implements the PyTorch Dataset for loading sequence windows.
CRITICAL: Uses SequentialSampler (no shuffle) for temporal causality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ElectricityDataset(Dataset):
    """
    Dataset for Electricity Price Forecasting.
    Generates causal sliding windows over historical data with forecast targets.
    """
    def __init__(self,
                 feature_matrix_path: Path | str,
                 regime_labels_path: Path | str,
                 hurst_path: Path | str,
                 split: str = 'train',
                 seq_len: int = 168,
                 forecast_horizon: int = 24):
        
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        fm = pd.read_parquet(feature_matrix_path)
        rl = pd.read_parquet(regime_labels_path)
        hs = pd.read_parquet(hurst_path)
        
        # Filter by split
        mask = fm["split"] == split
        if not mask.any():
            raise ValueError(f"No data found for split '{split}'")
            
        fm = fm.loc[mask].copy()
        rl = rl.loc[mask].copy()
        hs = hs.loc[mask].copy()
        
        # Target column is raw price
        self.y = fm["price"].values
        self.timestamps = fm.index
        
        # Regime and Hurst
        self.regimes = rl["regime"].values
        self.hurst = hs["H"].values
        
        # Historical features (exclude split, price, static calendar)
        # Assuming calendar cyclical encodings are known in the future,
        # but the prompt treats "x_static" as calendar features.
        static_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", 
                       "month_sin", "month_cos", "is_holiday", "is_weekend"]
        hist_cols = [c for c in fm.columns if c not in static_cols + ["split", "price"]]
        
        self.x_hist = fm[hist_cols].values
        self.x_static = fm[static_cols].values
        
        # Setup valid start indices so that we have seq_len context and forecast_horizon targets
        self.n_samples = len(fm) - self.seq_len - self.forecast_horizon + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"Not enough data for sequence {seq_len} + {forecast_horizon} in split {split}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Context window: [idx, idx + seq_len)
        ctx_start = idx
        ctx_end = idx + self.seq_len
        
        # Target window: [idx + seq_len, idx + seq_len + forecast_horizon)
        tgt_start = ctx_end
        tgt_end = ctx_end + self.forecast_horizon
        
        x_hist = torch.tensor(self.x_hist[ctx_start:ctx_end], dtype=torch.float32)
        
        # Static calendar features correspond to the start of the target window (the origin) 
        # or represent the target period. For simplicity, we sample static features at origin.
        x_static = torch.tensor(self.x_static[tgt_start], dtype=torch.float32)
        
        h_t = torch.tensor([self.hurst[ctx_end - 1]], dtype=torch.float32)
        
        y = torch.tensor(self.y[tgt_start:tgt_end], dtype=torch.float32)
        regime = torch.tensor(self.regimes[tgt_start:tgt_end], dtype=torch.long)
        
        # Extract Pandas timestamp of forecast origin (which is ctx_end)
        # Note: PyTorch DataLoader cannot collate complex Pandas objects easily,
        # so we convert it to unix timestamp (seconds)
        timestamp_unix = self.timestamps[tgt_start].timestamp()

        return {
            'x_hist': x_hist,            # [seq_len, n_hist_features]
            'x_static': x_static,        # [n_static_features]
            'h_t': h_t,                  # [1]
            'y': y,                      # [forecast_horizon]
            'regime': regime,            # [forecast_horizon]
            'timestamp': timestamp_unix  # float
        }

# ---------------------------------------------------------------------------
# DataLoaders & Samplers
# ---------------------------------------------------------------------------

def get_dataloader(feature_matrix_path: Path | str,
                   regime_labels_path: Path | str,
                   hurst_path: Path | str,
                   split: str,
                   batch_size: int,
                   seq_len: int = 168,
                   forecast_horizon: int = 24,
                   num_workers: int = 4) -> DataLoader:
    """
    CRITICAL: Uses SequentialSampler (shuffle=False) to preserve temporal causality.
    """
    dataset = ElectricityDataset(
        feature_matrix_path=feature_matrix_path,
        regime_labels_path=regime_labels_path,
        hurst_path=hurst_path,
        split=split,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon
    )
    
    # Strict temporal causality constraint: no shuffling permitted.
    sampler = SequentialSampler(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

# Dry-run execution
if __name__ == "__main__":
    from pprint import pprint
    ROOT = Path(__file__).resolve().parents[2]
    
    try:
        ds = ElectricityDataset(
            feature_matrix_path=ROOT / "data/processed/feature_matrix.parquet",
            regime_labels_path=ROOT / "data/regimes/regime_labels.parquet",
            hurst_path=ROOT / "data/regimes/hurst_series.parquet",
            split="val",
            seq_len=168,
            forecast_horizon=24
        )
        print(f"Dataset length: {len(ds)}")
        sample = ds[0]
        print("Dry run __getitem__ shapes/types:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"  {k}: type {type(v)}")
        
        # Test causality / ordering
        origin_dt = pd.to_datetime(sample['timestamp'], unit='s', utc=True)
        print(f"Forecast origin timestamp: {origin_dt}")
        
    except Exception as e:
        print(f"Dry run failed: {e}")
