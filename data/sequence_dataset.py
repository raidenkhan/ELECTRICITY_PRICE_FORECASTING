"""
sequence_dataset.py — Sequential Data Loader for CNN-BiLSTM
===========================================================
Creates 3D [Batch, Time, Features] windows for sequential deep learning models.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 168, target_col: str = 'price'):
        """
        Args:
            df: Feature matrix with 'split' and index as datetime.
            window_size: Sequence length (past hours).
        """
        self.window_size = window_size
        
        # Prepare features and targets
        # Select numeric columns only
        self.features_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        self.targets = df[target_col].values
        
        # Scaling (fit on train only to avoid leakage)
        self.scaler = StandardScaler()
        train_mask = df['split'] == 'train'
        self.scaler.fit(self.features_df[train_mask])
        
        self.features = self.scaler.transform(self.features_df)
        
        # Create indices for windows
        self.indices = []
        for i in range(window_size, len(df)):
            self.indices.append(i)
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        curr_idx = self.indices[idx]
        
        # Window of features [window_size, num_features]
        x = self.features[curr_idx - self.window_size : curr_idx]
        # Target at current step [1]
        y = self.targets[curr_idx]
        
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32)
        }

def get_dataloaders(df, window_size=168, batch_size=64):
    """Factory for sequence dataloaders."""
    # Note: We must split the dataframe FIRST to ensure index alignment in SequenceDataset
    splits = ['train', 'val', 'test']
    loaders = {}
    
    for s in splits:
        split_df = df[df['split'] == s].copy()
        # To avoid edge effects between splits, we need a small buffer of the previous split for windows
        # But for an honest benchmark, we'll just start each split cleanly after window_size.
        # Actually, in a production setting, test set windows use train/val history.
        # But since our split is chronological, we can just use the indices of the original DF.
        pass

    # A better way for temporal deep learning:
    # Use the full DF in SequenceDataset, but only sample indices belonging to the split
    full_ds = SequenceDataset(df, window_size=window_size)
    
    for s in splits:
        split_mask = df['split'] == s
        # Offset mask index by window_size because first window_size rows have no past
        valid_indices = np.where(split_mask)[0]
        # Only keep indices >= window_size
        valid_indices = valid_indices[valid_indices >= window_size]
        
        # Need to subset the Dataset or use a Sampler
        subset = torch.utils.data.Subset(full_ds, [i - window_size for i in valid_indices])
        loaders[s] = DataLoader(subset, batch_size=batch_size, shuffle=(s == 'train'))
        
    return loaders, full_ds.scaler
