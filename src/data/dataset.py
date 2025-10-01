# src/data/dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
from pathlib import Path

class CommodityDataset(Dataset):
    """Dataset for commodity price prediction"""
    
    def __init__(
        self,
        data_path: str = "data/train.csv",
        labels_path: str = "data/train_labels.csv",
        lookback: int = 60,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True
    ):
        self.lookback = lookback
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        self.labels = pd.read_csv(labels_path)
        
        print(f"Initial data shape: {self.data.shape}")
        print(f"Initial labels shape: {self.labels.shape}")
        
        # Check for NaN/Inf
        print(f"NaN in data: {self.data.isna().sum().sum()}")
        print(f"NaN in labels: {self.labels.isna().sum().sum()}")
        
        # Handle NaN in data - fill with forward fill then backward fill
        feature_cols_initial = [col for col in self.data.columns if col != 'date_id']
        self.data[feature_cols_initial] = self.data[feature_cols_initial].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Handle NaN in labels - fill with 0 (neutral return)
        target_cols_initial = [col for col in self.labels.columns if col.startswith('target_')]
        self.labels[target_cols_initial] = self.labels[target_cols_initial].fillna(0)
        
        # Replace inf with large finite values
        self.data = self.data.replace([np.inf, -np.inf], [1e10, -1e10])
        self.labels = self.labels.replace([np.inf, -np.inf], [1e10, -1e10])
        
        print(f"After cleaning - NaN in data: {self.data.isna().sum().sum()}")
        print(f"After cleaning - NaN in labels: {self.labels.isna().sum().sum()}")
        
        # Merge
        self.df = self.data.merge(self.labels, on='date_id', how='inner')
        print(f"Merged shape: {self.df.shape}")
        
        # Get columns
        self.feature_cols = [col for col in self.data.columns if col != 'date_id']
        self.target_cols = [col for col in self.labels.columns if col.startswith('target_')]
        
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.target_cols)}")
        print(f"Total samples: {len(self.df)}")
        
        # Prepare arrays
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_cols].values.astype(np.float32)
        
        # Final check for NaN/Inf
        if np.any(np.isnan(self.features)) or np.any(np.isinf(self.features)):
            print("WARNING: NaN/Inf still present in features after cleaning!")
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if np.any(np.isnan(self.targets)) or np.any(np.isinf(self.targets)):
            print("WARNING: NaN/Inf still present in targets after cleaning!")
            self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize features
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
                print(f"Features normalized - mean: {self.features.mean():.4f}, std: {self.features.std():.4f}")
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        
        # Final check after normalization
        if np.any(np.isnan(self.features)):
            print("ERROR: NaN in features after normalization!")
            self.features = np.nan_to_num(self.features, nan=0.0)
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences")
        
        # Final validation
        assert not np.any(np.isnan(self.sequences)), "NaN in sequences!"
        assert not np.any(np.isnan(self.sequence_targets)), "NaN in targets!"
        print("✓ Data validation passed")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        
        for i in range(self.lookback, len(self.features)):
            seq = self.features[i - self.lookback:i]
            target = self.targets[i]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[idx])
        target = torch.from_numpy(self.sequence_targets[idx])
        
        # Safety check
        if torch.isnan(sequence).any() or torch.isnan(target).any():
            print(f"WARNING: NaN in batch at index {idx}")
            sequence = torch.nan_to_num(sequence, nan=0.0)
            target = torch.nan_to_num(target, nan=0.0)
        
        return sequence, target
    
    def save_scaler(self, path: str = "models/scaler.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")


def create_dataloaders(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    lookback: int = 60,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Load full dataset
    full_dataset = CommodityDataset(
        data_path="data/train.csv",
        labels_path="data/train_labels.csv",
        lookback=lookback,
        fit_scaler=True
    )
    
    # Save scaler
    full_dataset.save_scaler("models/scaler.pkl")
    
    # Calculate splits
    total_len = len(full_dataset)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    
    print(f"\nDataset splits:")
    print(f"Train: {train_size} samples ({train_ratio*100:.1f}%)")
    print(f"Val: {val_size} samples ({val_ratio*100:.1f}%)")
    print(f"Test: {total_len - train_size - val_size} samples")
    
    # Create indices
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
