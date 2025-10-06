# src/data/dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
from pathlib import Path


class CommodityDataset(Dataset):
    """Dataset for commodity price prediction with NO DATA LEAKAGE"""
    
    def __init__(
        self,
        data_path: str,
        labels_path: str,
        lookback: int = 60,
        feature_scaler: Optional[StandardScaler] = None,
        fit_scalers: bool = True,
        indices: Optional[list] = None,
        use_enhanced: bool = False
    ):
        self.lookback = lookback
        
        # Convert to Path and make absolute
        data_path = Path(data_path)
        labels_path = Path(labels_path)
        
        if not data_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / data_path
            labels_path = project_root / labels_path
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Load full data
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        labels = pd.read_csv(labels_path)
        
        print(f"Full data shape: {data.shape}")
        print(f"Full labels shape: {labels.shape}")
        
        # CRITICAL FIX: Merge FIRST, then filter by indices
        self.df = data.merge(labels, on='date_id', how='inner')
        print(f"After merge: {self.df.shape}")
        
        # Get columns
        feature_cols_temp = [col for col in data.columns if col != 'date_id']
        target_cols_temp = [col for col in labels.columns if col.startswith('target_')]
        
        # Drop NaN targets
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=target_cols_temp)
        if len(self.df) < initial_len:
            print(f"Dropped {initial_len - len(self.df)} rows with NaN targets")
        
        # NOW apply indices AFTER merge and cleaning
        if indices is not None:
            print(f"Filtering to indices {indices[0]}-{indices[-1]} ({len(indices)} samples)...")
            self.df = self.df.iloc[indices].reset_index(drop=True)
            print(f"After filtering: {self.df.shape}")
        
        # Get final columns
        self.feature_cols = [col for col in self.df.columns if col not in target_cols_temp and col != 'date_id']
        self.target_cols = [col for col in self.df.columns if col.startswith('target_')]
        
        # Handle missing values in features
        if len(self.df) > 0:
            self.df[self.feature_cols] = self.df[self.feature_cols].ffill().bfill()
        
        print(f"Final data shape: {self.df.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.target_cols)}")
        
        if len(self.df) == 0:
            print("WARNING: Dataset is empty after filtering!")
            self.features = np.array([]).reshape(0, len(self.feature_cols)).astype(np.float32)
            self.targets = np.array([]).reshape(0, len(self.target_cols)).astype(np.float32)
            self.feature_scaler = StandardScaler()
            self.sequences = np.array([]).reshape(0, lookback, len(self.feature_cols)).astype(np.float32)
            self.sequence_targets = np.array([]).reshape(0, len(self.target_cols)).astype(np.float32)
            return
        
        # Extract arrays
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_cols].values.astype(np.float32)
        
        # Replace inf/nan in features
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize features
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            if fit_scalers:
                self.features = self.feature_scaler.fit_transform(self.features)
                print(f"Features normalized - mean: {self.features.mean():.4f}, std: {self.features.std():.4f}")
        else:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(self.features)
            print(f"Features transformed using provided scaler")
        
        # Target statistics
        print(f"\nTarget statistics (raw log returns):")
        print(f"  Range: [{self.targets.min():.6f}, {self.targets.max():.6f}]")
        print(f"  Mean: {self.targets.mean():.6f}")
        print(f"  Std: {self.targets.std():.6f}")
        print(f"  Median: {np.median(self.targets):.6f}")
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences")
        
        if len(self.sequences) > 0:
            assert not np.any(np.isnan(self.sequences)), "NaN in sequences!"
            assert not np.any(np.isnan(self.sequence_targets)), "NaN in targets!"
            assert not np.any(np.isinf(self.sequences)), "Inf in sequences!"
            assert not np.any(np.isinf(self.sequence_targets)), "Inf in targets!"
            print("Data validation passed")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        
        for i in range(self.lookback, len(self.features)):
            seq = self.features[i - self.lookback:i]
            target = self.targets[i]
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) == 0:
            return np.array([], dtype=np.float32).reshape(0, self.lookback, len(self.feature_cols)), \
                   np.array([], dtype=np.float32).reshape(0, len(self.target_cols))
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[idx])
        target = torch.from_numpy(self.sequence_targets[idx])
        return sequence, target
    
    def save_scalers(self, feature_path: str = "models/feature_scaler.pkl"):
        feature_path = Path(feature_path)
        if not feature_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            feature_path = project_root / feature_path
        
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_scaler, feature_path)
        print(f"Feature scaler saved to {feature_path}")


def create_dataloaders(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    lookback: int = 60,
    num_workers: int = 4,
    use_enhanced: bool = False,
    use_forward_returns: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with NO DATA LEAKAGE"""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Choose data path
    if use_forward_returns:
        data_file = project_root / "data" / "processed" / "train_forward_returns.csv"
        labels_file = project_root / "data" / "raw" / "train_labels.csv"
        print("Using forward returns features")
    elif use_enhanced:
        data_file = project_root / "data" / "processed" / "train_enhanced.csv"
        labels_file = project_root / "data" / "raw" / "train_labels.csv"
        print("Using enhanced features")
    else:
        data_file = project_root / "data" / "raw" / "train.csv"
        labels_file = project_root / "data" / "raw" / "train_labels.csv"
        print("Using raw features")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    # Load and merge data
    print(f"Loading data for splitting from {data_file}...")
    data = pd.read_csv(data_file)
    labels = pd.read_csv(labels_file)
    
    df = data.merge(labels, on='date_id', how='inner')
    target_cols = [col for col in labels.columns if col.startswith('target_')]
    
    # Drop NaN targets
    initial_len = len(df)
    df = df.dropna(subset=target_cols)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN targets ({dropped/initial_len*100:.1f}%)")
    
    # Calculate split indices on MERGED data
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))
    
    print(f"\nDataset splits (TEMPORAL):")
    print(f"Total clean samples: {total_len}")
    print(f"Train: {len(train_indices)} samples (indices {train_indices[0]}-{train_indices[-1]})")
    print(f"Val: {len(val_indices)} samples (indices {val_indices[0]}-{val_indices[-1]})")
    print(f"Test: {len(test_indices)} samples (indices {test_indices[0]}-{test_indices[-1]})")
    print(f"Note: After lookback={lookback}, each split will have ~{lookback} fewer sequences")
    
    # Create datasets
    print("\n" + "="*80)
    print("Creating TRAIN dataset...")
    print("="*80)
    train_dataset = CommodityDataset(
        data_path=str(data_file),
        labels_path=str(labels_file),
        lookback=lookback,
        feature_scaler=None,
        fit_scalers=True,
        indices=train_indices,
        use_enhanced=use_enhanced
    )
    
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty!")
    
    train_dataset.save_scalers()
    
    print("\n" + "="*80)
    print("Creating VAL dataset...")
    print("="*80)
    val_dataset = CommodityDataset(
        data_path=str(data_file),
        labels_path=str(labels_file),
        lookback=lookback,
        feature_scaler=train_dataset.feature_scaler,
        fit_scalers=False,
        indices=val_indices,
        use_enhanced=use_enhanced
    )
    
    if len(val_dataset) == 0:
        raise ValueError("Val dataset is empty!")
    
    print("\n" + "="*80)
    print("Creating TEST dataset...")
    print("="*80)
    test_dataset = CommodityDataset(
        data_path=str(data_file),
        labels_path=str(labels_file),
        lookback=lookback,
        feature_scaler=train_dataset.feature_scaler,
        fit_scalers=False,
        indices=test_indices,
        use_enhanced=use_enhanced
    )
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty!")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader