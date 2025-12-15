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
    """
    Dataset for commodity price prediction with NO DATA LEAKAGE
    
    Supports:
    - Temporal ordering (no shuffle in train/val/test split)
    - Lookback windows для sequences
    - Standard scaling на train, применение на val/test
    - Lagged targets как features (доступны через API)
    """
    
    def __init__(
        self,
        data_path: str,
        lookback: int = 60,
        feature_scaler: Optional[StandardScaler] = None,
        fit_scalers: bool = True,
        indices: Optional[list] = None
    ):
        """
        Args:
            data_path: Путь к CSV с features и targets
            lookback: Размер временного окна для sequences
            feature_scaler: Готовый scaler (для val/test)
            fit_scalers: Fit scaler на этих данных (только для train)
            indices: Индексы строк для фильтрации (для split)
        """
        self.lookback = lookback
        
        # Convert to absolute path
        data_path = Path(data_path)
        if not data_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / data_path
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        print(f"Full data shape: {self.df.shape}")
        
        # Identify columns
        self.target_cols = [col for col in self.df.columns if col.startswith('target_') and '_lag' not in col]
        self.feature_cols = [col for col in self.df.columns 
                            if col not in self.target_cols and col != 'date_id']
        
        # Apply indices filter if provided
        if indices is not None:
            print(f"Filtering to indices {indices[0]}-{indices[-1]} ({len(indices)} samples)...")
            self.df = self.df.iloc[indices].reset_index(drop=True)
            print(f"After filtering: {self.df.shape}")
        
        # Check for empty dataset
        if len(self.df) == 0:
            print("WARNING: Dataset is empty!")
            self._create_empty_dataset()
            return
        
        # Drop NaN targets
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=self.target_cols)
        if len(self.df) < initial_len:
            dropped = initial_len - len(self.df)
            print(f"Dropped {dropped} rows with NaN targets ({dropped/initial_len*100:.1f}%)")
        
        # Handle missing values in features
        self.df[self.feature_cols] = self.df[self.feature_cols].ffill().bfill().fillna(0)
        
        # Replace inf
        self.df[self.feature_cols] = self.df[self.feature_cols].replace([np.inf, -np.inf], 0)
        
        print(f"Final data shape: {self.df.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.target_cols)}")
        
        # Extract arrays
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_cols].values.astype(np.float32)
        
        # Normalize features
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            if fit_scalers:
                self.features = self.feature_scaler.fit_transform(self.features)
                print(f"Features normalized (fitted) - mean: {self.features.mean():.4f}, std: {self.features.std():.4f}")
            else:
                raise ValueError("Must provide feature_scaler if fit_scalers=False")
        else:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(self.features)
            print(f"Features normalized (transformed)")
        
        # Target statistics
        print(f"\nTarget statistics:")
        print(f"  Range: [{self.targets.min():.6f}, {self.targets.max():.6f}]")
        print(f"  Mean:  {self.targets.mean():.6f}")
        print(f"  Std:   {self.targets.std():.6f}")
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences (lookback={lookback})")
        
        # Validation
        if len(self.sequences) > 0:
            assert not np.any(np.isnan(self.sequences)), "NaN in sequences!"
            assert not np.any(np.isnan(self.sequence_targets)), "NaN in targets!"
            assert not np.any(np.isinf(self.sequences)), "Inf in sequences!"
            assert not np.any(np.isinf(self.sequence_targets)), "Inf in targets!"
            print("✓ Data validation passed")
    
    def _create_empty_dataset(self):
        """Create empty dataset placeholders"""
        self.features = np.array([]).reshape(0, len(self.feature_cols)).astype(np.float32)
        self.targets = np.array([]).reshape(0, len(self.target_cols)).astype(np.float32)
        self.feature_scaler = StandardScaler()
        self.sequences = np.array([]).reshape(0, self.lookback, len(self.feature_cols)).astype(np.float32)
        self.sequence_targets = np.array([]).reshape(0, len(self.target_cols)).astype(np.float32)
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create lookback sequences"""
        sequences = []
        targets = []
        
        # Start from lookback (need history)
        for i in range(self.lookback, len(self.features)):
            seq = self.features[i - self.lookback:i]  # [lookback, features]
            target = self.targets[i]  # [targets]
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) == 0:
            return (np.array([], dtype=np.float32).reshape(0, self.lookback, len(self.feature_cols)),
                   np.array([], dtype=np.float32).reshape(0, len(self.target_cols)))
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[idx])
        target = torch.from_numpy(self.sequence_targets[idx])
        return sequence, target
    
    def save_scaler(self, path: str = "models/feature_scaler.pkl"):
        """Save feature scaler"""
        path = Path(path)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path
        
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_scaler, path)
        print(f"✓ Feature scaler saved to {path}")


def create_dataloaders(
    data_path: str = 'data/processed/train_features_v2.csv',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    lookback: int = 60,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with PROPER temporal split
    
    КРИТИЧНО:
    - Временное разбиение (не random)
    - Lookback overlap (val/test могут использовать контекст из предыдущих splits)
    - Нормализация только на train
    
    Args:
        data_path: Путь к CSV с признаками и таргетами
        train_ratio: Доля train (0.7 = 70%)
        val_ratio: Доля validation (0.15 = 15%)
        batch_size: Batch size
        lookback: Размер временного окна
        num_workers: Workers для DataLoader
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    project_root = Path(__file__).parent.parent.parent
    
    # Full path
    data_file = project_root / data_path
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}\n"
                              f"Did you run 'python scripts/create_proper_features.py'?")
    
    print(f"\n{'='*80}")
    print(f"CREATING DATALOADERS")
    print(f"{'='*80}")
    print(f"Data: {data_file}")
    print(f"Train/Val/Test ratio: {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")
    print(f"Lookback: {lookback}")
    print(f"Batch size: {batch_size}")
    
    # Load data to determine split indices
    df = pd.read_csv(data_file)
    target_cols = [c for c in df.columns if c.startswith('target_') and '_lag' not in c]
    
    # Drop NaN targets
    initial_len = len(df)
    df = df.dropna(subset=target_cols)
    if len(df) < initial_len:
        print(f"\nDropped {initial_len - len(df)} rows with NaN targets")
    
    # Calculate split points
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    
    # КРИТИЧНО: Val и Test начинаются с overlap для lookback context
    # Train: [0, train_size)
    # Val: [train_size - lookback, train_size + val_size)  ← overlap!
    # Test: [train_size + val_size - lookback, total_len)  ← overlap!
    
    train_indices = list(range(0, train_size))
    
    val_start_with_context = max(0, train_size - lookback)
    val_indices = list(range(val_start_with_context, train_size + val_size))
    
    test_start_with_context = max(0, train_size + val_size - lookback)
    test_indices = list(range(test_start_with_context, total_len))
    
    print(f"\n{'='*80}")
    print(f"TEMPORAL SPLIT (with lookback overlap)")
    print(f"{'='*80}")
    print(f"Total samples: {total_len}")
    print(f"Train: {len(train_indices)} samples [indices {train_indices[0]}-{train_indices[-1]}]")
    print(f"Val:   {len(val_indices)} samples [indices {val_indices[0]}-{val_indices[-1]}] (with {lookback} overlap)")
    print(f"Test:  {len(test_indices)} samples [indices {test_indices[0]}-{test_indices[-1]}] (with {lookback} overlap)")
    print(f"\nNote: После создания sequences с lookback={lookback}:")
    print(f"  Train sequences: ~{len(train_indices) - lookback}")
    print(f"  Val sequences:   ~{len(val_indices) - lookback}")
    print(f"  Test sequences:  ~{len(test_indices) - lookback}")
    
    # Create datasets
    print(f"\n{'='*80}")
    print("Creating TRAIN dataset...")
    print(f"{'='*80}")
    train_dataset = CommodityDataset(
        data_path=str(data_file),
        lookback=lookback,
        feature_scaler=None,
        fit_scalers=True,
        indices=train_indices
    )
    
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty!")
    
    # Save scaler
    train_dataset.save_scaler()
    
    print(f"\n{'='*80}")
    print("Creating VAL dataset...")
    print(f"{'='*80}")
    val_dataset = CommodityDataset(
        data_path=str(data_file),
        lookback=lookback,
        feature_scaler=train_dataset.feature_scaler,
        fit_scalers=False,
        indices=val_indices
    )
    
    if len(val_dataset) == 0:
        raise ValueError("Val dataset is empty!")
    
    print(f"\n{'='*80}")
    print("Creating TEST dataset...")
    print(f"{'='*80}")
    test_dataset = CommodityDataset(
        data_path=str(data_file),
        lookback=lookback,
        feature_scaler=train_dataset.feature_scaler,
        fit_scalers=False,
        indices=test_indices
    )
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty!")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Важно: не shuffle для временных рядов!
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n{'='*80}")
    print("✓ DATALOADERS CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader