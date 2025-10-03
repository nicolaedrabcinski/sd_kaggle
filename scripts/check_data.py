#!/usr/bin/env python3
# check_data.py - Diagnostic script to analyze data

import pandas as pd
import numpy as np
from pathlib import Path


def check_data():
    """Check data quality and distribution"""
    
    print("=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    train = pd.read_csv('data/train.csv')
    labels = pd.read_csv('data/train_labels.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check merge compatibility
    print("\n[2] Checking merge compatibility...")
    merged = train.merge(labels, on='date_id', how='inner')
    print(f"Merged shape: {merged.shape}")
    print(f"Lost {len(train) - len(merged)} rows in merge")
    
    # Check for missing values
    print("\n[3] Missing values in features:")
    missing = train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing.head(10))
    else:
        print("No missing values in features!")
    
    print("\n[4] Missing values in targets:")
    target_cols = [col for col in labels.columns if col.startswith('target_')]
    missing_targets = labels[target_cols].isnull().sum()
    missing_targets = missing_targets[missing_targets > 0].sort_values(ascending=False)
    if len(missing_targets) > 0:
        print(f"Total targets with missing: {len(missing_targets)}/{len(target_cols)}")
        print(missing_targets.head(10))
    else:
        print("No missing values in targets!")
    
    # Analyze target distribution
    print("\n[5] Target distribution analysis:")
    print(f"Total targets: {len(target_cols)}")
    
    all_targets = labels[target_cols].values.flatten()
    all_targets = all_targets[~np.isnan(all_targets)]
    
    print(f"\nAll targets statistics:")
    print(f"  Count: {len(all_targets):,}")
    print(f"  Mean: {all_targets.mean():.8f}")
    print(f"  Std: {all_targets.std():.8f}")
    print(f"  Min: {all_targets.min():.8f}")
    print(f"  Max: {all_targets.max():.8f}")
    print(f"  Median: {np.median(all_targets):.8f}")
    print(f"  25th percentile: {np.percentile(all_targets, 25):.8f}")
    print(f"  75th percentile: {np.percentile(all_targets, 75):.8f}")
    
    # Check if targets look like log returns
    print("\n[6] Target validation:")
    if abs(all_targets.mean()) < 0.001:
        print("  Mean close to 0 - GOOD for log returns")
    else:
        print(f"  WARNING: Mean is {all_targets.mean():.6f}, expected ~0")
    
    if 0.001 < all_targets.std() < 0.05:
        print("  Std in reasonable range for log returns - GOOD")
    else:
        print(f"  WARNING: Std is {all_targets.std():.6f}, expected 0.001-0.05")
    
    # Sample 5 random targets
    print("\n[7] Sample target distributions:")
    sample_targets = np.random.choice(target_cols, min(5, len(target_cols)), replace=False)
    for col in sample_targets:
        values = labels[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean: {values.mean():.8f}")
        print(f"  Std: {values.std():.8f}")
        print(f"  Range: [{values.min():.8f}, {values.max():.8f}]")
    
    # Check feature statistics
    print("\n[8] Feature statistics:")
    feature_cols = [col for col in train.columns if col != 'date_id']
    features = train[feature_cols].values
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature mean: {np.nanmean(features):.4f}")
    print(f"Feature std: {np.nanstd(features):.4f}")
    print(f"Feature min: {np.nanmin(features):.4f}")
    print(f"Feature max: {np.nanmax(features):.4f}")
    
    # Check for inf values
    print("\n[9] Checking for infinite values:")
    inf_features = np.isinf(features).sum()
    inf_targets = np.isinf(labels[target_cols].values).sum()
    print(f"Inf in features: {inf_features}")
    print(f"Inf in targets: {inf_targets}")
    
    # Check temporal order
    print("\n[10] Checking temporal order:")
    if 'date_id' in train.columns:
        date_ids = train['date_id'].values
        is_sorted = np.all(date_ids[:-1] <= date_ids[1:])
        print(f"Data is temporally sorted: {is_sorted}")
        if not is_sorted:
            print("  WARNING: Data is not sorted by date_id!")
    
    print("\n" + "=" * 80)
    print("DATA CHECK COMPLETE")
    print("=" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if len(missing) > 0:
        print("- Features have missing values - will be filled with ffill/bfill")
    
    if len(missing_targets) > 0:
        print(f"- {len(missing_targets)} targets have missing values")
        print("  These rows will be dropped during training")
    
    if abs(all_targets.mean()) > 0.01 or all_targets.std() > 0.1:
        print("- WARNING: Target distribution looks unusual for log returns")
        print("  Check if targets are correctly calculated")
    
    if inf_features > 0 or inf_targets > 0:
        print("- WARNING: Data contains infinite values")
        print("  These will be clipped during preprocessing")
    
    print("\nNext steps:")
    print("1. If data looks good, delete old scalers: rm models/*.pkl")
    print("2. Run training: python train.py")


if __name__ == "__main__":
    check_data()