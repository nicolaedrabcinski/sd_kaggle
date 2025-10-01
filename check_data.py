import pandas as pd
import numpy as np

print("Checking data quality...\n")

# Load data
train = pd.read_csv('data/train.csv')
labels = pd.read_csv('data/train_labels.csv')

print(f"Train shape: {train.shape}")
print(f"Labels shape: {labels.shape}")

# Check for NaN
print(f"\nNaN in train: {train.isna().sum().sum()}")
print(f"NaN in labels: {labels.isna().sum().sum()}")

# Check for inf
print(f"\nInf in train: {np.isinf(train.select_dtypes(include=[np.number])).sum().sum()}")
print(f"Inf in labels: {np.isinf(labels.select_dtypes(include=[np.number])).sum().sum()}")

# Show columns with NaN
train_nan = train.isna().sum()
if train_nan.sum() > 0:
    print("\nColumns with NaN in train:")
    print(train_nan[train_nan > 0])

labels_nan = labels.isna().sum()
if labels_nan.sum() > 0:
    print("\nColumns with NaN in labels:")
    print(labels_nan[labels_nan > 0])

# Check data types
print("\nData types in train:")
print(train.dtypes.value_counts())

print("\nFirst few rows of train:")
print(train.head())

print("\nFirst few rows of labels:")
print(labels.head())