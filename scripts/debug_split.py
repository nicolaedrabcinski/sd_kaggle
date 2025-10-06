# scripts/debug_split.py
import pandas as pd

data = pd.read_csv('data/processed/train_forward_returns.csv')
labels = pd.read_csv('data/raw/train_labels.csv')

df = data.merge(labels, on='date_id', how='inner')
target_cols = [c for c in labels.columns if c.startswith('target_')]

# Drop NaN
df = df.dropna(subset=target_cols)

print(f"Total clean samples: {len(df)}")

# Split
total_len = len(df)
train_size = int(total_len * 0.7)
val_size = int(total_len * 0.15)

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, total_len))

print(f"\nSplit sizes:")
print(f"Train indices: {len(train_indices)} ({train_indices[0]}-{train_indices[-1]})")
print(f"Val indices: {len(val_indices)} ({val_indices[0]}-{val_indices[-1]})")
print(f"Test indices: {len(test_indices)} ({test_indices[0]}-{test_indices[-1]})")

lookback = 20
print(f"\nAfter lookback={lookback}:")
print(f"Train sequences: {max(0, len(train_indices) - lookback)}")
print(f"Val sequences: {max(0, len(val_indices) - lookback)}")
print(f"Test sequences: {max(0, len(test_indices) - lookback)}")