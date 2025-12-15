# check_samples.py
import pandas as pd

data = pd.read_csv('data/raw/train.csv')
labels = pd.read_csv('data/raw/train_labels.csv')

df = data.merge(labels, on='date_id')
target_cols = [c for c in labels.columns if c.startswith('target_')]

# После ffill
df[target_cols] = df[target_cols].fillna(method='ffill')
df = df.dropna(subset=target_cols)

print(f"После очистки NaN: {len(df)} строк")

# После lookback
lookback = 60
available_sequences = len(df) - lookback

print(f"После lookback=60: {available_sequences} sequences")

# Splits
train_size = int(available_sequences * 0.7)
val_size = int(available_sequences * 0.15)
test_size = available_sequences - train_size - val_size

print(f"\nSplits:")
print(f"  Train: {train_size} sequences")
print(f"  Val: {val_size} sequences")
print(f"  Test: {test_size} sequences")

# Оценка сложности
print(f"\nTask complexity:")
print(f"  Features: 557")
print(f"  Targets: 424")
print(f"  Samples per target: {train_size / 424:.1f}")
print(f"  Parameters needed: {557 * 60 * 424 / 1000:.0f}K+")

if train_size / 424 < 10:
    print("\n⚠️  WARNING: Very few samples per target!")
    print("   Deep learning might struggle here.")