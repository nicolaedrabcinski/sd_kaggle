# scripts/create_proper_features.py
import pandas as pd
import numpy as np
import warnings

def generate_log_returns(data, lag):
    log_returns = pd.Series(np.nan, index=data.index)
    for t in range(len(data)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                log_returns.iloc[t] = np.log(data.iloc[t + lag + 1] / data.iloc[t + 1])
            except Exception:
                log_returns.iloc[t] = np.nan
    return log_returns

# Load data
train = pd.read_csv('data/raw/train.csv')
labels = pd.read_csv('data/raw/train_labels.csv')
target_pairs = pd.read_csv('data/raw/target_pairs.csv')

# Merge
df = train.merge(labels, on='date_id')

# КРИТИЧНО: Добавь forward returns как features
for col in train.columns:
    if col == 'date_id':
        continue
    
    # Forward returns с разными лагами (это то, что модель должна предсказать)
    for lag in [1, 2, 3, 4]:
        returns = generate_log_returns(train[col], lag)
        df[f'{col}_fwd_return_lag{lag}'] = returns

# Добавь исторические returns (backward looking)
for col in train.columns:
    if col == 'date_id':
        continue
    
    # Historical returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'{col}_hist_return_lag{lag}'] = np.log(train[col] / train[col].shift(lag))

# Добавь moving averages
for col in train.columns:
    if col == 'date_id':
        continue
    
    for window in [5, 10, 20]:
        df[f'{col}_MA{window}'] = train[col].rolling(window).mean()
        df[f'{col}_std{window}'] = train[col].rolling(window).std()

# Добавь lagged targets (будут доступны во время inference)
target_cols = [c for c in labels.columns if c.startswith('target_')]
for lag in [1, 2, 3, 4]:
    for col in target_cols:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Drop rows with NaN in targets
df = df.dropna(subset=target_cols)

print(f"Total samples: {len(df)}")
print(f"Features: {len([c for c in df.columns if c not in target_cols and c != 'date_id'])}")

# Save
feature_cols = [c for c in df.columns if c not in target_cols and c != 'date_id']
df[['date_id'] + feature_cols].to_csv('data/processed/train_forward_returns.csv', index=False)

print("\nFeature breakdown:")
print(f"  Forward returns: {len([c for c in feature_cols if 'fwd_return' in c])}")
print(f"  Historical returns: {len([c for c in feature_cols if 'hist_return' in c])}")
print(f"  Moving averages: {len([c for c in feature_cols if 'MA' in c or 'std' in c])}")
print(f"  Lagged targets: {len([c for c in feature_cols if 'target_' in c and 'lag' in c])}")
print(f"  Original: {len([c for c in feature_cols if c in train.columns])}")