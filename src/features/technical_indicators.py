# src/features/technical_indicators.py

import pandas as pd
import numpy as np
from typing import List


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands"""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    position = (series - ma) / (std * std_dev)
    
    return upper, lower, position


def add_technical_features(df: pd.DataFrame, price_columns: List[str] = None) -> pd.DataFrame:
    """
    Add technical indicators as features
    
    Args:
        df: DataFrame with price data
        price_columns: List of column names to calculate indicators for.
                      If None, will auto-detect price columns.
    
    Returns:
        DataFrame with added technical features
    """
    
    df = df.copy()
    
    # Auto-detect price columns if not provided
    if price_columns is None:
        # Look for columns that might be prices
        price_keywords = ['Close', 'Open', 'High', 'Low', 'adj_close', '_Close']
        price_columns = [col for col in df.columns 
                        if any(keyword in col for keyword in price_keywords)]
        print(f"Detected {len(price_columns)} price columns")
    
    print(f"Adding technical indicators for {len(price_columns)} columns...")
    
    for col in price_columns:
        if col not in df.columns:
            continue
        
        series = df[col]
        
        # Skip if all NaN
        if series.isna().all():
            continue
        
        # 1. Returns (most important!)
        df[f'{col}_return_1d'] = series.pct_change(1)
        df[f'{col}_return_5d'] = series.pct_change(5)
        df[f'{col}_return_20d'] = series.pct_change(20)
        
        # 2. Moving Averages
        df[f'{col}_ma_5'] = series.rolling(5).mean()
        df[f'{col}_ma_20'] = series.rolling(20).mean()
        df[f'{col}_ma_60'] = series.rolling(60).mean()
        
        # MA crossovers
        df[f'{col}_ma_5_20_diff'] = df[f'{col}_ma_5'] - df[f'{col}_ma_20']
        df[f'{col}_ma_20_60_diff'] = df[f'{col}_ma_20'] - df[f'{col}_ma_60']
        
        # Price relative to MA
        df[f'{col}_to_ma_5'] = series / df[f'{col}_ma_5'] - 1
        df[f'{col}_to_ma_20'] = series / df[f'{col}_ma_20'] - 1
        
        # 3. Volatility
        df[f'{col}_vol_5'] = series.rolling(5).std()
        df[f'{col}_vol_20'] = series.rolling(20).std()
        df[f'{col}_vol_60'] = series.rolling(60).std()
        
        # 4. RSI
        df[f'{col}_rsi_14'] = calculate_rsi(series, 14)
        
        # 5. MACD
        macd, signal = calculate_macd(series)
        df[f'{col}_macd'] = macd
        df[f'{col}_macd_signal'] = signal
        df[f'{col}_macd_diff'] = macd - signal
        
        # 6. Bollinger Bands
        bb_upper, bb_lower, bb_position = calculate_bollinger_bands(series, 20)
        df[f'{col}_bb_upper'] = bb_upper
        df[f'{col}_bb_lower'] = bb_lower
        df[f'{col}_bb_position'] = bb_position
        
        # 7. Momentum
        df[f'{col}_momentum_5'] = series - series.shift(5)
        df[f'{col}_momentum_20'] = series - series.shift(20)
        
        # 8. Rate of Change
        df[f'{col}_roc_5'] = (series / series.shift(5) - 1) * 100
        df[f'{col}_roc_20'] = (series / series.shift(20) - 1) * 100
    
    # Fill NaN created by indicators
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    print(f"Added technical features. New shape: {df.shape}")
    
    return df


def add_minimal_features(df: pd.DataFrame, price_columns: List[str] = None) -> pd.DataFrame:
    """
    Add only the most important features (faster)
    Use this if full technical_features is too slow
    """
    
    df = df.copy()
    
    if price_columns is None:
        price_keywords = ['Close', 'adj_close', '_Close']
        price_columns = [col for col in df.columns 
                        if any(keyword in col for keyword in price_keywords)]
    
    print(f"Adding minimal features for {len(price_columns)} columns...")
    
    for col in price_columns:
        if col not in df.columns or df[col].isna().all():
            continue
        
        series = df[col]
        
        # Only the most important ones
        df[f'{col}_return_1d'] = series.pct_change(1)
        df[f'{col}_return_5d'] = series.pct_change(5)
        df[f'{col}_ma_20'] = series.rolling(20).mean()
        df[f'{col}_vol_20'] = series.rolling(20).std()
        df[f'{col}_to_ma_20'] = series / df[f'{col}_ma_20'] - 1
    
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    print(f"Added minimal features. New shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test on your data
    import sys
    sys.path.append('..')
    
    print("Testing technical indicators...")
    
    # Load data
    df = pd.read_csv('../../data/train.csv')
    print(f"Original shape: {df.shape}")
    
    # Add features
    df_with_features = add_technical_features(df)
    
    print(f"Final shape: {df_with_features.shape}")
    print(f"New features added: {df_with_features.shape[1] - df.shape[1]}")
    
    # Show some new columns
    new_cols = [col for col in df_with_features.columns if col not in df.columns]
    print(f"\nSample of new features:")
    print(new_cols[:20])