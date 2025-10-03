# preprocess_data.py

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.features.technical_indicators import add_technical_features, add_minimal_features


def preprocess_data(use_minimal: bool = False):
    """
    Preprocess train.csv and test.csv by adding technical indicators
    
    Args:
        use_minimal: If True, only add most important features (faster)
    """
    
    print("=" * 80)
    print("Data Preprocessing: Adding Technical Indicators")
    print("=" * 80)
    
    # Process train data
    print("\nProcessing train.csv...")
    train = pd.read_csv('data/train.csv')
    print(f"Original train shape: {train.shape}")
    
    if use_minimal:
        train_enhanced = add_minimal_features(train)
    else:
        train_enhanced = add_technical_features(train)
    
    print(f"Enhanced train shape: {train_enhanced.shape}")
    print(f"New features added: {train_enhanced.shape[1] - train.shape[1]}")
    
    # Save enhanced train data
    train_enhanced.to_csv('data/train_enhanced.csv', index=False)
    print("✓ Saved to data/train_enhanced.csv")
    
    # Process test data
    print("\nProcessing test.csv...")
    test = pd.read_csv('data/test.csv')
    print(f"Original test shape: {test.shape}")
    
    if use_minimal:
        test_enhanced = add_minimal_features(test)
    else:
        test_enhanced = add_technical_features(test)
    
    print(f"Enhanced test shape: {test_enhanced.shape}")
    
    # Save enhanced test data
    test_enhanced.to_csv('data/test_enhanced.csv', index=False)
    print("✓ Saved to data/test_enhanced.csv")
    
    print("\n" + "=" * 80)
    print("Preprocessing completed!")
    print("=" * 80)
    print("\nTo use enhanced data, modify dataset.py to load:")
    print("  data_path='data/train_enhanced.csv'")
    print("  instead of 'data/train.csv'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimal', action='store_true',
                       help='Use minimal features only (faster)')
    args = parser.parse_args()
    
    preprocess_data(use_minimal=args.minimal)