import pandas as pd
import os

# Путь к вашим данным
base_path = "/home/nicolaedrabcinski/sd_kaggle/data"

files_to_check = {
    "train.csv": f"{base_path}/raw/train.csv",
    "train_labels.csv": f"{base_path}/raw/train_labels.csv",
    "target_pairs.csv": f"{base_path}/raw/target_pairs.csv",
    "test.csv": f"{base_path}/raw/test.csv",
    "test_labels_lag_1.csv": f"{base_path}/raw/lagged_test_labels/test_labels_lag_1.csv",
    "test_labels_lag_2.csv": f"{base_path}/raw/lagged_test_labels/test_labels_lag_2.csv",
    "test_labels_lag_3.csv": f"{base_path}/raw/lagged_test_labels/test_labels_lag_3.csv",
    "test_labels_lag_4.csv": f"{base_path}/raw/lagged_test_labels/test_labels_lag_4.csv",
    "train_enhanced.csv": f"{base_path}/processed/train_enhanced.csv",
    "test_enhanced.csv": f"{base_path}/processed/test_enhanced.csv",
    "train_forward_returns.csv": f"{base_path}/processed/train_forward_returns.csv",
}

print("=" * 80)
for name, path in files_to_check.items():
    print(f"\n{'=' * 80}")
    print(f"FILE: {name}")
    print(f"{'=' * 80}")
    
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=10)
        
        # Базовая информация
        print(f"\nShape (first 10 rows): {df.shape}")
        print(f"Columns ({len(df.columns)}): {list(df.columns[:20])}...")  # первые 20 колонок
        
        # Первые строки
        print(f"\nFirst 10 rows:")
        print(df.to_string())
        
        # Типы данных
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
    else:
        print(f"❌ File not found: {path}")
    
    print("\n" + "=" * 80)

# Дополнительная статистика
print("\n" + "=" * 80)
print("ADDITIONAL INFO")
print("=" * 80)

# Полный размер файлов без загрузки всех данных
for name, path in files_to_check.items():
    if os.path.exists(path):
        # Читаем только для подсчета строк
        with open(path, 'r') as f:
            line_count = sum(1 for line in f) - 1  # -1 для header
        
        # Читаем колонки
        df_cols = pd.read_csv(path, nrows=0)
        
        print(f"\n{name}:")
        print(f"  Total rows: {line_count:,}")
        print(f"  Total columns: {len(df_cols.columns)}")
        print(f"  File size: {os.path.getsize(path) / (1024**2):.2f} MB")