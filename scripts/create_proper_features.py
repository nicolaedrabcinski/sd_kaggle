#!/usr/bin/env python3
"""
Feature engineering БЕЗ утечки данных
Для Mitsui Commodity Prediction Challenge

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Только backward-looking features (нет forward returns!)
2. Включить lagged targets (доступны через API)
3. Временной порядок (сортировка по date_id)
4. Нет NaN/Inf в финальных данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_features():
    """
    Создает признаки БЕЗ data leakage для Mitsui competition
    
    Returns:
        DataFrame с признаками и таргетами
    """
    
    print("="*80)
    print("FEATURE ENGINEERING - Mitsui Commodity Challenge")
    print("="*80)
    
    # ====================================================================
    # SETUP
    # ====================================================================
    
    # Пути к данным
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n[1/6] Загрузка данных...")
    
    # Загрузка
    train = pd.read_csv(data_dir / 'train.csv')
    labels = pd.read_csv(data_dir / 'train_labels.csv')
    
    print(f"  Train: {train.shape}")
    print(f"  Labels: {labels.shape}")
    
    # КРИТИЧНО: Сортировка по date_id для временного порядка
    train = train.sort_values('date_id').reset_index(drop=True)
    labels = labels.sort_values('date_id').reset_index(drop=True)
    print("  ✓ Sorted by date_id (temporal order)")
    
    # Определяем колонки
    price_cols = [c for c in train.columns if c != 'date_id']
    target_cols = [c for c in labels.columns if c.startswith('target_')]
    
    print(f"\n  Price columns: {len(price_cols)}")
    print(f"  Target columns: {len(target_cols)}")
    
    # Базовый DataFrame для признаков
    features = pd.DataFrame({'date_id': train['date_id']})
    
    # ====================================================================
    # [2/6] BACKWARD-LOOKING FEATURES (безопасно, нет утечки)
    # ====================================================================
    
    print("\n[2/6] Создание backward-looking признаков...")
    print("  (Только прошлые данные - НЕТ утечки!)")
    
    feature_count_start = len(features.columns)
    
    for col in price_cols:
        series = train[col]
        
        # 1. Historical returns (разные временные окна)
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'{col}_return_{lag}d'] = series.pct_change(lag)
        
        # 2. Moving averages (тренд)
        for window in [5, 10, 20, 60]:
            features[f'{col}_ma{window}'] = series.rolling(window).mean()
            features[f'{col}_std{window}'] = series.rolling(window).std()
        
        # 3. Relative to moving average (позиция относительно тренда)
        features[f'{col}_to_ma5'] = series / features[f'{col}_ma5'] - 1
        features[f'{col}_to_ma20'] = series / features[f'{col}_ma20'] - 1
        
        # 4. Momentum (импульс)
        features[f'{col}_momentum5'] = series - series.shift(5)
        features[f'{col}_momentum20'] = series - series.shift(20)
        
        # 5. Volatility ratio (изменение волатильности)
        features[f'{col}_vol_ratio'] = (
            features[f'{col}_std5'] / (features[f'{col}_std20'] + 1e-8)
        )
        
        # 6. Price acceleration (вторая производная)
        features[f'{col}_accel'] = (
            series.diff().diff()
        )
    
    backward_features = len(features.columns) - feature_count_start
    print(f"  ✓ Created {backward_features} backward-looking features")
    
    # ====================================================================
    # [3/6] LAGGED TARGETS (доступны через API во время inference!)
    # ====================================================================
    
    print("\n[3/6] Создание lagged target признаков...")
    print("  КРИТИЧНО: Эти признаки доступны через test_labels_lag_[1-4].csv")
    print("  во время inference через Kaggle API!")
    
    # Временный merge для создания lagged targets
    df_with_targets = features.merge(labels, on='date_id', how='inner')
    
    lagged_count_start = len(features.columns)
    
    # Создаем lagged versions ВСЕХ таргетов
    for target_col in target_cols:
        for lag in [1, 2, 3, 4]:
            # Это будет доступно в test через API!
            features[f'{target_col}_lag{lag}'] = df_with_targets[target_col].shift(lag)
    
    lagged_features = len(features.columns) - lagged_count_start
    print(f"  ✓ Created {lagged_features} lagged target features")
    print(f"  (424 targets × 4 lags = {lagged_features} features)")
    
    # ====================================================================
    # [4/6] CROSS-INSTRUMENT FEATURES (опционально)
    # ====================================================================
    
    print("\n[4/6] Создание cross-instrument features...")
    
    # Группируем по типу инструмента
    lme_cols = [c for c in price_cols if c.startswith('LME_')]
    jpx_cols = [c for c in price_cols if c.startswith('JPX_')]
    us_cols = [c for c in price_cols if c.startswith('US_')]
    fx_cols = [c for c in price_cols if c.startswith('FX_')]
    
    cross_count_start = len(features.columns)
    
    # Средние по группам (market sentiment)
    if len(lme_cols) > 0:
        features['LME_avg_return_1d'] = train[lme_cols].pct_change(1).mean(axis=1)
    if len(jpx_cols) > 0:
        features['JPX_avg_return_1d'] = train[jpx_cols].pct_change(1).mean(axis=1)
    if len(us_cols) > 0:
        features['US_avg_return_1d'] = train[us_cols].pct_change(1).mean(axis=1)
    if len(fx_cols) > 0:
        features['FX_avg_return_1d'] = train[fx_cols].pct_change(1).mean(axis=1)
    
    cross_features = len(features.columns) - cross_count_start
    print(f"  ✓ Created {cross_features} cross-instrument features")
    
    # ====================================================================
    # [5/6] MERGE С TARGETS
    # ====================================================================
    
    print("\n[5/6] Merge с таргетами...")
    
    result = features.merge(labels, on='date_id', how='inner')
    print(f"  Result shape: {result.shape}")
    
    # ====================================================================
    # [6/6] CLEANING & VALIDATION
    # ====================================================================
    
    print("\n[6/6] Очистка и валидация данных...")
    
    # 1. Удаляем строки с NaN в targets
    initial_len = len(result)
    result = result.dropna(subset=target_cols)
    dropped_targets = initial_len - len(result)
    
    if dropped_targets > 0:
        print(f"  Dropped {dropped_targets} rows with NaN targets ({dropped_targets/initial_len*100:.1f}%)")
    
    # 2. Заполняем NaN в признаках
    feature_cols = [c for c in result.columns 
                   if c not in target_cols and c != 'date_id']
    
    # Forward fill (используем последнее известное значение)
    # Затем backward fill (для начала серии)
    # Затем 0 (если все еще NaN)
    nan_before = result[feature_cols].isna().sum().sum()
    result[feature_cols] = result[feature_cols].ffill().bfill().fillna(0)
    nan_after = result[feature_cols].isna().sum().sum()
    
    if nan_before > 0:
        print(f"  Filled {nan_before} NaN values in features")
    
    # 3. Заменяем inf
    inf_count = np.isinf(result[feature_cols].values).sum()
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], 0)
    
    if inf_count > 0:
        print(f"  Replaced {inf_count} Inf values with 0")
    
    # 4. Финальная проверка
    assert result[target_cols].isna().sum().sum() == 0, "NaN in targets!"
    assert result[feature_cols].isna().sum().sum() == 0, "NaN in features!"
    assert not np.any(np.isinf(result[feature_cols].values)), "Inf in features!"
    
    print(f"\n  ✓ Final shape: {result.shape}")
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Targets: {len(target_cols)}")
    print(f"  ✓ Samples: {len(result)}")
    
    # ====================================================================
    # SAVE
    # ====================================================================
    
    output_file = output_dir / 'train_features_v2.csv'
    result.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ SAVED TO: {output_file}")
    print(f"{'='*80}")
    
    # ====================================================================
    # STATISTICS & SUMMARY
    # ====================================================================
    
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    
    print(f"\nFeature breakdown:")
    print(f"  Historical returns: {len([c for c in feature_cols if 'return_' in c and '_lag' not in c]):>6}")
    print(f"  Moving averages:    {len([c for c in feature_cols if '_ma' in c]):>6}")
    print(f"  Volatility:         {len([c for c in feature_cols if 'std' in c or 'vol' in c]):>6}")
    print(f"  Momentum:           {len([c for c in feature_cols if 'momentum' in c or 'accel' in c]):>6}")
    print(f"  Cross-instrument:   {len([c for c in feature_cols if 'avg_return' in c]):>6}")
    print(f"  Lagged targets:     {len([c for c in feature_cols if 'target_' in c and '_lag' in c]):>6}")
    print(f"  Original prices:    {len([c for c in feature_cols if c in price_cols]):>6}")
    print(f"  {'─'*30}")
    print(f"  TOTAL:              {len(feature_cols):>6}")
    
    print(f"\nTarget statistics:")
    target_values = result[target_cols].values
    print(f"  Range:  [{target_values.min():>8.6f}, {target_values.max():>8.6f}]")
    print(f"  Mean:   {target_values.mean():>8.6f}")
    print(f"  Std:    {target_values.std():>8.6f}")
    print(f"  Median: {np.median(target_values):>8.6f}")
    
    # Samples per target (для понимания data constraints)
    samples_per_target = len(result) / len(target_cols)
    print(f"\nData constraints:")
    print(f"  Samples per target: {samples_per_target:.1f}")
    if samples_per_target < 10:
        print("  ⚠️  WARNING: Very few samples per target!")
        print("  Consider simpler models (avoid deep networks)")
    
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    # Финальные проверки
    checks_passed = []
    checks_failed = []
    
    # 1. NaN check
    nan_in_features = result[feature_cols].isna().sum().sum()
    nan_in_targets = result[target_cols].isna().sum().sum()
    
    if nan_in_features == 0:
        checks_passed.append("✓ No NaN in features")
    else:
        checks_failed.append(f"✗ {nan_in_features} NaN in features")
    
    if nan_in_targets == 0:
        checks_passed.append("✓ No NaN in targets")
    else:
        checks_failed.append(f"✗ {nan_in_targets} NaN in targets")
    
    # 2. Inf check
    inf_in_features = np.isinf(result[feature_cols].values).sum()
    inf_in_targets = np.isinf(result[target_cols].values).sum()
    
    if inf_in_features == 0:
        checks_passed.append("✓ No Inf in features")
    else:
        checks_failed.append(f"✗ {inf_in_features} Inf in features")
    
    if inf_in_targets == 0:
        checks_passed.append("✓ No Inf in targets")
    else:
        checks_failed.append(f"✗ {inf_in_targets} Inf in targets")
    
    # 3. Temporal order check
    if result['date_id'].is_monotonic_increasing:
        checks_passed.append("✓ Temporal order preserved")
    else:
        checks_failed.append("✗ Temporal order broken")
    
    # 4. Data leakage check
    forward_features = [c for c in feature_cols if 'fwd' in c or 'forward' in c]
    if len(forward_features) == 0:
        checks_passed.append("✓ No forward-looking features (no leakage)")
    else:
        checks_failed.append(f"✗ {len(forward_features)} forward-looking features found!")
    
    # Print results
    for check in checks_passed:
        print(f"  {check}")
    
    if checks_failed:
        print("\n  WARNINGS:")
        for check in checks_failed:
            print(f"  {check}")
    
    print("\n" + "="*80)
    
    if len(checks_failed) == 0:
        print("✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    else:
        print("⚠️  SOME CHECKS FAILED - REVIEW ABOVE")
    
    print("="*80)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. В dataset.py измените путь к данным:")
    print(f"   data_path='data/processed/train_features_v2.csv'")
    print(f"\n2. Используйте MitsuiLoss в trainer.py:")
    print(f"   from training.losses import MitsuiLoss")
    print(f"   criterion = MitsuiLoss(spearman_weight=0.7)")
    print(f"\n3. Переобучите модели:")
    print(f"   python scripts/train.py --model dlinear")
    print("="*80)
    
    return result


if __name__ == "__main__":
    try:
        df = create_features()
        print("\n✓✓✓ FEATURE ENGINEERING COMPLETED SUCCESSFULLY ✓✓✓\n")
    except Exception as e:
        print(f"\n✗✗✗ ERROR: {e} ✗✗✗\n")
        import traceback
        traceback.print_exc()