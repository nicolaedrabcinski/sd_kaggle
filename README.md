# MITSUI Commodity Prediction Challenge

Time series forecasting для 424 финансовых инструментов (металлы, акции, forex) используя современные Deep Learning модели.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Особенности проекта

- 🎯 **Метрика соревнования**: Spearman correlation (Modified Sharpe Ratio)
- 🔥 **16+ моделей**: От простых (DLinear) до SOTA (PatchTST, N-HiTS)
- 📊 **Правильная работа с временными рядами**: Без data leakage
- ⚡ **Hyperparameter tuning**: Optuna для оптимизации
- 📈 **Tracking**: WandB для экспериментов

## Быстрый старт

### 1. Установка

```bash
git clone git@github.com:nicolaedrabcinski/sd_kaggle.git
cd sd_kaggle

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Или установить проект как пакет
pip install -e .
```

### 2. Загрузить данные

```bash
# Через Kaggle API
kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw/
cd data/raw && unzip mitsui-commodity-prediction-challenge.zip && cd ../..
```

### 3. Создать признаки

```bash
# КРИТИЧНО: Запустить feature engineering БЕЗ data leakage
python scripts/create_proper_features.py

# Это создаст: data/processed/train_features_v2.csv
# С lagged targets и техническими индикаторами
```

### 4. Тренировать модель

```bash
# Список доступных моделей
python scripts/train.py --list

# Тренировать базовую модель (быстро)
python scripts/train.py --model dlinear --epochs 100

# Тренировать SOTA модель
python scripts/train.py --model patchtst --epochs 100

# Тренировать все модели (долго!)
python scripts/train.py --all --epochs 50
```

### 5. Сравнить результаты

```bash
python scripts/train.py --compare
```

### 6. Создать submission

```bash
python scripts/create_submission.py --model dlinear
```

## Структура проекта

```
.
├── data/
│   ├── raw/                    # Исходные данные Kaggle (не в Git)
│   └── processed/              # Обработанные данные (не в Git)
│
├── models/
│   └── checkpoints/            # Чекпоинты моделей (не в Git)
│
├── notebooks/                  # Jupyter notebooks для EDA
│   ├── 01_eda.ipynb
│   └── eda_commodity_prediction.ipynb
│
├── scripts/                    # Скрипты для обучения
│   ├── train.py               # Главный скрипт обучения
│   ├── create_proper_features.py  # Feature engineering
│   ├── create_submission.py   # Создание submission
│   ├── optimize.py            # Hyperparameter optimization
│   ├── preprocess.py          # Препроцессинг данных
│   └── debug/                 # Отладочные скрипты
│
├── src/                       # Основной код проекта
│   ├── data/
│   │   └── dataset.py         # Dataset и DataLoader
│   │
│   ├── models/                # Реализации моделей
│   │   ├── registry.py        # Регистр всех моделей
│   │   ├── dlinear.py         # DLinear, NLinear, RLinear, FITS
│   │   ├── patchtst.py        # PatchTST, CrossFormer
│   │   ├── nhits.py           # N-HiTS, N-BEATS
│   │   ├── timesnet.py        # TimesNet, Autoformer, FEDformer
│   │   ├── cnn_attention.py   # CNN + Attention
│   │   ├── tabnet.py          # TabNet, MLP
│   │   └── ft_transformer.py  # Feature Tokenizer Transformer
│   │
│   ├── training/
│   │   ├── trainer.py         # Trainer класс
│   │   └── losses.py          # MitsuiLoss, SpearmanLoss
│   │
│   ├── features/
│   │   └── technical_indicators.py  # Технические индикаторы
│   │
│   └── utils/
│
├── outputs/                   # Результаты экспериментов
├── submissions/               # Файлы для submission
│
├── requirements.txt           # Зависимости
├── pyproject.toml            # Конфигурация проекта
├── .gitignore
└── README.md
```

## Доступные модели

### Baseline (быстрые)
- **dlinear** - Decomposition + Linear (рекомендуется для старта)
- **nlinear** - Normalized Linear
- **rlinear** - RevIN + Linear
- **fits** - Frequency Interpolation

### Transformer-based
- **patchtst** - PatchTST (SOTA для time series) ⭐
- **patchtst_ci** - PatchTST с Channel Independence
- **ft_transformer** - Feature Tokenizer Transformer
- **performer** - Линейный attention
- **crossformer** - Cross-dimension dependencies

### Advanced Time Series
- **nhits** - N-HiTS (SOTA forecasting) ⭐
- **nbeats** - N-BEATS (интерпретируемый)
- **autoformer** - Auto-correlation
- **fedformer** - Frequency enhanced
- **timesnet** - Multi-periodicity

### Tabular-focused
- **tabnet** - TabNet (attentive)
- **residual_mlp** - Deep MLP с residuals
- **xgboost_nn** - Neural network в стиле XGBoost

### CNN-based
- **cnn_attention** - Multi-scale CNN + Attention
- **wavenet** - Dilated causal convolutions

## Метрики соревнования

**Главная метрика**: Modified Sharpe Ratio на основе Spearman correlation

```python
# Для каждого инструмента вычисляется Spearman correlation
correlations = [spearman(predictions[i], targets[i]) for i in range(424)]

# Финальная метрика
mean_correlation = mean(correlations)
std_correlation = std(correlations)
modified_sharpe = mean_correlation / std_correlation
```

**Важно**:
- MSE/RMSE НЕ являются целевыми метриками
- R² может быть отрицательным для efficient markets (это нормально!)
- Directional accuracy (>50%) важнее точности значений

## Hyperparameter Optimization

```bash
# Оптимизация гиперпараметров для модели
python scripts/optimize.py --model patchtst --trials 100

# Визуализация результатов
python scripts/visualize_optuna.py
```

## Важные детали

### Без Data Leakage!

1. **Временное разбиение**: Train/Val/Test строго по времени (не random!)
2. **Lagged targets**: Используются только прошлые значения (lag 1-4)
3. **Нормализация**: Fit только на train, transform на val/test
4. **Lookback**: Val/Test имеют overlap для контекста

### Loss Function

Используется **MitsuiLoss** - комбинация:
- Spearman correlation loss (70%)
- Directional loss (30%)

```python
# В scripts/train.py
python train.py --model dlinear --loss mitsui --spearman-weight 0.7
```

### Feature Engineering

Создается `train_features_v2.csv` с:
- Lagged targets (lag 1-4 для каждого инструмента)
- Технические индикаторы (MA, EMA, RSI, Bollinger)
- Momentum features
- Volatility measures

## Troubleshooting

### Ошибка: "Data file not found"

```bash
# Убедитесь что запустили feature engineering:
python scripts/create_proper_features.py
```

### Ошибка: "CUDA out of memory"

```bash
# Уменьшите batch size:
python scripts/train.py --model patchtst --epochs 100
# Затем в коде MODEL_REGISTRY уменьшите batch_size
```

### Модель выдает NaN

- Проверьте наличие NaN в данных
- Уменьшите learning rate
- Используйте gradient clipping (уже включено)

## Результаты

Лучшие модели сохраняются в:
- `models/checkpoints/{model_name}/best_model.pth`
- `outputs/{model_name}_v2_results.json`

Сравнение моделей:
```bash
python scripts/train.py --compare
```

## Рекомендации

1. **Начните с dlinear** - быстрый baseline
2. **Оптимизируйте hyperparameters** для вашей лучшей модели
3. **Ансамбль** - комбинируйте предсказания нескольких моделей
4. **Feature engineering** - добавьте domain-specific признаки

## Технологический стек

- **PyTorch 2.0+** - Deep Learning framework
- **PyTorch Lightning** - Training utilities
- **Optuna** - Hyperparameter optimization
- **WandB** - Experiment tracking
- **Pandas/NumPy** - Data processing
- **SciPy** - Spearman correlation

## Автор

**Nicolae Drabcinski**
UTM FCIM, SD-241M
Email: drabcinski@gmail.com

## License

MIT License
