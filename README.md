# MITSUI Commodity Prediction

Time series forecasting для 424 финансовых инструментов (металлы, акции, forex) используя современные Deep Learning модели.

## Setup

```bash
git clone git@github.com:nicolaedrabcinski/sd_kaggle.git
cd sd_kaggle

# Install dependencies
uv pip install -r requirements.txt

# Download data
kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/raw/
```

## Models

- **DLinear/NLinear** - Fast baselines
- **PatchTST** - Transformer для временных рядов
- **N-HiTS** - Hierarchical interpolation (SOTA)
- **TimesNet** - Multi-periodicity modeling
- **CNN Attention** - Multi-scale convolutions

## Usage

```bash
# Preprocess data (add technical indicators)
python scripts/preprocess.py

# Train single model
python scripts/train.py --model dlinear

# Train all models
python scripts/train.py --all

# Hyperparameter optimization
python scripts/optimize.py --model patchtst --trials 50

# Compare results
python scripts/train.py --compare
```

## Structure

```
├── data/           # Kaggle competition data (not in git)
├── models/         # Saved checkpoints (not in git)
├── notebooks/      # EDA and analysis
├── scripts/        # Training scripts
└── src/
    ├── data/       # Data loading
    ├── models/     # Model implementations
    └── training/   # Training logic
```

## Stack

- PyTorch 2.0+
- Optuna (hyperparameter optimization)
- pandas, numpy, scikit-learn

## Author

Nicolae Drabcinski  
UTM FCIM, SD-241M  
drabcinski@gmail.com