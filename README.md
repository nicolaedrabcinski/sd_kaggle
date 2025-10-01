```markdown
# MITSUI Commodity Prediction Challenge

Deep Learning pentru predicția prețurilor mărfurilor - MITSUI&CO. Kaggle Competition

## Despre Proiect

Proiect de Data Science pentru competiția Kaggle MITSUI&CO. Commodity Prediction Challenge. Scopul este predicția log returns pentru 424 instrumente financiare (metale, acțiuni, forex) folosind Deep Learning.

- **Competiție:** MITSUI Commodity Prediction Challenge
- **Premiu:** $100,000 USD
- **Participanți:** 1,501 echipe
- **Deadline:** 6 Octombrie 2025

## Quick Start

```bash
# Clone repository
git clone git@github.com:nicolaedrabcinski/sd_kaggle.git
cd sd_kaggle

# Setup environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download data from Kaggle
kaggle competitions download -c mitsui-commodity-prediction-challenge -p data/
cd data && unzip mitsui-commodity-prediction-challenge.zip && rm *.zip && cd ..

# Run exploratory data analysis
jupyter notebook notebooks/01_eda.ipynb
```

## Structura Proiectului

```
sd_kaggle/
├── data/                      # Date competiție
│   ├── train.csv
│   ├── test.csv
│   ├── train_labels.csv
│   ├── target_pairs.csv
│   └── lagged_test_labels/
├── notebooks/                 # Jupyter notebooks
│   └── 01_eda.ipynb
├── src/                       # Source code
│   ├── data/
│   ├── features/
│   ├── models/
│   └── training/
├── models/                    # Model checkpoints
├── outputs/                   # Rezultate
├── presentation/              # Prezentare finală
└── requirements.txt
```

## Date și Features

### Dataset Overview

- **Train:** aproximativ 2,500 rânduri × 150 time series
- **Targets:** 424 log returns
- **Horizons:** 1-4 zile predicție
- **Markets:** LME (metale), JPX (Japonia), US (acțiuni), FX (forex)

### Feature Engineering (în lucru)

- Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Statistical Features: Rolling mean/std, lags, volatility
- Cross-Instrument: Correlații, spread-uri, ratios
- Time-based: Day of week, calendar effects

## Modele Planificate

### 1. LSTM Attention
- Bidirectional LSTM
- Multi-head attention
- Dropout regularization

### 2. Temporal Fusion Transformer
- Variable selection network
- Temporal attention
- Quantile outputs

### 3. CNN-LSTM Hybrid
- 1D CNN pentru patterns
- LSTM pentru dependencies
- Residual connections

### 4. Ensemble
- Weighted average
- Grid search optimization

## Rezultate Preliminare

**Status:** În dezvoltare

Baseline și experimente inițiale în curs de desfășurare.

## Tehnologii Utilizate

### Core Stack
- Python 3.10
- PyTorch 2.0
- PyTorch Lightning 2.0
- NumPy 1.24, Pandas 2.0
- Scikit-learn 1.3

### Feature Engineering
- TA-Lib
- pandas-ta
- scipy 1.11

### Visualization & Tracking
- matplotlib, seaborn, plotly
- Weights & Biases
- TensorBoard

### Development
- uv (package manager)
- Jupyter Notebook
- Kaggle API

### Infrastructure
- GPU: NVIDIA RTX 3090
- RAM: 32GB

## Training Strategy (planificat)

### Preprocessing
- StandardScaler normalization
- 60-day lookback window
- Walk-forward validation

### Configuration
- Optimizer: Adam
- Loss: Huber Loss
- Validation: TimeSeriesSplit (5 folds)
- Early stopping

## Timeline Proiect

- **Săptămâna 1:** Setup, EDA - Complet
- **Săptămâna 2-3:** Feature engineering - În progres
- **Săptămâna 4:** Baseline models
- **Săptămâna 5-7:** Deep Learning models
- **Săptămâna 8-9:** Optimization, ensemble
- **Săptămâna 10:** Validation, testing
- **Săptămâna 11:** Final submission

**Progres actual:** aproximativ 25% (Faza 2-3)

## Direcții Viitoare

### Pentru competiție
- Advanced feature engineering
- Model implementation și training
- Hyperparameter optimization
- Ensemble refinement

### După competiție
- JAX/Flax implementation
- Production deployment
- Model interpretability analysis


## Autor

**Nicolae Drabcinski**  
Grupa SD-241M  
Universitatea Tehnică a Moldovei - Facultatea FCIM

- Email: drabcinski@gmail.com
- GitHub: nicolaedrabcinski
- Kaggle: nickdrabcinski

## Proiect Academic

Proiect realizat în cadrul cursului "Știința Datelor"  
2025

## Note

- Datele nu sunt incluse în repository (prea mari)
- Model checkpoints nu sunt incluse
- Pentru rulare: descarcă datele de pe Kaggle

## License

Proiect educațional pentru Kaggle competition.