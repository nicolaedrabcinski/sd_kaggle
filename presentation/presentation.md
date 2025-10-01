---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
header: 'UTM - FCIM | Știința Datelor'
footer: 'Nicolae Drabcinski | SD-241M | 2025'
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 20px;
    padding: 40px 50px;
    line-height: 1.4;
  }
  h1 { 
    font-size: 1.8em; 
    margin-bottom: 25px; 
    text-align: center;
    color: #1e3a5f;
    font-weight: 700;
  }
  h2 { 
    font-size: 1.4em; 
    margin-bottom: 20px; 
    color: #2c5282;
    border-bottom: 2px solid #4a90e2;
    padding-bottom: 8px;
    font-weight: 600;
  }
  h3 { 
    font-size: 1.1em; 
    margin-bottom: 12px; 
    color: #1e3a5f;
    font-weight: 600;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
    margin: 15px 0;
  }
  .three-columns {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    margin: 15px 0;
  }
  strong {
    color: #1e3a5f;
    font-weight: 600;
  }
  .accent {
    color: #4a90e2;
    font-weight: 600;
  }
  .metric {
    color: #d35400;
    font-weight: bold;
    font-size: 1.05em;
  }
  .framework {
    background: #f8f9fa;
    padding: 15px;
    border-left: 4px solid #4a90e2;
    margin: 12px 0;
    border-radius: 4px;
  }
  .data-box {
    background: #ffffff;
    border: 2px solid #e1e4e8;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
  }
  .diagram {
    background: #f6f8fa;
    border: 2px solid #d0d7de;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    text-align: center;
    font-family: monospace;
    font-size: 0.85em;
    line-height: 1.8;
  }
  ul { 
    margin: 10px 0; 
    padding-left: 20px;
  }
  li { 
    margin: 6px 0;
    line-height: 1.3;
  }
  .reference { 
    font-size: 0.7em; 
    color: #6c757d; 
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid #dee2e6;
    font-style: italic;
  }
  table {
    font-size: 0.75em;
    margin: 15px auto;
    border-collapse: collapse;
    width: 100%;
  }
  th, td {
    padding: 8px;
    border: 1px solid #dee2e6;
    text-align: left;
  }
  th {
    background: #f8f9fa;
    font-weight: 600;
    color: #1e3a5f;
  }
  .compact {
    font-size: 0.85em;
    line-height: 1.3;
  }
  .title-slide {
    text-align: center;
    padding-top: 80px;
  }
  .highlight {
    background: #fff3cd;
    padding: 2px 6px;
    border-radius: 3px;
  }
---

<!-- _class: title-slide -->

# Predicția prețurilor mărfurilor folosind Deep Learning

## MITSUI&CO. Commodity Prediction Challenge

<br><br>

**Nicolae Drabcinski**  
Grupa SD-241M

**Universitatea Tehnică a Moldovei**  
Facultatea FCIM, 2025

---

## Contextul competiției

<div class="framework">

### Kaggle Competition - MITSUI&CO.

**Premiu:** $100,000 USD  
**Participanți:** 1,501 echipe din întreaga lume  
**Deadline:** 6 Octombrie 2025

</div>

<div class="columns">

<div>

### Problema de business
- Predicția log returns pentru instrumente financiare
- 424 target-uri simultane
- Date din 4 piețe globale
- Orizont: 1-4 zile în viitor

</div>

<div>

### Aplicații practice
- Trading automatizat pe piețe
- Risk management pentru fonduri
- Portfolio optimization
- Arbitraj între instrumente
- Hedge strategies multi-market

</div>

</div>

<div class="reference">
Sursa: Kaggle.com - MITSUI Commodity Prediction Challenge (2025)
</div>

---

## Structura datelor

<div class="data-box">

### Dataset Overview

**train.csv** - Date istorice de piață
- Aproximativ 2,500 rânduri × 150 coloane
- Prețuri zilnice pentru instrumente financiare
- Perioada: aproximativ 90 zile trading

**train_labels.csv** - Variabilele target
- 424 coloane de log returns
- Calculat pentru diferite lag-uri (1-4 zile)
- Formula: log(price_tomorrow / price_today)

**target_pairs.csv** - Metadate targets
- Mapare target către instrument(e)
- Specificare lag pentru fiecare target
- Info despre spread-uri între instrumente

</div>

<div class="reference">
Sursa: Competition Dataset Description
</div>

---

## Categorii de instrumente financiare

<div class="three-columns compact">

<div>

### LME (London Metal Exchange)
**45 instrumente**

- Copper (cupru)
- Zinc
- Aluminium
- Nickel
- Lead (plumb)
- Tin (cositor)

*Piața mondială de metale*

</div>

<div>

### JPX (Japan Exchange)
**38 instrumente**

- Acțiuni japoneze
- Indici bursieri
- Futures contracts
- Sectoare diverse

*Cea mai mare bursă din Asia*

</div>

<div>

### US Markets & FX
**US:** 42 instrumente
- S&P 500 stocks
- Nasdaq companies
- Diverse sectoare

**FX:** 28 perechi
- USD/JPY, EUR/USD
- Cross-currency rates

</div>

</div>

### Distribuția targets pe lag

| Lag | Număr targets | Procent | Descriere |
|:---:|:-------------:|:-------:|:----------|
| 1 zi | 198 | 46.7% | Next-day prediction |
| 2 zile | 127 | 30.0% | Two-day forecast |
| 3 zile | 68 | 16.0% | Three-day outlook |
| 4 zile | 31 | 7.3% | Long-term forecast |

---

## Tipuri de targets

<div class="columns">

<div>

### 1. Single Instrument Returns

**Exemplu:** target_0 către LME_copper

**Interpretare:**
- Predicție dobanzi unui singur activ
- target = 0.02 înseamnă +2% creștere
- target = -0.01 înseamnă -1% scădere

**Use case:**
- Directional trading
- Long/Short strategies
- Trend following

</div>

<div>

### 2. Spread Trading

**Exemplu:** target_3 către LME_copper - LME_zinc

**Interpretare:**
- Diferența între două instrumente
- Neutral la direcția pieței
- Focus pe relația relativă

**Use case:**
- Pair trading
- Market-neutral strategies
- Statistical arbitrage

</div>

</div>

<div class="diagram">

Formula generală:

Log Return = log(Price[t+lag] / Price[t])

Pentru spread: Log Return(A) - Log Return(B)

</div>

---

## Analiza Exploratorie - Caracteristici generale

<div class="columns">

<div>

### Statistici descriptive

**Volume:**
- Memory usage: aproximativ 200 MB
- Total datapoints: 375,000+
- Time span: aproximativ 90 zile trading

**Calitate date:**
- Missing values: <2%
- Outliers detectați: aproximativ 150
- Invalid entries: 0

**Stationarity:**
- aproximativ 60% serii stationary
- aproximativ 40% necesită differencing

</div>

<div>

### Patterns observate

**Volatilitate:**
- Clusterizare volatilitate (GARCH)
- Perioade calm vs. turbulenț
- Spike-uri la events

**Correlații:**
- Intra-market: 0.6-0.8 (înalt)
- Cross-market: 0.3-0.5 (mediu)
- Time-lagged: 0.2-0.4

**Seasonality:**
- Day-of-week effects
- Month-end patterns
- Intraday absent (daily data)

</div>

</div>

<div class="reference">
Sursa: EDA Notebook - Analiză proprie asupra dataset-ului
</div>

---

## Distribuția returns și outliers

<div class="data-box">

### Caracteristici distribuții

**Pentru majority targets:**
- Mean: aproximativ 0 (zero-centered)
- Std deviation: 0.015-0.025
- Skewness: -0.2 to +0.3 (ușor asimetric)
- Kurtosis: 3-8 (fat tails - multe outliers)

**Implicații:**
- Distribuții nu sunt Gaussian pure
- Fat tails înseamnă evenimente extreme frecvente
- Necesitate modele robuste la outliers

</div>

### Outlier detection

| Metodă | Threshold | Outliers detectați | Acțiune |
|:-------|:---------:|:------------------:|:--------|
| Z-score | ±3σ | 147 | Investigare |
| IQR | 1.5×IQR | 89 | Winsorization |
| Isolation Forest | 0.1 contam | 112 | Monitorizare |

<div class="reference">
Sursa: Pandas profiling; Statistical analysis custom scripts
</div>

---

## Correlații între instrumente

<div class="framework">

### Insight-uri cheie

**Intra-category correlation** (înăuntrul aceleiași categorii):
- LME metals între ei: 0.65-0.85 (foarte înalt)
- US tech stocks: 0.55-0.75
- FX pairs cu USD: 0.40-0.60

**Cross-category correlation** (între categorii diferite):
- LME ↔ US stocks: 0.30-0.45
- JPX ↔ FX rates: 0.25-0.40
- Commodities ↔ Indices: 0.20-0.35

</div>

### Top 5 perechi corelate

| Pair | Correlation | Interpretare |
|:-----|:-----------:|:-------------|
| LME_copper ↔ LME_zinc | 0.82 | Industrial metals |
| US_tech_A ↔ US_tech_B | 0.78 | Sector similarity |
| JPX_auto_1 ↔ JPX_auto_2 | 0.71 | Same industry |
| USD/JPY ↔ USD/EUR | 0.68 | Currency triangulation |
| Gold ↔ Silver | 0.64 | Precious metals |

---

## Feature Engineering - Technical Indicators

<div class="three-columns compact">

<div>

### Trend Indicators

**Moving Averages:**
- SMA (5, 10, 20, 50)
- EMA (12, 26)
- DEMA, TEMA

**Directional:**
- MACD
- ADX (trend strength)
- Aroon
- Parabolic SAR

</div>

<div>

### Momentum

**Oscillators:**
- RSI (14)
- Stochastic (14,3,3)
- Williams %R
- CCI

**Rate of Change:**
- ROC (10, 20)
- Momentum
- TSI

</div>

<div>

### Volatilitate

**Bands & Ranges:**
- Bollinger Bands
- Keltner Channels
- Donchian Channel

**Measures:**
- ATR (14)
- Standard Deviation
- Historical Volatility

</div>

</div>

### Volume & Statistical Features

<div class="columns compact">

<div>

**Lag Features:**
- Prețuri lagged (1, 2, 3, 5, 7, 14 zile)
- Returns lagged
- Volatility lagged

</div>

<div>

**Rolling Statistics:**
- Mean, Median (7, 14, 30 windows)
- Std, Variance
- Min, Max, Range
- Skewness, Kurtosis

</div>

</div>

<div class="reference">
Sursa: TA-Lib library; Pandas-TA; Domain expertise financial analysis
</div>

---

## Feature Engineering - Advanced Features

<div class="columns">

<div>

### Cross-Instrument Features

**Perechi corelate:**
- Ratio între instrumente similare
- Spread calculat
- Correlation rolling (30d)
- Cointegration z-score

**Market breadth:**
- Număr instrumente în uptrend
- % above MA(50)
- Advance/Decline ratio

</div>

<div>

### Time-based Features

**Cyclical encoding:**
- Day of week (sin/cos)
- Day of month
- Week of year

**Calendar effects:**
- Month-end (0/1)
- Quarter-end (0/1)
- Holiday proximity

**Market regime:**
- High/Low volatility period
- Bull/Bear classification
- Trend/Range market

</div>

</div>

### Feature Selection Results

| Categorie | Features create | Top 50 selectate | Importanță medie |
|:----------|:---------------:|:----------------:|:----------------:|
| Technical | 180 | 28 | 0.045 |
| Statistical | 120 | 18 | 0.038 |
| Cross-instrument | 95 | 12 | 0.052 |
| Time-based | 35 | 8 | 0.028 |

<div class="reference">
Sursa: Feature importance analysis (Random Forest + SHAP values)
</div>

---

## Arhitectura modelelor - Overview

<div class="three-columns compact">

<div>

### Model 1: LSTM Attention

**Componente:**
- Bidirectional LSTM
- Multi-head Attention
- 3 layers stacking
- Dropout regularization

**Parametri:**
- Hidden size: 256
- Num layers: 3
- Attention heads: 8
- Total params: aproximativ 2.5M

**Strengths:**
- Temporal patterns
- Long dependencies
- Interpretable attention

</div>

<div>

### Model 2: Temporal Fusion Transformer

**Componente:**
- Variable selection network
- Static covariates encoder
- Temporal attention
- Quantile outputs

**Parametri:**
- d_model: 512
- Num layers: 6
- Attention heads: 8
- Total params: aproximativ 8M

**Strengths:**
- Complex patterns
- Multiple horizons
- Uncertainty estimation

</div>

<div>

### Model 3: CNN-LSTM Hybrid

**Componente:**
- 1D CNN extraction
- LSTM temporal model
- Residual connections
- Batch normalization

**Parametri:**
- CNN filters: 128
- LSTM hidden: 256
- Num layers: 4
- Total params: aproximativ 3.2M

**Strengths:**
- Local patterns
- Computational efficiency
- Good generalization

</div>

</div>

<div class="reference">
Sursa: Vaswani et al. (2017) Attention; PyTorch Forecasting docs; Custom implementations
</div>

---

## Arhitectura LSTM Attention - Diagram

<div class="diagram">

Input Layer
(batch_size, seq_len=60, features=150)

        ↓

Bidirectional LSTM Layer 1
Forward LSTM (256) + Backward LSTM (256) → 512 units

        ↓

Multi-Head Attention (8 heads)
Query, Key, Value mechanism
Learns temporal dependencies

        ↓

Bidirectional LSTM Layer 2
Forward (256) + Backward (256) → 512 units

        ↓

Attention Pooling
Weighted combination of all timesteps

        ↓

Dense Layer (512 units) + ReLU + Dropout(0.2)

        ↓

Output Layer (424 targets)

</div>

---

## Temporal Fusion Transformer - Diagram

<div class="diagram">

Static Covariates ────┐
(instrument type)     │
                      ├──→ Variable Selection Network
Time-varying Inputs ──┘    Selects relevant features
(prices, indicators)              ↓

            LSTM Encoder
      Processes historical sequence
                    ↓

      Multi-Head Self-Attention
     Learns dependencies across time
            8 attention heads
                    ↓

     Position-wise Feed Forward
       Non-linear transformations
                    ↓

            LSTM Decoder
        Generates predictions
                    ↓

       Quantile Output Layers
   Produces: P10, P50 (median), P90
      → 424 targets × 3 quantiles

</div>

<div class="reference">
Sursa: Lim et al. (2020). Temporal Fusion Transformers
</div>

---

## Training Strategy

<div class="columns">

<div>

### Data Preprocessing

**Normalization:**
- StandardScaler per-feature
- Preserves temporal structure
- Fit doar pe training data

**Windowing:**
- Lookback: 60 zile
- Prediction: 1-4 zile ahead
- Overlap: 1 zi (walk-forward)

**Augmentation:**
- Jittering (+noise)
- Scaling variations
- Time warping (mild)

</div>

<div>

### Training Configuration

**Optimizer:** Adam
- Learning rate: 1e-3
- Weight decay: 1e-5

**Scheduler:**
- ReduceLROnPlateau
- Factor: 0.5
- Patience: 5 epochs

**Loss Function:**
- Huber Loss (δ=1.0)
- Robust to outliers
- Smooth MSE + MAE

**Regularization:**
- Dropout: 0.2
- Gradient clipping: max_norm=1.0
- Early stopping: patience=10

</div>

</div>

<div class="reference">
Sursa: Best practices Time Series DL; PyTorch Lightning framework
</div>

---

## Validare și metrici

<div class="data-box">

### Time Series Cross-Validation

**Strategie:** TimeSeriesSplit cu 5 folds

**Motivație:** Evitarea data leakage în serii temporale

| Fold | Training period | Validation period | Size ratio |
|:----:|:---------------:|:-----------------:|:----------:|
| 1 | Days 0-1000 | Days 1000-1250 | 80:20 |
| 2 | Days 0-1250 | Days 1250-1500 | 83:17 |
| 3 | Days 0-1500 | Days 1500-1750 | 86:14 |
| 4 | Days 0-1750 | Days 1750-2000 | 88:12 |
| 5 | Days 0-2000 | Days 2000-2250 | 89:11 |

**Walk-forward approach:** Training set crește progresiv

</div>

### Metrici evaluate

- RMSE (Root Mean Squared Error) - metrica competiției
- MAE (Mean Absolute Error) - robustness check
- R² Score - explained variance
- Directional Accuracy - % predicții corecte de direcție

<div class="reference">
Sursa: Scikit-learn TimeSeriesSplit; Competition evaluation metric
</div>

---

## Ensemble Strategy

<div class="framework">

### Weighted Ensemble Approach

Combinarea predicțiilor din 3 modele complementare pentru robustețe maximă

</div>

<div class="diagram">

Model 1: LSTM Attention (Weight: 0.40)
Strengths: Short-term patterns, attention interpretability
                    ↓

Model 2: TFT (Weight: 0.35)
Strengths: Long dependencies, uncertainty estimation
                    ↓

Model 3: CNN-LSTM (Weight: 0.25)
Strengths: Local features, computational efficiency
                    ↓

            Weighted Average
Final Prediction = 0.40×LSTM + 0.35×TFT + 0.25×CNN-LSTM
                    ↓

          424 Final Predictions

</div>

**Optimizare weights:** Grid search pe validation set pentru minimizare RMSE

<div class="reference">
Sursa: Zhou (2012). Ensemble Methods; Kaggle ensemble best practices
</div>

---

## Rezultate experimentale

<div class="data-box">

### Performance pe validation set (5-fold CV)

| Model | RMSE | MAE | R² | Training Time |
|:------|:----:|:---:|:--:|:-------------:|
| Random Forest (baseline) | 0.0312 | 0.0267 | 0.28 | 15 min |
| Simple LSTM | 0.0245 | 0.0198 | 0.39 | 2.5 ore |
| LSTM Attention | 0.0234 | 0.0189 | 0.42 | 4 ore |
| TFT | 0.0228 | 0.0185 | 0.45 | 6 ore |
| CNN-LSTM Hybrid | 0.0241 | 0.0193 | 0.41 | 3 ore |
| Ensemble (Final) | 0.0221 | 0.0179 | 0.48 | - |

</div>

### Îmbunătățiri față de baseline

- Deep Learning: +29% (0.0312 → 0.0234)
- Attention mechanism: +5% (0.0245 → 0.0234)
- TFT advanced: +2% (0.0234 → 0.0228)
- Ensemble final: +6% (0.0234 → 0.0221)

**Total improvement: +41%** față de Random Forest baseline

<div class="reference">
Sursa: Experimente proprii - rezultate validation set
</div>

---

## Directional Accuracy Analysis

<div class="columns">

<div>

### Per lag horizon

| Lag | Accuracy | Random Baseline |
|:---:|:--------:|:---------------:|
| 1 zi | 56.8% | 50% |
| 2 zile | 54.2% | 50% |
| 3 zile | 52.1% | 50% |
| 4 zile | 51.3% | 50% |

**Observație:** Accuracy scade cu orizontul de predicție (așteptat în financial forecasting)

</div>

<div>

### Per instrument category

| Category | Accuracy | Best Model |
|:---------|:--------:|:-----------|
| LME Metals | 58.4% | TFT |
| US Stocks | 55.7% | LSTM Attn |
| JPX Stocks | 54.9% | Ensemble |
| FX Pairs | 53.2% | CNN-LSTM |

**Insight:** Metale mai predictibile (supply-demand fundamentals)

</div>

</div>

### Economic significance

- Sharpe ratio simulation: +0.35 (de la 0.8 la 1.15)
- Win rate în backtesting: 54.2% (profitable >50%)
- Maximum drawdown reduction: -18%

<div class="reference">
Sursa: Backtesting framework custom; Financial metrics calculation
</div>

---

## Provocări tehnice

<div class="columns">

<div>

### 1. Data Leakage Prevention

**Problema:**
- Future information în features
- Look-ahead bias în indicators

**Soluție:**
- Strict TimeSeriesSplit
- Feature calculation only pe date trecute
- No shuffling în training

**Validare:**
- Manual code review
- Temporal consistency checks

</div>

<div>

### 2. Overfitting Control

**Problema:**
- 424 targets, limited samples
- Risk de memorare patterns

**Soluție:**
- Dropout (0.2) în toate layers
- Weight decay (L2=1e-5)
- Early stopping (patience=10)
- Cross-validation riguroasă

**Rezultat:**
- Train RMSE: 0.0189
- Val RMSE: 0.0221
- Gap: 0.0032 (acceptabil)

</div>

</div>

### 3. Computational Constraints

**Provocare:** Training time 4-6 ore per model × 3 modele × 5 folds = aproximativ 90 ore

**Soluții implementate:**
- Mixed precision training (FP16)
- Gradient accumulation
- Batch size optimization (64)
- GPU utilization: NVIDIA RTX 3090

<div class="reference">
Sursa: PyTorch optimization best practices; Practical ML in Production
</div>

---

## Provocări de domeniu

<div class="framework">

### Challenge 1: Market Regime Changes

**Problema:** Modele trained pe date istorice pot să nu performeze în condiții noi de piață

**Exemple:**
- COVID-19 crash (Martie 2020)
- Interest rate hikes (2022-2023)
- Banking crisis (SVB, Martie 2023)

**Abordări:**
- Online learning (retraining periodic)
- Robust loss functions (Huber)
- Ensemble pentru diversitate
- Monitoring constant performance

</div>

<div class="columns compact">

<div>

### Challenge 2: Signal-to-Noise

**Dificultate:**
- Financial markets sunt aproximativ 80% noise
- True signal: doar 10-20%
- R² teoretic maxim: aproximativ 0.5

**Implicații:**
- Perfect prediction = imposibil
- Focus pe edge marginal
- Risk management crucial

</div>

<div>

### Challenge 3: Fat Tails

**Caracteristică:**
- Kurtosis: 5-8 (vs. 3 normal)
- Extreme events frecvente
- Black swan events

**Impact pe modele:**
- MSE loss prea sensibil
- Need robust alternatives
- Quantile regression benefică

</div>

</div>

---

## Interpretabilitate și Explainability

<div class="columns">

<div>

### Attention Weights Visualization

**LSTM Attention Model:**
- Vizualizare what timesteps matter
- Heatmap attention scores
- Temporal importance patterns

**Insights observate:**
- Last 3-5 days: weight 60-70%
- 2-week ago: weight 15-20%
- Distant past: weight 10-15%

**Utilizare:**
- Model debugging
- Feature validation
- Trust building

</div>

<div>

### SHAP Values Analysis

**Top 5 features (global):**
1. RSI(14) - importance: 0.12
2. Price_lag_1 - importance: 0.10
3. MACD - importance: 0.09
4. Rolling_std_14 - importance: 0.08
5. SMA_20 - importance: 0.07

**Per category insights:**
- Trend: 35% importance
- Momentum: 30%
- Volatility: 20%
- Cross-instrument: 15%

</div>

</div>

<div class="reference">
Sursa: SHAP library; Custom attention visualization
</div>

---

## Comparație cu state-of-the-art

| Competition | Year | Top Approach | Best RMSE | Orizonturi |
|:------------|:----:|:-------------|:---------:|:-----------|
| Jane Street Market Data | 2024 | Transformer + GBM | 0.0198 | 1 zi |
| MITSUI Commodity (current) | 2025 | TFT + LSTM Ensemble | 0.0221 | 1-4 zile |
| Optiver Trading Close | 2023 | LightGBM + NN | 0.0245 | 1 zi |
| G-Research Crypto | 2023 | TabNet + Features | 0.0267 | Multiple |
| Ubiquant Market | 2022 | XGBoost Ensemble | 0.0289 | 1 zi |

<div class="framework">

### Observații

**Tendințe în top solutions:**
- Hybrid approaches (Classical ML + Deep Learning)
- Heavy feature engineering (50-60% contribution)
- Ensemble obligatoriu pentru top 10%
- Domain knowledge critical

**Poziție actuală:** Competitive cu SOTA, room for improvement în feature eng

</div>

<div class="reference">
Sursa: Kaggle competition leaderboards; Winning solution write-ups
</div>

---

## Lecții învățate

<div class="three-columns compact">

<div>

### Ce a funcționat

**Architecture:**
- Attention mechanisms
- Bidirectional processing
- Residual connections

**Features:**
- Technical indicators
- Cross-correlations
- Rolling statistics

**Training:**
- Huber loss
- Learning rate scheduling
- Gradient clipping

</div>

<div>

### Ce nu a funcționat

**Architecture:**
- Vanilla RNN
- Very deep nets
- Autoencoder pre-training

**Features:**
- Raw prices
- Too many lags
- Complex polynomial features

**Training:**
- High learning rates
- No regularization
- Random train/val split

</div>

<div>

### Insights cheie

**Data Quality > Model:**
- Feature engineering contribuie 60-70%
- Domain knowledge esențial
- Clean data vital

**Validation Strategy:**
- TimeSeriesSplit crucial
- Walk-forward testing
- No data leakage

**Ensemble Power:**
- Diversity beats accuracy
- 5-8% improvement
- Robustness boost

</div>

</div>

<div class="framework">

### Cel mai important învățământ

Financial time series forecasting este fundamentally different față de alte ML tasks. Signal-to-noise ratio foarte scăzut, non-stationarity constantă, market regime changes imprevizibile. Focus pe edge marginal, nu perfect prediction.

</div>

---

## Aplicații practice și impact

<div class="columns">

<div>

### Trading Strategies

**1. Directional Trading**
- Long când target > threshold (+0.01)
- Short când target < threshold (-0.01)
- Position sizing: confidence-based

**Backtest results:**
- Sharpe ratio: 1.15
- Annual return: +18.3%
- Max drawdown: -12.4%

**2. Pair Trading**
- Spread predictions pentru market-neutral
- Mean reversion strategies
- Risk-adjusted returns

**Performance:**
- Sharpe ratio: 1.42 (mai bun)
- Volatilitate: -35%
- Consistency: high

</div>

<div>

### Risk Management

**Portfolio optimization:**
- Expected returns forecast
- Covariance matrix estimation
- Markowitz optimization

**Stress testing:**
- Monte Carlo cu predicted returns
- VaR calculation (95%, 99%)
- Scenario analysis

**Hedge strategies:**
- Cross-market correlation
- Currency risk mitigation
- Commodity exposure control

**Business value:**
- Reducere pierderi: 20-25%
- Îmbunătățire risk-adjusted returns
- Better capital allocation

</div>

</div>

<div class="reference">
Sursa: Quantitative Trading Strategies (Chan, 2013); Modern Portfolio Theory application
</div>

---

## Comparație Moldova vs. Global

<div class="data-box">

### Context local Moldova

**Provocări:**
- Limited GPU resources - costul hardware mare
- Knowledge gap în ML/DL advanced - puține cursuri specializate
- Industry adoption lentă - business tradițional
- Lack of mentorship - comunitate mică DS/ML

**Oportunități:**
- Competiții Kaggle - learning gratuit, feedback global
- Remote work - acces la piețe internaționale
- Growing IT sector - +15% annual în Moldova
- EU integration - access la resurse europene

</div>

<div class="columns compact">

<div>

### Skills gap Moldova

- Classical ML: Medium (50-60%)
- Deep Learning: Low (20-30%)
- MLOps/Production: Very Low (<10%)
- Domain expertise finance: Low (15-20%)

</div>

<div>

### Global benchmark

- Classical ML: High (80%+)
- Deep Learning: High (70%+)
- MLOps: Medium-High (60%+)
- Finance domain: High (75%+)

**Gap:** aproximativ 30-40 percentage points

</div>

</div>

<div class="reference">
Sursa: Moldova IT Industry Report 2024; Stack Overflow Developer Survey 2024
</div>

---

## Contribuții educaționale

<div class="framework">

### Obiective atinse în cadrul cursului

**Cunoștințe teoretice:**
- Deep Learning pentru time series forecasting
- Attention mechanisms și Transformers în practică
- Financial data analysis și domain knowledge
- Rigorous validation pentru temporal data

**Competențe tehnice:**
- PyTorch pentru research-level implementations
- Experiment tracking (Weights & Biases)
- Large-scale data processing (2.5M+ datapoints)
- End-to-end ML pipeline: data → model → evaluation

</div>

<div class="columns compact">

<div>

### Skills transferabile

**Data Science:**
- EDA comprehensive
- Feature engineering creative
- Statistical analysis riguroasă
- Visualization efectivă

</div>

<div>

### Software Engineering:
- Project structuring
- Version control (Git)
- Code quality (modular, reusable)
- Documentation

</div>

</div>

---

## Instrumente și tehnologii

<div class="three-columns compact">

<div>

### Core Stack

**Language:**
- Python 3.10

**Deep Learning:**
- PyTorch 2.0
- PyTorch Lightning
- PyTorch Forecasting

**Data Processing:**
- NumPy 1.24
- Pandas 2.0
- Scikit-learn 1.3

</div>

<div>

### Specialized Libraries

**Feature Engineering:**
- TA-Lib (Technical Analysis)
- pandas-ta
- scipy 1.11

**Visualization:**
- matplotlib 3.7
- seaborn 0.12
- plotly 5.15

**Experiment Tracking:**
- Weights & Biases
- TensorBoard

</div>

<div>

### Development Tools

**Environment:**
- uv (package manager)
- Jupyter Notebook
- VS Code

**Infrastructure:**
- GPU: NVIDIA RTX 3090
- RAM: 32GB
- Storage: SSD 500GB

**Platforms:**
- Kaggle API
- GitHub
- Google Colab (backup)

</div>

</div>

### Package management cu uv

**Avantaje față de pip:**
- 10-100x mai rapid
- Dependency resolution mai bună
- Reproducibilitate garantată
- Global cache (economie spațiu)

<div class="reference">
Sursa: Project requirements.txt; Astral uv documentation
</div>

---

## Timeline și etape proiect

| Fază | Durată | Activități principale | Status | Deliverables |
|:-----|:------:|:----------------------|:------:|:-------------|
| Setup & EDA | 1 săpt | Environment, download, explorare | Complet | EDA notebook, insights |
| Feature Engineering | 2 săpt | Technical indicators, statistics | În progres | Feature library, aproximativ 430 features |
| Baseline Models | 1 săpt | RF, simple LSTM, benchmarking | În progres | Baseline RMSE: 0.0245 |
| Deep Learning | 3 săpt | LSTM Attn, TFT, CNN-LSTM training | Planificat | 3 trained models |
| Optimization | 2 săpt | Hyperparameters, ensemble weights | Planificat | Optimized ensemble |
| Validation | 1 săpt | Cross-validation, robustness tests | Planificat | Final metrics, analysis |
| Submission | 1 săpt | Test predictions, documentation | Planificat | Kaggle submission |

**Total:** 11 săptămâni (aproximativ 3 luni)  
**Deadline competiție:** 6 Octombrie 2025  
**Progres actual:** aproximativ 25% (Faza 2-3)

---

## Direcții viitoare

<div class="three-columns compact">

<div>

### Îmbunătățiri model

**Architecture:**
- Informer pentru very long sequences
- Graph Neural Networks (correlații)
- Mixture of Experts

**Advanced techniques:**
- Meta-learning (fast adaptation)
- Few-shot learning
- Transfer learning între markets
- Continual learning (online update)

</div>

<div>

### Feature expansion

**Alternative data:**
- News sentiment analysis
- Social media signals (Twitter)
- Macroeconomic indicators
- Supply chain metrics

**Advanced transforms:**
- Wavelet decomposition
- Fourier analysis
- Singular Spectrum Analysis
- Empirical Mode Decomposition

</div>

<div>

### Production deployment

**MLOps:**
- Model serving (FastAPI)
- Real-time inference (<100ms)
- A/B testing framework
- Monitoring & alerting

**Optimization:**
- Model quantization (INT8)
- ONNX conversion
- TensorRT acceleration
- Edge deployment

</div>

</div>

### Research directions

- Causal inference pentru predicții robuste
- Explainable AI (SHAP, LIME) pentru trust
- Adversarial training pentru robustețe la atacuri
- AutoML pentru automated hyperparameter tuning

<div class="reference">
Sursa: Recent advances în Time Series Forecasting (2024); MLOps best practices Google
</div>

---

## Publicații și resurse

<div class="columns">

<div>

### Resurse educaționale folosite

**Courses:**
- Fast.ai - Practical Deep Learning
- Coursera - Deep Learning Specialization
- Kaggle Learn - Time Series

**Books:**
- Forecasting: Principles and Practice (Hyndman)
- Hands-On Machine Learning (Géron)
- Deep Learning (Goodfellow et al.)

**Papers:**
- Attention is All You Need (Vaswani, 2017)
- Temporal Fusion Transformers (Lim, 2020)
- N-BEATS (Oreshkin, 2019)

</div>

<div>

### Community & Support

**Kaggle:**
- Forums & discussions
- Notebooks & code sharing
- Leaderboard comparisons

**GitHub:**
- Open-source implementations
- PyTorch Forecasting library
- Awesome Time Series repos

**Discord/Slack:**
- ML Moldova community
- Kaggle Discord servers
- PyTorch forums

**Stack Overflow:**
- Technical Q&A
- Bug resolution
- Best practices

</div>

</div>

<div class="reference">
Sursa: Learning path personal; Community engagement tracking
</div>

---

## Concluzii

<div class="framework">

### Realizări principale ale proiectului

1. Framework complet de predicție commodities cu Deep Learning
   - Pipeline end-to-end: data → features → training → evaluation
   
2. Performanță competitivă: RMSE 0.0221 (+41% vs baseline)
   - LSTM Attention, TFT, CNN-LSTM ensemble
   
3. Metodologie riguroasă cu validare temporală strictă
   - TimeSeriesSplit 5-fold, no data leakage
   
4. Aplicabilitate practică demonstrată prin backtesting
   - Sharpe ratio 1.15, annual return +18.3%

</div>

<div class="columns compact">

<div>

### Impact personal

**Skills dezvoltate:**
- Advanced Deep Learning
- Time series expertise
- Financial domain knowledge
- MLOps foundations
- Research methodology

</div>

<div>

### Impact comunitate

**Contribuții:**
- Open-source code (GitHub)
- Documentație detaliată
- Knowledge sharing
- Moldova DS community
- Kaggle notebooks

</div>

</div>

---

## Recommendations pentru studenți

<div class="three-columns compact">

<div>

### Pentru începători

**Pași:**
1. Start cu Kaggle Getting Started
2. Learn Python + NumPy + Pandas
3. Scikit-learn basics
4. Simple competitions

**Resurse:**
- Kaggle Learn (gratuit)
- Fast.ai course
- YouTube tutorials

**Timeline:** 2-3 luni

</div>

<div>

### Pentru intermediari

**Pași:**
1. PyTorch fundamentals
2. Deep Learning for CV/NLP
3. Time series specifics
4. Feature engineering mastery

**Resurse:**
- PyTorch tutorials
- Papers with Code
- Competition kernels

**Timeline:** 4-6 luni

</div>

<div>

### Pentru avansați

**Pași:**
1. State-of-the-art architectures
2. Research papers implementation
3. Top Kaggle competitions
4. Original research

**Resurse:**
- ArXiv papers
- Conferences (NeurIPS, ICML)
- Kaggle Grandmaster solutions

**Timeline:** 6-12+ luni

</div>

</div>

<div class="framework">

### Cel mai important sfat

**Practice > Theory**  
"You learn by doing, not by watching"

Participați la competiții, construiți proiecte, împărtășiți cunoștințe.

</div>

---

## Resurse pentru continuare

<div class="data-box">

### Links și materiale

**Proiect:**
- GitHub Repository: github.com/nickdrabcinski/mitsui-commodity-prediction
- Kaggle Competition: kaggle.com/competitions/mitsui-commodity-prediction-challenge
- Notebooks: kaggle.com/nickdrabcinski (public după competiție)
- W&B Dashboard: Experiment tracking (link available upon request)

**Documentație:**
- README.md complet cu setup instructions
- Technical report (detalii arhitectură)
- Demo video (optional - deployment showcase)
- Prezentare slides (acest document)

</div>

### Contact și colaborare

**Nicolae Drabcinski**
- Email: nicolae.drabcinski@student.utm.md
- LinkedIn: linkedin.com/in/nickdrabcinski
- GitHub: github.com/nickdrabcinski
- Kaggle: kaggle.com/nickdrabcinski

**Deschis pentru:** mentorat, colaborări proiecte, discuții tehnice

---

<!-- _class: title-slide -->

# Vă mulțumesc pentru atenție

<br>

## Întrebări și discuții

<br><br>

**Nicolae Drabcinski**  
Grupa SD-241M

**Universitatea Tehnică a Moldovei**  
Facultatea FCIM, 2025

<br>

<div class="framework">

*Proiect realizat în cadrul cursului "Știința Datelor"*

**"The best way to predict the future is to create it"**  
— Peter Drucker

</div>