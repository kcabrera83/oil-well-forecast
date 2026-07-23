# Architecture - Oil Well Production Forecast

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5002       |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v-------+  +------v--------+  +------v--------+
| Production      |  | Decline       |  | Anomaly       |
| Predictor       |  | Analyzer      |  | Detector      |
| (ExtraTrees)    |  | (Arps curves) |  | (Iso Forest)  |
+---------+-------+  +------+--------+  +------+--------+
          |                  |                  |
+---------v------------------v------------------v-------+
|                WellPreprocessor                        |
|         (Feature Engineering + Scaling)                |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (200 wells x 36mo) |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic well data generator (`WellDataGenerator`) producing 200 wells with 36 months of history
- **Reservoir Types**: Sandstone, limestone, dolomite, shale
- **Well States**: Active, productive, pumping, shut-in
- **Preprocessing**: Feature engineering (drawdown, ratios, productivity index), scaling

### Model Layer

#### Production Predictor (ML)
- **Algorithm**: ExtraTrees (best) among GradientBoosting, RandomForest, MLP, Ridge, SVR, ElasticNet
- **Features**: 21 variables (9 static + 6 operating + 6 engineered)
- **Target**: Oil production rate (bbl/d)
- **Metrics**: R2=0.9902, MAE=20.14 bbl/d, MAPE=4.94%
- **Cross-validation**: R2=0.9905 +/- 0.0005

#### Decline Analyzer (Physics-based)
- **Method**: Arps decline curve fitting
- **Types**: Exponential, Hyperbolic, Harmonic
- **Output**: Future rate forecasts, EUR estimation
- **Configuration**: Initial rate (qi), decline rate (di), b-factor, economic limit

#### Anomaly Detector
- **Algorithm**: Isolation Forest
- **Input**: oil_rate, gas_rate, water_rate, gas_oil_ratio, bhp
- **Threshold**: contamination=5%, score < -0.1 = anomaly
- **Purpose**: Detect wells with abnormal production behavior

### API Layer
- **Framework**: Flask
- **Endpoints**: 6 REST endpoints (health, dashboard, predict, well_forecast, anomaly_check, docs)
- **Model Loading**: Lazy loading on first request

### Dashboard Layer
- **Frontend**: Flask + Chart.js
- **Sections**: 4 interactive panels (Dashboard, ML Prediction, Decline Forecast, Anomalies)
- **Charts**: Model comparison, decline curves, prediction scatter plots

## Data Flow

1. **Input**: Well properties and operating conditions
2. **Feature Engineering**: `WellPreprocessor` computes derived features (drawdown, ratios, productivity)
3. **ML Prediction**: ExtraTrees model predicts oil rate
4. **Decline Forecast**: Arps equations project future rates and EUR
5. **Anomaly Detection**: Isolation Forest scores abnormal readings
6. **Dashboard**: All results displayed in interactive charts

## Training Pipeline
1. Generate synthetic well data (200 wells, 36 months)
2. Visualize production curves and decline analysis
3. Engineer features from raw well data
4. Train 7 regression models, select best by R2
5. Save best model to `outputs/models/production_predictor.pkl`
6. Generate evaluation plots and model comparison

## File Structure
```
oil-well-forecast/
├── oil_well_forecast/
│   ├── data_generator.py          # Synthetic well data
│   ├── models/
│   │   ├── production_predictor.py # ML rate prediction
│   │   ├── decline_analyzer.py    # Arps decline curves
│   │   └── anomaly_detector.py    # Isolation Forest
│   └── utils/
│       ├── preprocessor.py        # Feature engineering
│       ├── visualizer.py          # Charts
│       └── metrics.py             # MAE, RMSE, R2, MAPE
├── scripts/
│   ├── train.py                   # Training pipeline
│   ├── predict.py                 # Predictions + forecast
│   └── evaluate.py                # Full evaluation
├── templates/index.html           # Dashboard
├── app.py                         # Flask server
├── test_api.py                    # API tests
└── outputs/
    ├── models/                    # Trained models
    └── plots/                     # Generated charts
```
