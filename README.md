# Oil Well Production Forecast

Oil well production forecasting system based on Time Series analysis. Integrates Facebook Prophet, ARIMA (pmdarima), Holt-Winters exponential smoothing, decline curve analysis (Arps), anomaly detection, and an interactive web dashboard with FastAPI.

---

## Architecture

```
oil-well-forecast/
├── oil_well_forecast/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic well data generator
│   ├── models/
│   │   ├── production_predictor.py  # 7 ML regression models
│   │   ├── decline_analyzer.py     # Arps decline analysis
│   │   └── anomaly_detector.py     # Anomaly detection (Isolation Forest)
│   └── utils/
│       ├── preprocessor.py          # Feature engineering and preprocessing
│       ├── visualizer.py            # Production and evaluation charts
│       └── metrics.py               # Evaluation metrics (MAE, RMSE, R2, MAPE)
├── scripts/
│   ├── train.py                    # Model training
│   ├── predict.py                  # Prediction + forecast + EUR
│   └── evaluate.py                 # Full evaluation with CV
├── templates/
│   └── index.html                  # Interactive web dashboard
├── app.py                          # Flask server (5 REST endpoints)
├── test_api.py                     # Automated API tests
├── requirements.txt
├── .github/workflows/test.yml      # CI/CD with GitHub Actions
└── .gitignore
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Time Series Framework | **Facebook Prophet** (≥1.1) |
| ARIMA Models | **pmdarima** (≥2.0) — auto_arima |
| Exponential Smoothing | **statsmodels** (≥0.14) — Holt-Winters |
| Data Processing | pandas, numpy, joblib |
| Visualization | matplotlib, seaborn |
| Web Server | FastAPI + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |

## Time Series Models

| Model | Type | Seasonality |
|-------|------|-------------|
| Prophet | Additive trend + yearly seasonality | Automatic changepoints |
| ARIMA (auto_arima) | Seasonal ARIMA(12) | Stepwise auto-selection |
| Holt-Winters | Exponential Smoothing | 12-period seasonal, additive |

---

## Features (21 variables)

### Static well properties
- `depth_m` — Well depth (m)
- `permeability_md` — Permeability (mD)
- `porosity` — Porosity
- `net_thickness_m` — Net productive thickness (m)
- `initial_pressure_psi` — Initial reservoir pressure (psi)
- `initial_oil_rate_bbl_d` — Initial oil rate (bbl/d)
- `b_factor` — Arps b factor
- `decline_rate` — Decline rate
- `water_cut_init` — Initial water cut

### Operating conditions
- `flowing_bhp_psi` — Flowing bottomhole pressure (psi)
- `tubing_head_psi` — Tubing head pressure (psi)
- `choke_size_64` — Choke size (1/64 in)
- `pump_speed_spm` — Pump speed (SPM)
- `temperature_f` — Temperature (F)
- `month` — Production month

### Cumulative and engineering
- `cumulative_oil_bbl` — Cumulative oil (bbl)
- `drawdown_psi` — Drawdown (psi)
- `drawdown_ratio` — Drawdown ratio
- `pressure_ratio` — Pressure ratio
- `decline_cumulative_ratio` — Decline/cumulative ratio
- `thickness_productivity` — Productivity per thickness

---

## Decline Analysis (Arps)

Automatic curve fitting of decline curves for each well:

| Type | Equation |
|------|----------|
| Exponential | q(t) = qi * exp(-di * t) |
| Hyperbolic | q(t) = qi / (1 + b * di * t)^(1/b) |
| Harmonic | q(t) = qi / (1 + di * t) |

Calculation of **EUR** (Estimated Ultimate Recovery) with configurable economic limit.

---

## Anomaly Detection

**Isolation Forest** to detect wells with abnormal behavior:

- Unexpected oil rate
- Gas-oil ratio out of range
- Abnormal bottomhole pressure
- Configurable threshold (contamination=5%)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service status |
| `/api/dashboard` | GET | Metrics + predictions |
| `/api/predict` | POST | ML prediction with 16 parameters |
| `/api/well_forecast` | POST | Arps decline forecast |
| `/api/anomaly_check` | POST | Anomaly verification |

---

## Web Dashboard

Flask + Chart.js — interactive panel with 4 sections:

1. **Dashboard** — model metrics and prediction table
2. **ML Prediction** — 16-field form for real-time prediction
3. **Decline Forecast** — interactive decline curve chart + EUR
4. **Anomalies** — anomaly verification with score

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python scripts/train.py

# Run predictions
python scripts/predict.py

# Evaluate models
python scripts/evaluate.py

# Start web server
python app.py

# Open http://localhost:5000
```

---

## Synthetic Data

Realistic data generator based on Arps decline models:

- 200 wells, 36 months of history
- 4 reservoir types: sandstone, limestone, dolomite, shale
- Varied geological properties (permeability, porosity, thickness)
- Realistic operating conditions (BHP, tubing, choke, pump)
- State distribution: active, productive, pumping, shut-in

---

## CI/CD

GitHub Actions runs on every push:

1. Model training
2. Predictions
3. Full evaluation
4. API tests (5 endpoints)

---

## License

Machine Learning research project for the oil and gas industry.
