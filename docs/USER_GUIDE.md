# User Guide - Oil Well Production Forecast

## Overview
Oil well production forecasting system using Machine Learning. Integrates 7 regression models, Arps decline curve analysis, Isolation Forest anomaly detection, and an interactive web dashboard.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/oil-well-forecast.git
cd oil-well-forecast
pip install -r requirements.txt
```

### Training Models
```bash
python scripts/train.py
```

### Running Predictions
```bash
python scripts/predict.py
```

### Full Evaluation
```bash
python scripts/evaluate.py
```

### Starting the Server
```bash
python app.py
```
Open http://localhost:5002 in your browser.

## Dashboard Features

### 1. Dashboard Section
- Model comparison table (7 models: ExtraTrees, GradientBoosting, RandomForest, MLP, Ridge, SVR, ElasticNet)
- Best model selection with R2, MAE, MAPE metrics
- Sample predictions visualization

### 2. ML Prediction Section
- 16-field form for real-time oil rate prediction
- Input fields for well properties, operating conditions, and production data
- Instant prediction with status indicator

### 3. Decline Forecast Section
- Arps decline curve input (qi, b, decline rate, months)
- Interactive decline curve chart
- EUR (Estimated Ultimate Recovery) calculation
- Economic limit configuration

### 4. Anomaly Detection Section
- 5-field anomaly verification form
- Isolation Forest-based anomaly scoring
- Binary anomaly classification with score

## Decline Analysis Types
| Type | Equation |
|------|----------|
| Exponential | q(t) = qi * exp(-di * t) |
| Hyperbolic | q(t) = qi / (1 + b * di * t)^(1/b) |
| Harmonic | q(t) = qi / (1 + di * t) |

## API Usage

### Using curl
```bash
# Predict oil rate
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"depth": 3000, "permeability": 100, "porosity": 0.15, "thickness": 30, "oil_rate": 100, "month": 12}'

# Decline curve forecast
curl -X POST http://localhost:5002/api/well_forecast \
  -H "Content-Type: application/json" \
  -d '{"qi": 500, "b_factor": 0.5, "decline_rate": 0.05, "months": 24}'

# Check for anomalies
curl -X POST http://localhost:5002/api/anomaly_check \
  -H "Content-Type: application/json" \
  -d '{"oil_rate": 100, "gas_rate": 500, "water_rate": 20, "gas_oil_ratio": 5000, "bhp": 150}'

# Health check
curl http://localhost:5002/api/health
```

### Using Python
```python
import requests

# Predict oil rate
response = requests.post("http://localhost:5002/api/predict", json={
    "depth": 3000, "permeability": 100, "porosity": 0.15,
    "thickness": 30, "oil_rate": 100, "month": 12
})
print(f"Predicted rate: {response.json()['predicted_oil_rate']} bbl/d")

# Decline forecast
response = requests.post("http://localhost:5002/api/well_forecast", json={
    "qi": 500, "b_factor": 0.5, "decline_rate": 0.05, "months": 24
})
forecast = response.json()
print(f"EUR: {forecast['eur_bbl']} bbl")
```

## Model Performance
| Model | R2 | MAE (bbl/d) | MAPE (%) |
|-------|-----|-------------|----------|
| ExtraTrees | 0.9902 | 20.14 | 4.94 |
| GradientBoosting | 0.9891 | 21.09 | 5.58 |
| RandomForest | 0.9878 | 22.49 | 5.81 |
| MLP | 0.9289 | 58.63 | 20.50 |
| Ridge | 0.8275 | 97.20 | 55.30 |

## Well Features (21 variables)
### Static Properties
- depth_m, permeability_md, porosity, net_thickness_m, initial_pressure_psi, initial_oil_rate_bbl_d, b_factor, decline_rate, water_cut_init

### Operating Conditions
- flowing_bhp_psi, tubing_head_psi, choke_size_64, pump_speed_spm, temperature_f, month

### Engineered Features
- cumulative_oil_bbl, drawdown_psi, drawdown_ratio, pressure_ratio, decline_cumulative_ratio, thickness_productivity
