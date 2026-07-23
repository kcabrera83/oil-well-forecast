# API Documentation - Oil Well Production Forecast

## Base URL
```
http://localhost:5002
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with 4 sections (Dashboard, ML Prediction, Decline Forecast, Anomalies).

---

### GET /api/health
Service health check.

**Response (200):**
```json
{"status": "ok", "service": "oil-well-forecast"}
```

---

### GET /api/dashboard
Evaluation report and sample predictions.

**Response (200):**
```json
{
  "report": {
    "best_model": "ExtraTrees",
    "metrics": {
      "R2": 0.9902,
      "MAE": 20.14,
      "MAPE": 4.94
    }
  },
  "predictions": [
    {"actual": 350.0, "predicted": 345.2, "error": 1.37}
  ]
}
```

---

### POST /api/predict
Predict well production rate using ML models (21 features, 16 user-provided + 5 engineered).

**Request:**
```json
{
  "depth": 3000,
  "permeability": 100,
  "porosity": 0.15,
  "thickness": 30,
  "initial_pressure": 300,
  "oil_rate": 100,
  "b_factor": 0.5,
  "decline_rate": 0.05,
  "water_cut": 0.1,
  "cumulative_oil": 100000,
  "flowing_bhp": 150,
  "tubing_head": 40,
  "choke": 16,
  "pump_speed": 100,
  "temperature": 100,
  "month": 12
}
```

**Response (200):**
```json
{
  "predicted_oil_rate": 285.50,
  "status": "success"
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### POST /api/well_forecast
Future production forecast using Arps decline curve analysis.

**Request:**
```json
{
  "qi": 500,
  "b_factor": 0.5,
  "decline_rate": 0.05,
  "months": 24,
  "econ_limit": 10
}
```

**Response (200):**
```json
{
  "months": [1, 2, 3, ..., 24],
  "forecast_rates": [475.0, 451.2, 428.7, ...],
  "eur_bbl": 520000,
  "status": "success"
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### POST /api/anomaly_check
Detect anomalies in well readings using Isolation Forest.

**Request:**
```json
{
  "oil_rate": 100,
  "gas_rate": 500,
  "water_rate": 20,
  "gas_oil_ratio": 5000,
  "bhp": 150
}
```

**Response (200):**
```json
{
  "is_anomaly": false,
  "anomaly_score": -0.0523,
  "status": "success"
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

**Response (200):**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "Oil Well Forecast", "version": "1.0.0"},
  "paths": { ... }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - invalid input or processing error |
| 404 | Resource not found |
| 500 | Internal server error |
