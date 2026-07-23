"""FastAPI web server for the well forecast system."""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

app = FastAPI(
    title="Oil Well Forecast - Well Production Forecasting",
    description="Well production forecasting with Prophet, ARIMA, and Holt-Winters",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    try:
        bundle = joblib.load("outputs/models/time_series_models.pkl")
        models["prophet"] = bundle["models"]["prophet"]
        models["arima"] = bundle["models"]["arima"]
        models["holt_winters"] = bundle["models"]["holt_winters"]
        models["production_series"] = bundle["production_series"]
    except Exception as e:
        print(f"  Error loading models: {e}")


class PredictRequest(BaseModel):
    start_date: str = "2023-01-01"
    periods: int = 365
    model_name: str = "prophet"


class ForecastRequest(BaseModel):
    qi: float = 500.0
    b_factor: float = 0.5
    decline_rate: float = 0.05
    months: int = 24
    econ_limit: float = 10.0


class AnomalyRequest(BaseModel):
    oil_rate: float = 100.0
    gas_rate: float = 500.0
    water_rate: float = 20.0
    gas_oil_ratio: float = 5000.0
    bhp: float = 150.0


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "oil-well-forecast", "framework": "prophet+arima+statsmodels"}


@app.get("/api/dashboard")
async def api_dashboard():
    report_path = Path("outputs/evaluation_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {"best_model": "N/A", "metrics": {}}
    predictions_path = Path("outputs/predictions.json")
    if predictions_path.exists():
        with open(predictions_path) as f:
            predictions = json.load(f)
    else:
        predictions = []
    return {"report": report, "predictions": predictions}


@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        model_name = request.model_name
        if model_name not in models:
            model_name = "prophet"

        if model_name == "prophet":
            future = pd.DataFrame({
                "ds": pd.date_range(start=request.start_date, periods=request.periods, freq="D"),
            })
            forecast = models["prophet"].predict(future)
            result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            return {
                "forecast": result_df.to_dict("records"),
                "model_used": "prophet",
                "status": "success",
            }
        elif model_name == "arima":
            forecast_vals, conf_int = models["arima"].predict(
                n_periods=request.periods, return_conf_int=True
            )
            dates = pd.date_range(start=request.start_date, periods=request.periods, freq="D")
            return {
                "forecast": [
                    {
                        "ds": str(d),
                        "yhat": round(float(v), 2),
                        "yhat_lower": round(float(lo), 2),
                        "yhat_upper": round(float(hi), 2),
                    }
                    for d, v, lo, hi in zip(dates, forecast_vals, conf_int[:, 0], conf_int[:, 1])
                ],
                "model_used": "arima",
                "status": "success",
            }
        elif model_name == "holt_winters":
            forecast_vals = models["holt_winters"].forecast(request.periods)
            dates = pd.date_range(start=request.start_date, periods=request.periods, freq="D")
            return {
                "forecast": [
                    {"ds": str(d), "yhat": round(float(v), 2)}
                    for d, v in zip(dates, forecast_vals)
                ],
                "model_used": "holt_winters",
                "status": "success",
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/well_forecast")
async def api_well_forecast(request: ForecastRequest):
    try:
        qi = request.qi
        di = request.decline_rate
        b = request.b_factor
        months = request.months
        rates = []
        for t in range(1, months + 1):
            if abs(b - 1.0) < 1e-9:
                q = qi * np.exp(-di * t)
            elif abs(b) < 1e-9:
                q = qi * np.exp(-di * t)
            else:
                q = qi / (1 + b * di * t) ** (1.0 / b)
            rates.append(max(q, 0))

        eur = sum(rates)
        return {
            "months": list(range(1, months + 1)),
            "forecast_rates": [round(float(r), 2) for r in rates],
            "eur_bbl": round(float(eur), 0),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/anomaly_check")
async def api_anomaly_check(request: AnomalyRequest):
    try:
        from oil_well_forecast.models.anomaly_detector import WellAnomalyDetector
        df = pd.DataFrame([{
            "oil_rate_bbl_d": request.oil_rate,
            "gas_rate_mcf_d": request.gas_rate,
            "water_rate_bbl_d": request.water_rate,
            "gas_oil_ratio": request.gas_oil_ratio,
            "flowing_bhp_psi": request.bhp,
        }])
        detector = WellAnomalyDetector()
        detector.fit(pd.DataFrame({
            "oil_rate_bbl_d": np.random.uniform(10, 500, 100),
            "gas_rate_mcf_d": np.random.uniform(100, 5000, 100),
            "water_rate_bbl_d": np.random.uniform(5, 200, 100),
            "gas_oil_ratio": np.random.uniform(100, 3000, 100),
            "flowing_bhp_psi": np.random.uniform(50, 300, 100),
        }))
        _, scores = detector.predict(df)
        is_anomaly = scores[0] < -0.1
        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(scores[0]), 4),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
