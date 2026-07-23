"""FastAPI web server for the well forecast system."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Oil Well Forecast - Well Production Forecasting",
    description="Well production forecasting with decline curve analysis and anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    from oil_well_forecast.models.production_predictor import ProductionPredictor
    from oil_well_forecast.models.decline_analyzer import DeclineAnalyzer
    try:
        models["predictor"] = ProductionPredictor.load("outputs/models/production_predictor.pkl")
        models["analyzer"] = DeclineAnalyzer()
    except Exception as e:
        print(f"  Error loading models: {e}")


class PredictRequest(BaseModel):
    depth: float = 3000.0
    permeability: float = 100.0
    porosity: float = 0.15
    thickness: float = 30.0
    initial_pressure: float = 300.0
    oil_rate: float = 100.0
    b_factor: float = 0.5
    decline_rate: float = 0.05
    water_cut: float = 0.1
    cumulative_oil: float = 100000.0
    flowing_bhp: float = 150.0
    tubing_head: float = 40.0
    choke: float = 16.0
    pump_speed: float = 100.0
    temperature: float = 100.0
    month: float = 12.0


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
    return {"status": "ok", "service": "oil-well-forecast"}


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
    if "predictor" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        row = {
            "depth_m": request.depth,
            "permeability_md": request.permeability,
            "porosity": request.porosity,
            "net_thickness_m": request.thickness,
            "initial_pressure_psi": request.initial_pressure,
            "initial_oil_rate_bbl_d": request.oil_rate,
            "b_factor": request.b_factor,
            "decline_rate": request.decline_rate,
            "water_cut_init": request.water_cut,
            "cumulative_oil_bbl": request.cumulative_oil,
            "flowing_bhp_psi": request.flowing_bhp,
            "tubing_head_psi": request.tubing_head,
            "choke_size_64": request.choke,
            "pump_speed_spm": request.pump_speed,
            "temperature_f": request.temperature,
            "month": request.month,
        }
        df = pd.DataFrame([row])
        from oil_well_forecast.utils.preprocessor import WellPreprocessor
        prep = WellPreprocessor()
        features = prep.extract_features(df)
        prediction = models["predictor"].predict(features)[0]
        return {
            "predicted_oil_rate": round(float(prediction), 2),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/well_forecast")
async def api_well_forecast(request: ForecastRequest):
    if "analyzer" not in models:
        raise HTTPException(status_code=503, detail="Analyzer not loaded")
    try:
        analyzer = models["analyzer"]
        forecast_rates = analyzer.forecast(request.qi, request.decline_rate, request.b_factor, request.months)
        eur = analyzer.estimate_eur(request.qi, request.decline_rate, request.b_factor, request.econ_limit)
        return {
            "months": list(range(1, request.months + 1)),
            "forecast_rates": [round(float(r), 2) for r in forecast_rates],
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
