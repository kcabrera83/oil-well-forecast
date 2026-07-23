"""Servidor web Flask para el sistema de pronostico de pozos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)


def _load_models():
    from oil_well_forecast.models.production_predictor import ProductionPredictor
    from oil_well_forecast.models.decline_analyzer import DeclineAnalyzer
    model = ProductionPredictor.load("outputs/models/production_predictor.pkl")
    analyzer = DeclineAnalyzer()
    return model, analyzer


_predictor = None
_analyzer = None


def get_models():
    global _predictor, _analyzer
    if _predictor is None:
        _predictor, _analyzer = _load_models()
    return _predictor, _analyzer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    import json
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
    return jsonify({"report": report, "predictions": predictions})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        import pandas as pd
        data = request.get_json()
        row = {
            "depth_m": float(data.get("depth", 3000)),
            "permeability_md": float(data.get("permeability", 100)),
            "porosity": float(data.get("porosity", 0.15)),
            "net_thickness_m": float(data.get("thickness", 30)),
            "initial_pressure_psi": float(data.get("initial_pressure", 300)),
            "initial_oil_rate_bbl_d": float(data.get("oil_rate", 100)),
            "b_factor": float(data.get("b_factor", 0.5)),
            "decline_rate": float(data.get("decline_rate", 0.05)),
            "water_cut_init": float(data.get("water_cut", 0.1)),
            "cumulative_oil_bbl": float(data.get("cumulative_oil", 100000)),
            "flowing_bhp_psi": float(data.get("flowing_bhp", 150)),
            "tubing_head_psi": float(data.get("tubing_head", 40)),
            "choke_size_64": float(data.get("choke", 16)),
            "pump_speed_spm": float(data.get("pump_speed", 100)),
            "temperature_f": float(data.get("temperature", 100)),
            "month": float(data.get("month", 12)),
        }
        df = pd.DataFrame([row])
        from oil_well_forecast.utils.preprocessor import WellPreprocessor
        prep = WellPreprocessor()
        features = prep.extract_features(df)
        predictor, _ = get_models()
        prediction = predictor.predict(features)[0]
        return jsonify({
            "predicted_oil_rate": round(float(prediction), 2),
            "status": "success",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/well_forecast", methods=["POST"])
def api_well_forecast():
    try:
        data = request.get_json()
        qi = float(data.get("qi", 500))
        b = float(data.get("b_factor", 0.5))
        di = float(data.get("decline_rate", 0.05))
        months = int(data.get("months", 24))
        econ_limit = float(data.get("econ_limit", 10))

        _, analyzer = get_models()
        forecast_rates = analyzer.forecast(qi, di, b, months)
        eur = analyzer.estimate_eur(qi, di, b, econ_limit)

        months_arr = list(range(1, months + 1))
        rates_list = [round(float(r), 2) for r in forecast_rates]

        return jsonify({
            "months": months_arr,
            "forecast_rates": rates_list,
            "eur_bbl": round(float(eur), 0),
            "status": "success",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/anomaly_check", methods=["POST"])
def api_anomaly_check():
    try:
        data = request.get_json()
        features = [
            float(data.get("oil_rate", 100)),
            float(data.get("gas_rate", 500)),
            float(data.get("water_rate", 20)),
            float(data.get("gas_oil_ratio", 5000)),
            float(data.get("bhp", 150)),
        ]
        from oil_well_forecast.models.anomaly_detector import WellAnomalyDetector
        import pandas as pd
        df = pd.DataFrame([{
            "oil_rate_bbl_d": features[0],
            "gas_rate_mcf_d": features[1],
            "water_rate_bbl_d": features[2],
            "gas_oil_ratio": features[3],
            "flowing_bhp_psi": features[4],
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

        return jsonify({
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(scores[0]), 4),
            "status": "success",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "oil-well-forecast"})


@app.route("/api/docs")
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Oil Well Forecast - Pronostico de Pozos", "version": "1.0.0"},
        "paths": {
            "/": {"get": {"summary": "Dashboard principal"}},
            "/api/health": {"get": {"summary": "Health check del servicio"}},
            "/api/dashboard": {"get": {"summary": "Reporte de evaluacion y predicciones"}},
            "/api/predict": {"post": {"summary": "Predecir tasa de produccion de un pozo"}},
            "/api/well_forecast": {"post": {"summary": "Pronostico de produccion a futuro con curva de declive"}},
            "/api/anomaly_check": {"post": {"summary": "Detectar anomalias en lecturas del pozo"}},
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
