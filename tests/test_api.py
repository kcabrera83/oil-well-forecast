import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "oil-well-forecast"


def test_dashboard(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "predictions" in data


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_valid(client):
    response = client.post("/api/predict", json={})
    assert response.status_code in (200, 503)


def test_predict_with_params(client):
    payload = {
        "depth": 3500.0,
        "permeability": 150.0,
        "porosity": 0.20,
        "oil_rate": 200.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (200, 503)


def test_well_forecast_valid(client):
    response = client.post("/api/well_forecast", json={})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "months" in data
        assert "forecast_rates" in data
        assert "eur_bbl" in data


def test_well_forecast_with_params(client):
    payload = {
        "qi": 800.0,
        "b_factor": 0.7,
        "decline_rate": 0.08,
        "months": 36,
        "econ_limit": 5.0,
    }
    response = client.post("/api/well_forecast", json=payload)
    assert response.status_code in (200, 503)


def test_anomaly_check_valid(client):
    response = client.post("/api/anomaly_check", json={})
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data


def test_anomaly_check_with_params(client):
    payload = {
        "oil_rate": 150.0,
        "gas_rate": 800.0,
        "water_rate": 30.0,
        "gas_oil_ratio": 5000.0,
        "bhp": 200.0,
    }
    response = client.post("/api/anomaly_check", json=payload)
    assert response.status_code == 200
