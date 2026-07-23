import sys
sys.path.insert(0, '.')
from app import app

from fastapi.testclient import TestClient
client = TestClient(app)
tests = [
    ('GET', '/api/health', None),
    ('GET', '/api/dashboard', None),
    ('POST', '/api/predict', {'oil_rate':100,'depth':3000,'porosity':0.15,'permeability':100,'thickness':30,'initial_pressure':300,'flowing_bhp':150}),
    ('POST', '/api/well_forecast', {'qi':500,'b_factor':0.5,'decline_rate':0.05,'months':24}),
    ('POST', '/api/anomaly_check', {'oil_rate':100,'gas_rate':500,'water_rate':20,'gas_oil_ratio':5000,'bhp':150}),
]
for method, ep, body in tests:
    r = client.post(ep, json=body) if method == 'POST' else client.get(ep)
    status = "OK" if r.status_code == 200 else "FAIL"
    result = r.json()
    print(f"{ep}: {r.status_code} {status} {result.get('status', '')}")
    assert r.status_code == 200, f"FAILED: {ep} returned {r.status_code}"
print("All 5 API tests passed!")
