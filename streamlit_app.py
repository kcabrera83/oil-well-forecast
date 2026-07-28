import streamlit as st
import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Oil Well Forecast", layout="wide")
st.title("Oil Well Forecast")
st.markdown("Forecast oil well production and estimate EUR.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: pickle.load(open(d / v, "rb")) for k, v in [("production", "production_forecaster.pkl"), ("decline", "decline_curve_model.pkl"), ("eur", "eur_estimator.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
depth_ft = st.sidebar.slider("Depth Ft", 1000, 15000, 8000)
permeability_md = st.sidebar.slider("Permeability Md", 0, 1000, 500)
porosity_pct = st.sidebar.slider("Porosity Pct", 5, 35, 20)
initial_pressure_psi = st.sidebar.slider("Initial Pressure Psi", 1000, 10000, 5500)
bhp_psi = st.sidebar.slider("Bhp Psi", 500, 5000, 2750)
thp_psi = st.sidebar.slider("Thp Psi", 100, 2000, 1050)
choke_size_in = st.sidebar.slider("Choke Size In", 0, 3, 1)
rpm = st.sidebar.slider("Rpm", 0, 5000, 2500)
water_cut_pct = st.sidebar.slider("Water Cut Pct", 0, 100, 50)
gor_scf_bbl = st.sidebar.slider("Gor Scf Bbl", 0, 10000, 5000)
api_gravity = st.sidebar.slider("Api Gravity", 10, 50, 30)
viscosity_cp = st.sidebar.slider("Viscosity Cp", 0, 100, 50)
reservoir_temp_f = st.sidebar.slider("Reservoir Temp F", 50, 300, 175)
net_pay_ft = st.sidebar.slider("Net Pay Ft", 10, 500, 255)
skin_factor = st.sidebar.slider("Skin Factor", -5, 20, 7)
drainage_area_acres = st.sidebar.slider("Drainage Area Acres", 40, 2000, 1020)
kh_md_ft = st.sidebar.slider("Kh Md Ft", 50, 50000, 25025)
cumulative_oil_bbl = st.sidebar.slider("Cumulative Oil Bbl", 0, 1000000, 500000)
cumulative_water_bbl = st.sidebar.slider("Cumulative Water Bbl", 0, 500000, 250000)
cumulative_gas_mcf = st.sidebar.slider("Cumulative Gas Mcf", 0, 2000000, 1000000)
production_days = st.sidebar.slider("Production Days", 0, 5000, 2500)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[depth_ft, permeability_md, porosity_pct, initial_pressure_psi, bhp_psi, thp_psi, choke_size_in, rpm, water_cut_pct, gor_scf_bbl, api_gravity, viscosity_cp, reservoir_temp_f, net_pay_ft, skin_factor, drainage_area_acres, kh_md_ft, cumulative_oil_bbl, cumulative_water_bbl, cumulative_gas_mcf, production_days]])
        m = models["production"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Production", result if isinstance(result, str) else f"{result:.4f}")
        m = models["decline"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Decline", result if isinstance(result, str) else f"{result:.4f}")
        m = models["eur"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Eur", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")