import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Oil Well Forecast", layout="wide")
st.title("Oil Well Forecast")
st.markdown("Forecast oil well production and estimate EUR.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'oil_rate': joblib.load(d / 'production_forecaster.pkl'), 'decline': joblib.load(d / 'decline_curve_model.pkl'), 'eur': joblib.load(d / 'eur_estimator.pkl')}

st.sidebar.header("Input Parameters")
depth = st.sidebar.slider('Depth', 1000, 15000, 8000)
perm = st.sidebar.slider('Perm', 0, 1000, 500)
porosity = st.sidebar.slider('Porosity', 5, 35, 20)
pressure = st.sidebar.slider('Pressure', 1000, 10000, 5500)
bhp = st.sidebar.slider('Bhp', 500, 5000, 2750)
thp = st.sidebar.slider('Thp', 100, 2000, 1050)
choke = st.sidebar.slider('Choke', 0, 3, 1)
rpm = st.sidebar.slider('Rpm', 0, 5000, 2500)
water_cut = st.sidebar.slider('Water Cut', 0, 100, 50)
gor = st.sidebar.slider('Gor', 0, 10000, 5000)
api = st.sidebar.slider('Api', 10, 50, 30)
viscosity = st.sidebar.slider('Viscosity', 0, 100, 50)
res_temp = st.sidebar.slider('Res Temp', 50, 300, 175)
net_pay = st.sidebar.slider('Net Pay', 10, 500, 255)
skin = st.sidebar.slider('Skin', -5, 20, 7)
drainage = st.sidebar.slider('Drainage', 40, 2000, 1020)
kh = st.sidebar.slider('Kh', 50, 50000, 25025)
cum_oil = st.sidebar.slider('Cum Oil', 0, 1000000, 500000)
cum_water = st.sidebar.slider('Cum Water', 0, 500000, 250000)
cum_gas = st.sidebar.slider('Cum Gas', 0, 2000000, 1000000)
prod_days = st.sidebar.slider('Prod Days', 0, 5000, 2500)

if st.sidebar.button("Run"):
    try:
        x = np.array([[depth, perm, porosity, pressure, bhp, thp, choke, rpm, water_cut, gor, api, viscosity, res_temp, net_pay, skin, drainage, kh, cum_oil, cum_water, cum_gas, prod_days]])
        cols = st.columns(3)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))