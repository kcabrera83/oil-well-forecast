import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Oil Well Forecast", layout="centered")
st.title("Oil Well Forecast")

path = Path(__file__).parent / 'outputs' / 'models'
models = {}
models['rate'] = joblib.load(path / 'production_forecaster.pkl')
models['eur'] = joblib.load(path / 'eur_estimator.pkl')

def pipeline(x):
    out = {}
    m = models['rate']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        out['rate'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        out['rate'] = float(m.predict(x)[0])
    m = models['eur']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        out['eur'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        out['eur'] = float(m.predict(x)[0])
    return out

with st.form('inputs'):
    st.subheader('Input Parameters')
    cols = st.columns(2)
    depth = cols[0].slider('Depth', 1000, 15000, 8000)
    perm = cols[1].slider('Perm', 0, 1000, 500)
    poro = cols[0].slider('Poro', 5, 35, 20)
    pres = cols[1].slider('Pres', 1000, 10000, 5500)
    bhp = cols[0].slider('Bhp', 500, 5000, 2750)
    thp = cols[1].slider('Thp', 100, 2000, 1050)
    choke = cols[0].slider('Choke', 0, 3, 1)
    rpm = cols[1].slider('Rpm', 0, 5000, 2500)
    wcut = cols[0].slider('Wcut', 0, 100, 50)
    gor = cols[1].slider('Gor', 0, 10000, 5000)
    api = cols[0].slider('Api', 10, 50, 30)
    visc = cols[1].slider('Visc', 0, 100, 50)
    submitted = st.form_submit_button('Run', type='primary', use_container_width=True)

if submitted:
    results = pipeline(np.array([[depth, perm, poro, pres, bhp, thp, choke, rpm, wcut, gor, api, visc]]))
    st.divider()
    st.subheader('Results')
    mc = st.columns(len(results))
    for i, (k, v) in enumerate(results.items()):
        val = str(v) if isinstance(v, str) else f'{v:,.2f}'
        mc[i].metric(k.replace('_',' ').title(), val)