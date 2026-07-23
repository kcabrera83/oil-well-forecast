# Oil Well Production Forecast

Sistema de pronostico de produccion de pozos petroliferos basado en Machine Learning. Integra modelos de regresion, analisis de declinacion curvilinea (Arps), deteccion de anomalias y un dashboard web interactivo con Flask.

---

## Arquitectura

```
oil-well-forecast/
├── oil_well_forecast/
│   ├── __init__.py
│   ├── data_generator.py          # Generador de datos sinteticos de pozos
│   ├── models/
│   │   ├── production_predictor.py  # 7 modelos ML de regresion
│   │   ├── decline_analyzer.py     # Analisis de declinacion Arps
│   │   └── anomaly_detector.py     # Deteccion de anomalias (Isolation Forest)
│   └── utils/
│       ├── preprocessor.py          # Feature engineering y preprocesamiento
│       ├── visualizer.py            # Graficos de produccion y evaluacion
│       └── metrics.py               # Metricas de evaluacion (MAE, RMSE, R2, MAPE)
├── scripts/
│   ├── train.py                    # Entrenamiento de modelos
│   ├── predict.py                  # Prediccion + forecast + EUR
│   └── evaluate.py                 # Evaluacion completa con CV
├── templates/
│   └── index.html                  # Dashboard web interactivo
├── app.py                          # Servidor Flask (5 endpoints REST)
├── test_api.py                     # Tests automatizados de API
├── requirements.txt
├── .github/workflows/test.yml      # CI/CD con GitHub Actions
└── .gitignore
```

---

## Modelos ML

| Modelo | R2 | MAE (bbl/d) | MAPE (%) |
|--------|-----|-------------|----------|
| ExtraTrees | 0.9902 | 20.14 | 4.94 |
| GradientBoosting | 0.9891 | 21.09 | 5.58 |
| RandomForest | 0.9878 | 22.49 | 5.81 |
| MLP | 0.9289 | 58.63 | 20.50 |
| Ridge | 0.8275 | 97.20 | 55.30 |
| SVR | 0.5624 | 147.87 | 59.72 |
| ElasticNet | 0.7451 | 124.54 | 65.43 |

**Mejor modelo:** ExtraTrees con R2=0.9902 y MAPE=4.94% (cross-validation R2=0.9905 +/- 0.0005)

---

## Features (21 variables)

### Propiedades estaticas del pozo
- `depth_m` — Profundidad del pozo (m)
- `permeability_md` — Permeabilidad (mD)
- `porosity` — Porosidad
- `net_thickness_m` — Espesor neto productivo (m)
- `initial_pressure_psi` — Presion inicial del reservorio (psi)
- `initial_oil_rate_bbl_d` — Tasa inicial de aceite (bbl/d)
- `b_factor` — Factor b de Arps
- `decline_rate` — Tasa de declinacion
- `water_cut_init` — Water cut inicial

### Condiciones operativas
- `flowing_bhp_psi` — Presion de fondo flujo (psi)
- `tubing_head_psi` — Presion de cabeza de tuberia (psi)
- `choke_size_64` — Tamano de choke (1/64 pulg)
- `pump_speed_spm` — Velocidad de bomba (SPM)
- `temperature_f` — Temperatura (F)
- `month` — Mes de produccion

### Acumulado e ingenieria
- `cumulative_oil_bbl` — Aceite acumulado (bbl)
- `drawdown_psi` — Drawdown (psi)
- `drawdown_ratio` — Ratio de drawdown
- `pressure_ratio` — Ratio de presion
- `decline_cumulative_ratio` — Ratio declinacion/acumulado
- `thickness_productivity` — Productividad por espesor

---

## Analisis de Declinacion (Arps)

Ajuste automatico de curvas de declinacion para cada pozo:

| Tipo | Ecuacion |
|------|----------|
| Exponencial | q(t) = qi * exp(-di * t) |
| Hiperbolica | q(t) = qi / (1 + b * di * t)^(1/b) |
| Armonica | q(t) = qi / (1 + di * t) |

Calculo de **EUR** (Estimated Ultimate Recovery) con limite economico configurable.

---

## Deteccion de Anomalias

**Isolation Forest** para detectar pozos con comportamiento anormal:

- Tasa de aceite inesperada
- Gas-oil ratio fuera de rango
- Presion de fondo anormal
- Umbral configurable (contamination=5%)

---

## API Endpoints

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/health` | GET | Estado del servicio |
| `/api/dashboard` | GET | Metricas + predicciones |
| `/api/predict` | POST | Prediccion ML con 16 parametros |
| `/api/well_forecast` | POST | Forecast por declinacion Arps |
| `/api/anomaly_check` | POST | Verificacion de anomalias |

---

## Dashboard Web

Flask + Chart.js — panel interactivo con 4 secciones:

1. **Dashboard** — metricas del modelo y tabla de predicciones
2. **Prediccion ML** — formulario de 16 campos para prediccion en tiempo real
3. **Forecast Declinacion** — grafico interactivo de curva de declinacion + EUR
4. **Anomalias** — verificacion de anomalias con score

---

## Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelos
python scripts/train.py

# Ejecutar predicciones
python scripts/predict.py

# Evaluar modelos
python scripts/evaluate.py

# Iniciar servidor web
python app.py

# Abrir http://localhost:5000
```

---

## Datos Sinteticos

Generador de datos realistas basado en modelos de declinacion Arps:

- 200 pozos, 36 meses de historial
- 4 tipos de reservorio: arenisca, caliza, dolomita, esquisto
- Propiedades geologicas variadas (permeabilidad, porosidad, espesor)
- Condiciones operativas realistas (BHP, tubing, choke, pump)
- Distribucion de estado: activo, productivo, bombeo, cerrado

---

## CI/CD

GitHub Actions ejecuta en cada push:

1. Entrenamiento de modelos
2. Predicciones
3. Evaluacion completa
4. Tests de API (5 endpoints)

---

## License

Proyecto de investigacion en Machine Learning para la industria petrolifera.
