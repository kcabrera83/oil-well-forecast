#!/usr/bin/env python3
"""Script de entrenamiento de modelos de produccion de pozos."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from oil_well_forecast.data_generator import WellDataGenerator
from oil_well_forecast.utils.preprocessor import WellPreprocessor
from oil_well_forecast.utils.metrics import ForecastEvaluator
from oil_well_forecast.utils.visualizer import WellVisualizer


def train_prophet(production_series: pd.Series) -> Prophet:
    """Train Prophet model for well production forecasting."""
    df = pd.DataFrame({
        "ds": pd.date_range(start="2020-01-01", periods=len(production_series), freq="D"),
        "y": production_series.values,
    })
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    return model


def train_arima(production_series: pd.Series):
    """Train ARIMA model using pmdarima auto_arima."""
    auto_model = pm.auto_arima(
        production_series.values,
        seasonal=True,
        m=12,
        suppress_warnings=True,
        stepwise=True,
    )
    return auto_model


def train_holt_winters(production_series: pd.Series):
    """Train Holt-Winters Exponential Smoothing."""
    model = ExponentialSmoothing(
        production_series.values,
        seasonal_periods=12,
        trend="add",
        seasonal="add",
    ).fit()
    return model


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO - Sistema de Pronostico de Produccion")
    print("=" * 60)

    gen = WellDataGenerator(seed=42)
    df = gen.generate(n_wells=200, n_months=36)
    gen.save(df)
    print(f"  Datos generados: {len(df)} registros, {df['well_id'].nunique()} pozos")

    prep = WellPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = prep.prepare_supervised(df)
    print(f"  Features: {len(feature_names)} | Train: {len(X_train)} | Test: {len(X_test)}")

    viz = WellVisualizer()
    viz.plot_production_curves(df)
    viz.plot_decline_analysis(df)
    viz.plot_correlation(df)
    viz.plot_well_map(df)
    print("  Plots generados en outputs/plots/")

    print("\n  Entrenando modelos de series de tiempo...")
    production_col = "initial_oil_rate_bbl_d"
    if production_col not in df.columns:
        production_col = df.select_dtypes(include=[np.number]).columns[0]
    production_series = df.groupby("month")[production_col].mean().sort_index()

    models = {}
    print("\n  [1/3] Entrenando Prophet...")
    models["prophet"] = train_prophet(production_series)
    print("        Prophet entrenado")

    print("  [2/3] Entrenando ARIMA (auto_arima)...")
    models["arima"] = train_arima(production_series)
    print("        ARIMA entrenado")

    print("  [3/3] Entrenando Holt-Winters...")
    models["holt_winters"] = train_holt_winters(production_series)
    print("        Holt-Winters entrenado")

    print("\n  Evaluando modelos...")
    evaluator = ForecastEvaluator()
    best_name, best_model = None, None

    for name, model in models.items():
        try:
            if name == "prophet":
                future = pd.DataFrame({
                    "ds": pd.date_range(start="2020-01-01", periods=len(production_series), freq="D"),
                })
                forecast = model.predict(future)
                y_pred = forecast["yhat"].values
            elif name == "arima":
                y_pred = model.predict(n_periods=len(production_series))
            else:
                y_pred = model.fittedvalues

            metrics = evaluator.evaluate(production_series.values, y_pred, name)
            print(f"  {name:<25} R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.2f}")
            if best_name is None or metrics["R2"] > evaluator.results[best_name]["R2"]:
                best_name, best_model = name, model
        except Exception as e:
            print(f"  {name:<25} ERROR: {e}")

    evaluator.print_report()

    joblib.dump({
        "models": models,
        "production_series": production_series,
    }, "outputs/models/time_series_models.pkl")
    print(f"\n  Mejor modelo: {best_name}")
    print("  Modelos guardados en outputs/models/time_series_models.pkl")

    print("\n  Entrenamiento completado exitosamente!")
    return best_name, evaluator


if __name__ == "__main__":
    main()
