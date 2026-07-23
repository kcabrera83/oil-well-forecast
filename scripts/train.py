#!/usr/bin/env python3
"""Script de entrenamiento de modelos de produccion de pozos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oil_well_forecast.data_generator import WellDataGenerator
from oil_well_forecast.utils.preprocessor import WellPreprocessor
from oil_well_forecast.utils.metrics import ForecastEvaluator
from oil_well_forecast.utils.visualizer import WellVisualizer
from oil_well_forecast.models.production_predictor import ProductionPredictor


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

    print("\n  Entrenando modelos...")
    all_models = ProductionPredictor.train_all(X_train, y_train)

    evaluator = ForecastEvaluator()
    best_name, best_model = None, None

    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        metrics = evaluator.evaluate(y_test, y_pred, name)
        print(f"  {name:<25} R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.2f}")
        if best_name is None or metrics["R2"] > evaluator.results[best_name]["R2"]:
            best_name, best_model = name, model

    evaluator.print_report()
    best_model.save("outputs/models/production_predictor.pkl")
    print(f"\n  Mejor modelo guardado: {best_name}")

    y_pred_best = best_model.predict(X_test)
    viz.plot_forecast_vs_actual(y_test, y_pred_best, title=f"Pronostico - {best_name}")
    viz.plot_model_comparison(evaluator.compare())

    print("\n  Entrenamiento completado exitosamente!")
    return best_name, evaluator


if __name__ == "__main__":
    main()
