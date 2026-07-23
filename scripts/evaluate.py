#!/usr/bin/env python3
"""Script de evaluacion de modelos de produccion de pozos."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oil_well_forecast.data_generator import WellDataGenerator
from oil_well_forecast.utils.preprocessor import WellPreprocessor
from oil_well_forecast.utils.metrics import ForecastEvaluator
from oil_well_forecast.utils.visualizer import WellVisualizer
from oil_well_forecast.models.production_predictor import ProductionPredictor
from oil_well_forecast.models.decline_analyzer import DeclineAnalyzer
from oil_well_forecast.models.anomaly_detector import WellAnomalyDetector
import numpy as np


def main():
    print("=" * 60)
    print("  EVALUACION - Sistema de Pronostico de Produccion")
    print("=" * 60)

    gen = WellDataGenerator(seed=42)
    df = gen.generate(n_wells=200, n_months=36)

    prep = WellPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = prep.prepare_supervised(df)
    print(f"  Test set: {len(X_test)} muestras")

    print("\n  [1] Evaluacion con cross-validation...")
    all_models = ProductionPredictor.train_all(X_train, y_train)
    evaluator = ForecastEvaluator()
    for name, model in all_models.items():
        cv = model.cross_validate(X_train, y_train, cv=5)
        y_pred = model.predict(X_test)
        metrics = evaluator.evaluate(y_test, y_pred, f"{name}")
        print(f"  {name:<25} CV-R2={cv['mean_r2']:.4f} (+/-{cv['std_r2']:.4f})  Test-R2={metrics['R2']:.4f}")

    evaluator.print_report()

    print("\n  [2] Analisis de declinacion por pozo...")
    analyzer = DeclineAnalyzer()
    decline_results = []
    for well_id, group in df.groupby("well_id"):
        g = group.sort_values("month")
        res = analyzer.fit_decline(g["month"].values, g["oil_rate_bbl_d"].values)
        eur = analyzer.estimate_eur(res["qi"], res["di"], res["b"]) if res.get("qi", 0) > 0 else 0
        decline_results.append({
            "well_id": well_id,
            "model": res.get("model", "unknown"),
            "r2": round(res.get("r2", 0), 4),
            "eur_bbl": round(eur, 0),
        })
    valid = [r for r in decline_results if r["r2"] > 0.8]
    print(f"  Pozos con buen ajuste (R2>0.8): {len(valid)}/{len(decline_results)}")
    print(f"  EUR promedio: {np.mean([r['eur_bbl'] for r in decline_results]):.0f} bbl")

    print("\n  [3] Deteccion de anomalias...")
    detector = WellAnomalyDetector()
    df_anomaly, n_anomalies = detector.detect_anomalies(df)
    summary = detector.get_anomaly_summary(df_anomaly)
    print(f"  Anomalias detectadas: {summary['anomaly_count']} ({summary['anomaly_pct']}%)")

    print("\n  [4] Graficos de evaluacion...")
    viz = WellVisualizer()
    best_name = evaluator.best_model()
    best_model = all_models[best_name]
    y_pred_best = best_model.predict(X_test)
    viz.plot_forecast_vs_actual(y_test, y_pred_best, title=f"Evaluacion Final - {best_name}")
    viz.plot_model_comparison(evaluator.compare())

    eval_report = {
        "best_model": best_name,
        "metrics": evaluator.results[best_name],
        "decline_summary": {
            "good_fit_count": len(valid),
            "total": len(decline_results),
        },
        "anomaly_summary": summary,
    }
    output_path = Path("outputs/evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Reporte guardado en {output_path}")
    print("  Evaluacion completada exitosamente!")
    return evaluator


if __name__ == "__main__":
    main()
