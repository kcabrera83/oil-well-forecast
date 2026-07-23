#!/usr/bin/env python3
"""Script de prediccion de produccion de pozos."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oil_well_forecast.data_generator import WellDataGenerator
from oil_well_forecast.utils.preprocessor import WellPreprocessor
from oil_well_forecast.models.production_predictor import ProductionPredictor
from oil_well_forecast.models.decline_analyzer import DeclineAnalyzer


def main():
    print("=" * 60)
    print("  PREDICCION - Sistema de Pronostico de Produccion")
    print("=" * 60)

    gen = WellDataGenerator(seed=99)
    df = gen.generate(n_wells=10, n_months=24)

    last_records = df.groupby("well_id").tail(6)
    print(f"  Pozos a predecir: {last_records['well_id'].nunique()}")

    model = ProductionPredictor.load("outputs/models/production_predictor.pkl")
    prep = WellPreprocessor()

    analyzer = DeclineAnalyzer()
    predictions = []

    for well_id, group in last_records.groupby("well_id"):
        g = group.sort_values("month")
        X = prep.extract_features(g)
        pred = model.predict(X)
        qi = g.iloc[0]["oil_rate_bbl_d"]
        months = g["month"].values
        rates = g["oil_rate_bbl_d"].values
        decline_info = analyzer.fit_decline(months, rates)
        eur = analyzer.estimate_eur(
            decline_info["qi"], decline_info["di"], decline_info["b"]
        ) if decline_info.get("qi", 0) > 0 else 0

        entry = {
            "well_id": well_id,
            "current_rate": round(float(g.iloc[-1]["oil_rate_bbl_d"]), 2),
            "predicted_next": round(float(pred.mean()), 2),
            "decline_model": decline_info.get("model", "unknown"),
            "decline_r2": round(decline_info.get("r2", 0), 4),
            "estimated_eur_bbl": round(eur, 0),
        }
        predictions.append(entry)
        print(f"  {well_id}: actual={entry['current_rate']:.1f} bbl/d "
              f"predicho={entry['predicted_next']:.1f} bbl/d "
              f"EUR={entry['estimated_eur_bbl']:.0f} bbl "
              f"({entry['decline_model']}, R2={entry['decline_r2']:.4f})")

    output_path = Path("outputs/predictions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Predicciones guardadas en {output_path}")
    print("  Prediccion completada exitosamente!")
    return predictions


if __name__ == "__main__":
    main()
