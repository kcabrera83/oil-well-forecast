"""Evaluacion de modelos de pronostico de produccion."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)


class ForecastEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, y_true, y_pred, model_name="model"):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}
        self.results[model_name] = metrics
        return metrics

    def compare(self):
        return self.results

    def best_model(self, metric="R2"):
        if not self.results:
            return None
        if metric in ("MAE", "RMSE", "MAPE_%"):
            return min(self.results, key=lambda k: self.results[k][metric])
        return max(self.results, key=lambda k: self.results[k][metric])

    def print_report(self):
        print(f"\n{'='*65}")
        print(f"  {'Modelo':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10} {'MAPE%':>10}")
        print(f"{'='*65}")
        for name, m in self.results.items():
            print(f"  {name:<25} {m['MAE']:>10.2f} {m['RMSE']:>10.2f} {m['R2']:>10.4f} {m['MAPE_%']:>10.2f}")
        print(f"{'='*65}")
        best = self.best_model()
        print(f"  Mejor modelo: {best} (R2={self.results[best]['R2']:.4f})")
