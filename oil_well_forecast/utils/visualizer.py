"""Visualizacion de datos de produccion de pozos."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


RES_COLORS = {"arenisca": "#27ae60", "caliza": "#2980b9", "dolomita": "#e67e22", "esquisto": "#9b59b6"}
STATUS_COLORS = {"activo": "#2ecc71", "productivo": "#3498db", "bombeo": "#e67e22", "cerrado": "#e74c3c"}


class WellVisualizer:
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def plot_production_curves(self, df, n_wells=10):
        wells = df["well_id"].unique()[:n_wells]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for well_id in wells:
            w = df[df["well_id"] == well_id].sort_values("month")
            axes[0].plot(w["month"], w["oil_rate_bbl_d"], alpha=0.7, linewidth=1.5)
            axes[1].plot(w["month"], w["gas_rate_mcf_d"], alpha=0.7, linewidth=1.5)
            axes[2].plot(w["month"], w["water_rate_bbl_d"], alpha=0.7, linewidth=1.5)
        axes[0].set_title("Tasa de Aceite (bbl/d)", fontweight="bold")
        axes[1].set_title("Tasa de Gas (mcf/d)", fontweight="bold")
        axes[2].set_title("Tasa de Agua (bbl/d)", fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Mes")
            ax.set_ylabel("Tasa")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Curvas de Produccion - Primeros {n_wells} Pozos", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "01_production_curves")

    def plot_decline_analysis(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for res_type, color in RES_COLORS.items():
            subset = df[df["reservoir_type"] == res_type]
            if len(subset) > 0:
                avg_curve = subset.groupby("month")["oil_rate_bbl_d"].mean()
                axes[0].plot(avg_curve.index, avg_curve.values, label=res_type, color=color, linewidth=2)
        axes[0].set_title("Declinacion Promedio por Tipo de Reservorio", fontweight="bold")
        axes[0].set_xlabel("Mes")
        axes[0].set_ylabel("Tasa de Aceite (bbl/d)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        for res_type, color in RES_COLORS.items():
            subset = df[df["reservoir_type"] == res_type]
            if len(subset) > 0:
                axes[1].hist(subset.groupby("well_id")["decline_rate"].first(), bins=20, alpha=0.6, label=res_type, color=color)
        axes[1].set_title("Distribucion de Tasa de Declinacion", fontweight="bold")
        axes[1].set_xlabel("Decline Rate")
        axes[1].set_ylabel("Frecuencia")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "02_decline_analysis")

    def plot_correlation(self, df):
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 6})
        ax.set_title("Matriz de Correlacion - Produccion de Pozos", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "03_correlation_matrix")

    def plot_well_map(self, df):
        fig, ax = plt.subplots(figsize=(12, 8))
        well_info = df.groupby("well_id").first().reset_index()
        for res_type, color in RES_COLORS.items():
            subset = well_info[well_info["reservoir_type"] == res_type]
            ax.scatter(subset["longitude"], subset["latitude"], c=color, label=res_type, alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
        ax.set_title("Mapa de Pozos por Tipo de Reservorio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "04_well_map")

    def plot_forecast_vs_actual(self, y_true, y_pred, title="Pronostico vs Real"):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].scatter(y_true, y_pred, alpha=0.4, s=15, c="#3498db", edgecolors="#2c3e50", linewidth=0.3)
        lims = [min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1]
        axes[0].plot(lims, lims, "--", color="#e74c3c", linewidth=2, label="Prediccion perfecta")
        axes[0].set_xlabel("Real")
        axes[0].set_ylabel("Predicho")
        axes[0].set_title("Dispersion", fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, c="#e67e22", edgecolors="#2c3e50", linewidth=0.3)
        axes[1].axhline(y=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Predicho")
        axes[1].set_ylabel("Residuo")
        axes[1].set_title("Residuos", fontweight="bold")
        axes[1].grid(alpha=0.3)

        axes[2].hist(residuals, bins=40, color="#2ecc71", edgecolor="#2c3e50", alpha=0.7)
        axes[2].axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2)
        axes[2].set_xlabel("Residuo")
        axes[2].set_ylabel("Frecuencia")
        axes[2].set_title("Distribucion de Residuos", fontweight="bold")
        axes[2].grid(alpha=0.3)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "05_forecast_results")

    def plot_model_comparison(self, results):
        models = list(results.keys())
        metrics = ["R2", "MAE", "RMSE"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for i, metric in enumerate(metrics):
            vals = [results[m][metric] for m in models]
            bars = axes[i].barh(models, vals, color=colors, edgecolor="#2c3e50")
            axes[i].set_title(metric, fontweight="bold")
            axes[i].grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)
        fig.suptitle("Comparacion de Modelos", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self._save(fig, "06_model_comparison")
