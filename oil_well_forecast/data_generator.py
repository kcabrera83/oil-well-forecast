"""Generador de datos sinteticos de produccion de pozos petroliferos."""

import numpy as np
import pandas as pd
from pathlib import Path


class WellDataGenerator:
    """
    Genera datos sinteticos realistas de produccion de pozos petroliferos
    basado en modelos de declinacion curvilinea (Arps) y propiedades geologicas.
    """

    RESERVOIR_TYPES = {
        "arenisca": {"perm_range": (50, 500), "poro_range": (0.15, 0.30), "ip_range": (200, 2000)},
        "caliza": {"perm_range": (10, 200), "poro_range": (0.05, 0.20), "ip_range": (100, 1500)},
        "dolomita": {"perm_range": (5, 100), "poro_range": (0.08, 0.18), "ip_range": (80, 1200)},
        "esquisto": {"perm_range": (0.001, 1), "poro_range": (0.03, 0.10), "ip_range": (50, 800)},
    }

    WELL_STATUS = ["activo", "productivo", "bombeo", "cerrado"]

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_wells=200, n_months=36):
        wells = []
        for well_id in range(n_wells):
            well_data = self._generate_well(well_id, n_months)
            wells.append(well_data)
        return pd.concat(wells, ignore_index=True)

    def _generate_well(self, well_id, n_months):
        res_type = self.rng.choice(
            list(self.RESERVOIR_TYPES.keys()),
            p=[0.40, 0.25, 0.20, 0.15],
        )
        spec = self.RESERVOIR_TYPES[res_type]

        latitude = self.rng.uniform(26.0, 32.0)
        longitude = self.rng.uniform(-101.0, -96.0)
        depth_m = self.rng.uniform(1500, 4500)
        permeability = self.rng.uniform(*spec["perm_range"])
        porosity = self.rng.uniform(*spec["poro_range"])
        thickness = self.rng.uniform(5, 80)
        water_cut_init = self.rng.uniform(0.05, 0.60)
        initial_pressure = self.rng.uniform(150, 400)
        ip_oil = self.rng.uniform(*spec["ip_range"])

        b_factor = self.rng.uniform(0.1, 2.5)
        di = self.rng.uniform(0.01, 0.15)
        d_min = di * self.rng.uniform(0.05, 0.20)

        status = "activo"
        records = []
        cumulative_oil = 0
        cumulative_gas = 0
        cumulative_water = 0

        for month in range(1, n_months + 1):
            if b_factor > 0.001:
                qi_b = ip_oil / ((1 + b_factor * di * (month - 1)) ** (1 / b_factor))
            else:
                qi_b = ip_oil * np.exp(-di * (month - 1))

            qoil = max(qi_b + self.rng.normal(0, qi_b * 0.05), 0)
            gor = self.rng.uniform(100, 2000)
            qgas = qoil * gor / 1000
            wc = min(water_cut_init + self.rng.uniform(0, 0.01) * month, 0.95)
            qwater = qoil * wc / (1 - wc) if wc < 1 else qoil * 10

            cumulative_oil += qoil
            cumulative_gas += qgas
            cumulative_water += qwater

            flowing_bhp = initial_pressure * self.rng.uniform(0.3, 0.6)
            tubing_head_p = self.rng.uniform(10, 80)
            choke_size = self.rng.uniform(4, 64)
            pump_speed = self.rng.uniform(50, 300) if res_type != "esquisto" else 0
            temperature = self.rng.uniform(60, 140)

            status = self._assign_status(qoil, month, wc)

            records.append({
                "well_id": f"W-{well_id:04d}",
                "month": month,
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "depth_m": round(depth_m, 1),
                "reservoir_type": res_type,
                "permeability_md": round(permeability, 3),
                "porosity": round(porosity, 4),
                "net_thickness_m": round(thickness, 1),
                "initial_pressure_psi": round(initial_pressure, 1),
                "initial_oil_rate_bbl_d": round(ip_oil, 2),
                "b_factor": round(b_factor, 3),
                "decline_rate": round(di, 4),
                "water_cut_init": round(water_cut_init, 3),
                "oil_rate_bbl_d": round(qoil, 2),
                "gas_rate_mcf_d": round(qgas, 2),
                "water_rate_bbl_d": round(qwater, 2),
                "gas_oil_ratio": round(gor, 1),
                "cumulative_oil_bbl": round(cumulative_oil, 1),
                "cumulative_gas_mcf": round(cumulative_gas, 1),
                "cumulative_water_bbl": round(cumulative_water, 1),
                "flowing_bhp_psi": round(flowing_bhp, 1),
                "tubing_head_psi": round(tubing_head_p, 1),
                "choke_size_64": round(choke_size, 1),
                "pump_speed_spm": round(pump_speed, 1),
                "temperature_f": round(temperature, 1),
                "well_status": status,
                "recovery_factor": round(min(cumulative_oil / (thickness * porosity * 7758 * 0.8), 0.6), 4),
            })

        return pd.DataFrame(records)

    def _assign_status(self, qoil, month, wc):
        if qoil < 5:
            return "cerrado"
        if wc > 0.90:
            return "bombeo"
        if qoil > 100:
            return "productivo"
        return "activo"

    def generate_forecast_input(self, df_wells, forecast_months=12):
        last_months = df_wells.groupby("well_id").tail(6).copy()
        inputs = []
        for well_id, group in last_months.groupby("well_id"):
            g = group.sort_values("month")
            entry = {
                "well_id": well_id,
                "last_oil_rate": g.iloc[-1]["oil_rate_bbl_d"],
                "last_gas_rate": g.iloc[-1]["gas_rate_mcf_d"],
                "last_water_rate": g.iloc[-1]["water_rate_bbl_d"],
                "last_water_cut": g.iloc[-1]["water_rate_bbl_d"] / max(g.iloc[-1]["oil_rate_bbl_d"], 0.01),
                "oil_trend_3m": g["oil_rate_bbl_d"].diff(3).mean() if len(g) >= 3 else 0,
                "oil_trend_6m": g["oil_rate_bbl_d"].diff(6).mean() if len(g) >= 6 else 0,
                "avg_oil_3m": g["oil_rate_bbl_d"].tail(3).mean(),
                "avg_gas_3m": g["gas_rate_mcf_d"].tail(3).mean(),
                "cumulative_oil": g.iloc[-1]["cumulative_oil_bbl"],
                "month": g.iloc[-1]["month"],
                "depth_m": g.iloc[-1]["depth_m"],
                "reservoir_type": g.iloc[-1]["reservoir_type"],
                "permeability_md": g.iloc[-1]["permeability_md"],
                "porosity": g.iloc[-1]["porosity"],
                "net_thickness_m": g.iloc[-1]["net_thickness_m"],
                "initial_oil_rate_bbl_d": g.iloc[-1]["initial_oil_rate_bbl_d"],
                "b_factor": g.iloc[-1]["b_factor"],
                "decline_rate": g.iloc[-1]["decline_rate"],
                "forecast_months": forecast_months,
            }
            inputs.append(entry)
        return pd.DataFrame(inputs)

    def save(self, df, path="data/well_production.csv"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath


if __name__ == "__main__":
    gen = WellDataGenerator(seed=42)
    df = gen.generate(n_wells=200, n_months=36)
    path = gen.save(df, path="data/well_production.csv")
    print(f"Dataset generado: {len(df)} registros de {df['well_id'].nunique()} pozos")
    print(df.describe())
