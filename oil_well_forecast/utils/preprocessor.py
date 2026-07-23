"""Preprocesamiento de datos de produccion de pozos."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "depth_m", "permeability_md", "porosity", "net_thickness_m",
    "initial_pressure_psi", "initial_oil_rate_bbl_d", "b_factor", "decline_rate",
    "water_cut_init", "cumulative_oil_bbl",
    "flowing_bhp_psi", "tubing_head_psi",
    "choke_size_64", "pump_speed_spm", "temperature_f", "month",
]

DERIVED_FEATURES = [
    "drawdown_psi", "drawdown_ratio", "pressure_ratio",
    "decline_cumulative_ratio", "thickness_productivity",
]


class WellPreprocessor:
    def __init__(self, scaler_type="robust", test_size=0.2, random_state=42):
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self._fitted = False

    def _build_derived(self, df):
        df = df.copy()
        eps = 1e-6
        df["drawdown_psi"] = df["initial_pressure_psi"] - df["flowing_bhp_psi"]
        df["drawdown_ratio"] = df["flowing_bhp_psi"] / (df["initial_pressure_psi"] + eps)
        df["pressure_ratio"] = df["tubing_head_psi"] / (df["initial_pressure_psi"] + eps)
        df["decline_cumulative_ratio"] = df["decline_rate"] / (df["cumulative_oil_bbl"] + eps) * 1e6
        df["thickness_productivity"] = df["net_thickness_m"] * df["permeability_md"]
        return df

    def fit(self, df):
        df = self._build_derived(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(df[all_features])
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.transform(df[all_features]))
        for col in df.select_dtypes(include="object").columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].astype(str))
        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        df = self._build_derived(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES
        X = self.imputer.transform(df[all_features])
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=all_features, index=df.index)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def prepare_supervised(self, df, target_col="oil_rate_bbl_d", group_col="well_id"):
        df = self._build_derived(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES
        X = df[all_features].fillna(0)
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test, list(X.columns)

    def extract_features(self, df):
        df = self._build_derived(df)
        all_features = NUMERIC_FEATURES + DERIVED_FEATURES
        return df[all_features].fillna(0)

    def prepare_forecast(self, df, target_col="oil_rate_bbl_d"):
        df = df.copy()
        numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64") and c != target_col]
        X = df[numeric_cols].fillna(0)
        y = df[target_col].values if target_col in df.columns else None
        return X, y, numeric_cols

    def get_feature_names(self):
        return NUMERIC_FEATURES + DERIVED_FEATURES
