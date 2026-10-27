"""Shared helpers used by every training/eval script (baseline, SVM, ARIMA, ML models).

Previously this exact train/test split and MAPE formula were copy-pasted
across 4 files; kept here so the test horizon and MAPE definition only ever
change in one place.
"""
import numpy as np
import pandas as pd

TEST_HORIZON = 28

# MLflow is DagsHub-hosted, not a self-managed server -- DagsHub gives every
# repo a free MLflow-compatible tracking server + model registry (and UI)
# at this URL. Auth is HTTP basic, via the MLFLOW_TRACKING_USERNAME/
# MLFLOW_TRACKING_PASSWORD env vars (password = a DagsHub access token),
# which the mlflow client reads automatically -- no code needed for that
# part, just those two env vars present wherever this URI is used.
MLFLOW_TRACKING_URI_DEFAULT = "https://dagshub.com/tanmoyghosh704-lang/walmart-mlops-forecasting.mlflow"

CHAMPION_ALIAS = "champion"

# Registered-model names eligible for the "champion" alias -- global models
# trained on the full engineered feature set (see features.py). SVM/ARIMA are
# per-series models with no single name to alias, so they're excluded by
# design (see promote_champion.py).
PROMOTABLE_MODELS = {
    "linear_regression": "sales_linear_regression",
    "random_forest": "sales_random_forest",
    "xgboost": "sales_xgboost",
    "lightgbm": "sales_lightgbm",
}


def train_test_split(long_df: pd.DataFrame, horizon: int = TEST_HORIZON):
    dates = sorted(long_df["date"].unique())
    split_date = dates[-horizon]
    train = long_df[long_df["date"] < split_date].copy()
    test = long_df[long_df["date"] >= split_date].copy()
    return train, test


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)
