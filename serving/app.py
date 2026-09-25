"""
FastAPI serving layer.

- GET /health -- liveness check.
- GET /predict -- loads the SVR model for a given store-item series from
  the MLflow Model Registry and returns a forecast for the requested
  horizon.
- GET /metrics -- Prometheus scrape endpoint: request latency/count (via
  prometheus-fastapi-instrumentator) plus a custom prediction-drift metric.

Forecast window: models were trained on data through 2016-03-27 and
evaluated through 2016-04-24 (see src/train_svm.py). The `snap` feature
those models need for future dates comes from calendar.csv, which is
known through 2016-06-19 (SNAP eligibility is a published schedule, not
something we forecast). /predict serves forecasts starting the day after
evaluation ended, capped at how far the calendar actually extends.

Drift metric: this is a forecast-serving API, not a live feature-scoring
one, so there's no incoming raw feature vector to compare against a
training distribution directly. Instead: a rolling window of recently
served predictions is compared against the full historical sales
distribution via a two-sample KS test. A significant shift is a signal
worth investigating (model going stale, a genuine regime change, or a
bug upstream).
"""

import os
from collections import deque
from datetime import timedelta
from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from scipy.stats import ks_2samp

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
CALENDAR_PATH = "data/raw/calendar.csv"
SUBSET_PATH = "data/processed/subset_long.csv"
FEATURES = ["trend", "dow_sin", "dow_cos", "snap"]

app = FastAPI(title="M5 Forecasting API")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["date"]).set_index("date")
_subset = pd.read_csv(SUBSET_PATH, parse_dates=["date"])
_valid_series_ids = set(_subset["id"].unique())
_train_start = _subset["date"].min()
_forecast_start = _subset["date"].max() + timedelta(days=1)
_forecast_end = _calendar.index.max()

_training_sales_distribution = _subset["sales"].to_numpy()
_recent_predictions = deque(maxlen=200)
_MIN_SAMPLES_FOR_DRIFT_CHECK = 30

drift_ks_statistic = Gauge(
    "prediction_drift_ks_statistic",
    "KS-test statistic comparing recent served predictions to the training sales distribution",
)
drift_ks_pvalue = Gauge(
    "prediction_drift_ks_pvalue",
    "KS-test p-value for the same comparison (low p-value = distributions likely differ)",
)


def update_drift_metric():
    if len(_recent_predictions) < _MIN_SAMPLES_FOR_DRIFT_CHECK:
        return
    statistic, pvalue = ks_2samp(np.array(_recent_predictions), _training_sales_distribution)
    drift_ks_statistic.set(statistic)
    drift_ks_pvalue.set(pvalue)


Instrumentator().instrument(app).expose(app)


@lru_cache(maxsize=None)
def load_model(series_id: str):
    return mlflow.pyfunc.load_model(f"models:/svm_{series_id}/latest")


def build_future(series_id: str, horizon: int) -> pd.DataFrame:
    state = series_id.split("_")[-3]  # e.g. FOODS_1_218_TX_2_validation -> "TX"
    dates = pd.date_range(_forecast_start, periods=horizon)
    trend = (dates - _train_start).days
    dow = dates.dayofweek
    return pd.DataFrame({
        "trend": trend,
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "snap": _calendar.loc[dates, f"snap_{state}"].to_numpy(),
    }, index=dates)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "n_series_available": len(_valid_series_ids),
        "forecast_window": {
            "start": str(_forecast_start.date()),
            "end": str(_forecast_end.date()),
        },
    }


@app.get("/predict")
def predict(series_id: str, horizon: int = 7):
    if series_id not in _valid_series_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown series_id '{series_id}'. Must be one of the "
                    f"{len(_valid_series_ids)} series this project trained on.",
        )

    max_horizon = (_forecast_end - _forecast_start).days + 1
    if not (1 <= horizon <= max_horizon):
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be between 1 and {max_horizon} "
                    f"(limited by known calendar/SNAP data through {_forecast_end.date()}).",
        )

    try:
        model = load_model(series_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model for '{series_id}' from MLflow registry: {exc}",
        )

    future = build_future(series_id, horizon)
    yhat = np.clip(np.asarray(model.predict(future[FEATURES])).ravel(), a_min=0, a_max=None)

    _recent_predictions.extend(yhat.tolist())
    update_drift_metric()

    return {
        "series_id": series_id,
        "horizon": horizon,
        "forecast": [
            {"date": str(d.date()), "yhat": round(float(y), 2)}
            for d, y in zip(future.index, yhat)
        ],
    }
