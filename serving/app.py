"""
FastAPI serving layer.

- GET /health -- trivial liveness check.
- GET /predict -- loads the Prophet model for a given store-item series
  from the MLflow Model Registry (not a pickle file) and returns a
  forecast for the requested horizon.

Forecast window: models were trained on data through 2016-03-27 and
evaluated through 2016-04-24 (see src/train.py). The `snap` regressor
those models need for future dates comes from calendar.csv, which is
known through 2016-06-19 (SNAP eligibility is a published schedule, not
something we forecast). So /predict serves forecasts starting the day
after evaluation ended, capped at how far the calendar actually extends
-- past that, we'd have to fabricate the regressor, which we don't do.
"""

import logging
import os
from datetime import timedelta
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Env-overridable so the same image works talking to a local dev server
# (http://127.0.0.1:5000) or, from inside Docker Desktop on Windows/Mac,
# the host-reachable alias http://host.docker.internal:5000 -- see
# serving/Dockerfile and README for how this gets set at `docker run` time.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
CALENDAR_PATH = "data/raw/calendar.csv"
SUBSET_PATH = "data/processed/subset_long.csv"

app = FastAPI(title="M5 Prophet Forecasting API")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["date"]).set_index("date")
_subset = pd.read_csv(SUBSET_PATH, parse_dates=["date"])
_valid_series_ids = set(_subset["id"].unique())
_forecast_start = _subset["date"].max() + timedelta(days=1)
_forecast_end = _calendar.index.max()


@lru_cache(maxsize=None)
def load_model(series_id: str):
    """Cached so repeated requests for the same series don't hit the registry every time."""
    return mlflow.pyfunc.load_model(f"models:/prophet_{series_id}/latest")


def build_future(series_id: str, horizon: int) -> pd.DataFrame:
    state = series_id.split("_")[-3]  # e.g. FOODS_1_218_TX_2_validation -> "TX"
    dates = pd.date_range(_forecast_start, periods=horizon)
    snap_col = f"snap_{state}"
    snap_values = _calendar.loc[dates, snap_col].to_numpy()
    return pd.DataFrame({"ds": dates, "snap": snap_values})


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
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    return {
        "series_id": series_id,
        "horizon": horizon,
        "forecast": [
            {"date": str(row.ds.date()), "yhat": round(float(row.yhat), 2)}
            for row in forecast.itertuples()
        ],
    }
