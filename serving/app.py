import os
import sys
import time
from collections import deque
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features  # noqa: E402  (path must be set up before this import)
from common import CHAMPION_ALIAS, MLFLOW_TRACKING_URI_DEFAULT, PROMOTABLE_MODELS  # noqa: E402

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT)
RAW_DIR = "data/raw"
CALENDAR_PATH = f"{RAW_DIR}/calendar.csv"
SUBSET_PATH = "data/processed/subset_long.csv"
ENCODERS_PATH = "data/processed/encoders.json"
REFERENCE_FEATURES_PATH = "data/processed/ml_features.csv"

# How often to re-check the registry for a newly-promoted champion and
# re-read the latest feature-engineering snapshot, without restarting the
# container. Keeps "retrain -> promote" picking up in production on a
# bounded delay instead of needing an explicit reload signal.
CACHE_TTL_SECONDS = float(os.environ.get("MODEL_CACHE_TTL_SECONDS", "300"))

DRIFT_BUFFER_SIZE = int(os.environ.get("DRIFT_BUFFER_SIZE", "300"))
DRIFT_MIN_SAMPLES = 30
DRIFT_KS_THRESHOLD = float(os.environ.get("DRIFT_KS_THRESHOLD", "0.3"))

app = FastAPI(title="M5 Forecasting API")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["date"]).set_index("date")
_subset = pd.read_csv(SUBSET_PATH, parse_dates=["date"])
_valid_series_ids = set(_subset["id"].unique())
_train_start = _subset["date"].min()
_forecast_start = _subset["date"].max() + timedelta(days=1)
_forecast_end = _calendar.index.max()
_prices = features.load_prices(_subset, raw_dir=RAW_DIR)
_encoders = features.load_encoders(ENCODERS_PATH)

# Cap horizon at LAG: lag_28/rolling features for day T need real sales at
# T-28..T-1. That only holds while every forecast day is within LAG days of
# the last known actuals -- beyond that we'd need to feed the model's own
# predictions back in as "sales" (recursive forecasting), which this
# project doesn't implement.
_max_horizon = min((_forecast_end - _forecast_start).days + 1, features.LAG)

_champion_cache = {"model": None, "family": None, "name": None, "version": None, "loaded_at": 0.0}
_reference_cache = {"df": None, "loaded_at": 0.0}

_drift_buffers = {feat: deque(maxlen=DRIFT_BUFFER_SIZE) for feat in features.NUMERIC_FEATURES}

feature_drift_ks_statistic = Gauge(
    "feature_drift_ks_statistic",
    "KS-test statistic per feature: recent /predict inputs vs the last feature-engineering snapshot",
    ["feature"],
)
prediction_drift_score = Gauge(
    "prediction_drift_score",
    "Mean per-feature KS statistic across all numeric features (0 = no drift)",
)
prediction_drift_alert = Gauge(
    "prediction_drift_alert",
    f"1 if prediction_drift_score > {DRIFT_KS_THRESHOLD} (DRIFT_KS_THRESHOLD), else 0",
)

Instrumentator().instrument(app).expose(app)


def get_reference_features() -> pd.DataFrame:
    now = time.monotonic()
    if _reference_cache["df"] is None or now - _reference_cache["loaded_at"] > CACHE_TTL_SECONDS:
        _reference_cache["df"] = pd.read_csv(REFERENCE_FEATURES_PATH)
        _reference_cache["loaded_at"] = now
    return _reference_cache["df"]


def get_champion():
    """Resolve + load whichever promotable model currently carries the
    'champion' alias, caching it for CACHE_TTL_SECONDS so a fresh
    promotion is picked up without restarting this container.
    """
    now = time.monotonic()
    if _champion_cache["model"] is not None and now - _champion_cache["loaded_at"] < CACHE_TTL_SECONDS:
        return _champion_cache

    client = MlflowClient()
    found = []
    for family, registered_name in PROMOTABLE_MODELS.items():
        try:
            mv = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
        except Exception:
            continue
        found.append((family, registered_name, mv.version))

    if not found:
        raise RuntimeError(
            f"No registered model currently carries the '{CHAMPION_ALIAS}' alias. "
            f"Run src/promote_champion.py after a training run completes."
        )

    family, name, version = found[0]
    model = mlflow.pyfunc.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
    _champion_cache.update(model=model, family=family, name=name, version=version, loaded_at=now)
    return _champion_cache


def record_drift_and_update_metrics(X: pd.DataFrame):
    for feat in features.NUMERIC_FEATURES:
        _drift_buffers[feat].extend(X[feat].tolist())

    reference = get_reference_features()
    ks_stats = []
    for feat in features.NUMERIC_FEATURES:
        buf = _drift_buffers[feat]
        if len(buf) < DRIFT_MIN_SAMPLES:
            continue
        statistic, _ = ks_2samp(np.array(buf), reference[feat].dropna().to_numpy())
        feature_drift_ks_statistic.labels(feature=feat).set(statistic)
        ks_stats.append(statistic)

    if ks_stats:
        score = float(np.mean(ks_stats))
        prediction_drift_score.set(score)
        prediction_drift_alert.set(1.0 if score > DRIFT_KS_THRESHOLD else 0.0)


def prepare_model_input(X: pd.DataFrame, family: str) -> pd.DataFrame:
    X = X.copy()
    if family == "lightgbm":
        # A category dtype's underlying codes must span the same universe
        # the model was trained on. A naive `.astype("category")` on a
        # single-series request would only see that series' category
        # values, re-numbering codes from 0 and silently feeding the
        # booster the wrong split values. Reconstruct the full training
        # category universe from the persisted encoders instead.
        for col in features.CATEGORICAL_COLS:
            code_col = f"{col}_code"
            n_categories = len(_encoders[col])
            X[code_col] = pd.Categorical(X[code_col], categories=list(range(n_categories)))
    return X


@app.get("/health")
def health():
    info = {
        "status": "ok",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "n_series_available": len(_valid_series_ids),
        "max_horizon_days": _max_horizon,
        "forecast_window": {
            "start": str(_forecast_start.date()),
            "end": str(_forecast_end.date()),
        },
    }
    try:
        champion = get_champion()
        info["champion"] = {
            "family": champion["family"],
            "registered_name": champion["name"],
            "version": champion["version"],
        }
    except RuntimeError as exc:
        info["champion"] = None
        info["champion_error"] = str(exc)
    return info


@app.get("/predict")
def predict(series_id: str, horizon: int = 7):
    if series_id not in _valid_series_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown series_id '{series_id}'. Must be one of the "
                    f"{len(_valid_series_ids)} series this project trained on.",
        )

    if not (1 <= horizon <= _max_horizon):
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be between 1 and {_max_horizon} "
                    f"(limited by known calendar/SNAP data and the model's lag-28 features).",
        )

    try:
        champion = get_champion()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    future = features.build_future_frame(
        series_id, horizon, history=_subset, calendar=_calendar, prices=_prices, encoders=_encoders,
    )
    X = future[features.ALL_FEATURES]
    model_input = prepare_model_input(X, champion["family"])

    try:
        yhat = np.clip(np.asarray(champion["model"].predict(model_input)).ravel(), a_min=0, a_max=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    record_drift_and_update_metrics(X)

    return {
        "series_id": series_id,
        "horizon": horizon,
        "champion": {"family": champion["family"], "version": champion["version"]},
        "forecast": [
            {"date": str(d.date()), "yhat": round(float(y), 2)}
            for d, y in zip(future.index, yhat)
        ],
    }
