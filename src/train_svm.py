"""
Per-series SVR (Support Vector Regression), one model per series.

SVR has no built-in trend/seasonality decomposition, so weekly
seasonality and trend are engineered explicitly: a linear trend index
and a sin/cos encoding of day-of-week (cyclical, so Sunday and Monday
are close together instead of 6 apart like a raw 1-7 label would be).
SNAP is passed as a regular feature, same role it plays in ARIMA.

Features are standardized (SVR is scale-sensitive) via a
StandardScaler -> SVR pipeline.
"""

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("mlflow").setLevel(logging.WARNING)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
TEST_HORIZON = 28

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "m5-svm-forecasting"

FEATURES = ["trend", "dow_sin", "dow_cos", "snap"]


def train_test_split(long_df: pd.DataFrame):
    dates = sorted(long_df["date"].unique())
    split_date = dates[-TEST_HORIZON]
    return long_df[long_df["date"] < split_date].copy(), long_df[long_df["date"] >= split_date].copy()


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def add_features(df: pd.DataFrame, train_start: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["trend"] = (df["date"] - train_start).dt.days
    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    state = df["state_id"].iloc[0]
    df["snap"] = df[f"snap_{state}"]
    return df


def fit_one_series(series_id: str, s_train: pd.DataFrame, s_test: pd.DataFrame, train_start: pd.Timestamp):
    s_train = add_features(s_train, train_start)
    s_test = add_features(s_test, train_start)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.5)),
    ])
    model.fit(s_train[FEATURES], s_train["sales"])

    yhat = np.clip(model.predict(s_test[FEATURES]), a_min=0, a_max=None)
    series_mape = mape(s_test["sales"].to_numpy(), yhat)
    return series_id, model, yhat, series_mape


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    long_df = pd.read_csv(f"{PROCESSED_DIR}/subset_long.csv", parse_dates=["date"])
    train, test = train_test_split(long_df)
    train_start = train["date"].min()
    series_ids = sorted(long_df["id"].unique())
    print(f"Training SVR for {len(series_ids)} series...")

    per_series_rows = []
    all_forecasts = []
    run_name = f"svm_training_{pd.Timestamp.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_series", len(series_ids))
        mlflow.log_param("test_horizon_days", TEST_HORIZON)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("kernel", "rbf")

        for i, series_id in enumerate(series_ids, 1):
            s_train = train[train["id"] == series_id].sort_values("date")
            s_test = test[test["id"] == series_id].sort_values("date")

            sid, model, yhat, series_mape = fit_one_series(series_id, s_train, s_test, train_start)
            per_series_rows.append({"id": sid, "mape": series_mape})

            out = s_test[["id", "date", "sales"]].copy()
            out["forecast"] = yhat
            all_forecasts.append(out)

            with mlflow.start_run(run_name=series_id, nested=True):
                mlflow.log_param("series_id", series_id)
                mlflow.log_metric("mape", series_mape)
                mlflow.sklearn.log_model(model, name="model", registered_model_name=f"svm_{series_id}")

            if i % 10 == 0 or i == len(series_ids):
                print(f"  [{i}/{len(series_ids)}] fitted -- last series MAPE: {series_mape:.1f}%")

        per_series = pd.DataFrame(per_series_rows)
        forecasts = pd.concat(all_forecasts, ignore_index=True)

        mean_per_series_mape = per_series["mape"].mean()
        daily = forecasts.groupby("date")[["sales", "forecast"]].sum().reset_index()
        aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())

        mlflow.log_metric("mean_per_series_mape", mean_per_series_mape)
        mlflow.log_metric("aggregate_mape", aggregate_mape)

        print(f"\nSVM -- Mean per-series MAPE: {mean_per_series_mape:.2f}%  "
              f"Aggregate MAPE: {aggregate_mape:.2f}%")

        per_series.to_csv(f"{RESULTS_DIR}/svm_mape_per_series.csv", index=False)
        forecasts.to_csv(f"{RESULTS_DIR}/svm_forecasts.csv", index=False)
        pd.DataFrame([{
            "model": "svm", "mean_per_series_mape": mean_per_series_mape,
            "aggregate_mape": aggregate_mape, "test_horizon_days": TEST_HORIZON,
            "n_series": len(series_ids),
        }]).to_csv(f"{RESULTS_DIR}/svm_summary.csv", index=False)
        print(f"Saved to {RESULTS_DIR}/svm_summary.csv")


if __name__ == "__main__":
    main()
