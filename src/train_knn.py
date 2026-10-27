import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import TEST_HORIZON, MLFLOW_TRACKING_URI_DEFAULT, mape, train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("mlflow").setLevel(logging.WARNING)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT)
MLFLOW_EXPERIMENT = "m5-knn-forecasting"

FEATURES = ["trend", "dow_sin", "dow_cos", "snap"]

# 7 neighbors = one seasonal week, matching the dow_sin/dow_cos encoding
# already used for weekly seasonality; distance-weighting lets the closest
# (most similar trend/day-of-week/SNAP) days dominate the average instead
# of all 7 counting equally.
N_NEIGHBORS = 7


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
        ("knn", KNeighborsRegressor(n_neighbors=N_NEIGHBORS, weights="distance")),
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
    n_jobs = max(1, joblib.cpu_count() - 2)
    print(f"Training KNN regressors for {len(series_ids)} series in parallel "
          f"(using {n_jobs} of {joblib.cpu_count()} CPUs)...")

    jobs = [
        (sid, train[train["id"] == sid].sort_values("date"),
         test[test["id"] == sid].sort_values("date"))
        for sid in series_ids
    ]
    fitted = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(fit_one_series)(sid, s_train, s_test, train_start)
        for sid, s_train, s_test in jobs
    )

    per_series_rows = []
    all_forecasts = []
    run_name = f"knn_training_{pd.Timestamp.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_series", len(series_ids))
        mlflow.log_param("test_horizon_days", TEST_HORIZON)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("weights", "distance")

        for series_id, model, yhat, series_mape in fitted:
            per_series_rows.append({"id": series_id, "mape": series_mape})

            s_test = test[test["id"] == series_id].sort_values("date")
            out = s_test[["id", "date", "sales"]].copy()
            out["forecast"] = yhat
            all_forecasts.append(out)

            with mlflow.start_run(run_name=series_id, nested=True):
                mlflow.log_param("series_id", series_id)
                mlflow.log_metric("mape", series_mape)
                mlflow.sklearn.log_model(model, name="model", registered_model_name=f"knn_{series_id}")

        per_series = pd.DataFrame(per_series_rows)
        forecasts = pd.concat(all_forecasts, ignore_index=True)

        mean_per_series_mape = per_series["mape"].mean()
        daily = forecasts.groupby("date")[["sales", "forecast"]].sum().reset_index()
        aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())

        mlflow.log_metric("mean_per_series_mape", mean_per_series_mape)
        mlflow.log_metric("aggregate_mape", aggregate_mape)

        print(f"\nKNN -- Mean per-series MAPE: {mean_per_series_mape:.2f}%  "
              f"Aggregate MAPE: {aggregate_mape:.2f}%")

        per_series.to_csv(f"{RESULTS_DIR}/knn_mape_per_series.csv", index=False)
        forecasts.to_csv(f"{RESULTS_DIR}/knn_forecasts.csv", index=False)
        pd.DataFrame([{
            "model": "knn", "mean_per_series_mape": mean_per_series_mape,
            "aggregate_mape": aggregate_mape, "test_horizon_days": TEST_HORIZON,
            "n_series": len(series_ids),
        }]).to_csv(f"{RESULTS_DIR}/knn_summary.csv", index=False)
        print(f"Saved to {RESULTS_DIR}/knn_summary.csv")


if __name__ == "__main__":
    main()
