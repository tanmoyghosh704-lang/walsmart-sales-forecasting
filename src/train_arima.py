"""
Per-series SARIMA (seasonal ARIMA), one model per series -- like Prophet,
unlike the global ML models in train_ml_models.py. ARIMA has no natural
"global across series" form (it models a single series' own
autocorrelation structure), so per-series is the only sensible approach
here, not a stylistic choice.

- seasonal=True, m=7: EDA found real weekly seasonality (weekend spikes),
  so plain (non-seasonal) ARIMA would systematically miss it.
- snap passed as an exogenous regressor (X=), matching Prophet's
  add_regressor("snap") -- same reasoning: it's a real, known-in-advance
  signal EDA found, not something we'd need to forecast.
- auto_arima's order search is the slow part (~35s/series measured
  single-threaded). Parallelized fitting across all CPU cores via
  joblib, since each series is fit completely independently -- MLflow
  logging itself stays sequential in the main process afterward, since
  concurrent writers to nested runs across processes is asking for
  trouble.
"""

import logging
import os
import sys
import warnings

import joblib
import mlflow
import mlflow.pmdarima
import numpy as np
import pandas as pd
import pmdarima as pm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
TEST_HORIZON = 28

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "m5-arima-forecasting"


def train_test_split(long_df: pd.DataFrame):
    dates = sorted(long_df["date"].unique())
    split_date = dates[-TEST_HORIZON]
    train = long_df[long_df["date"] < split_date].copy()
    test = long_df[long_df["date"] >= split_date].copy()
    return train, test


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def snap_column(row_df: pd.DataFrame) -> pd.Series:
    state = row_df["state_id"].iloc[0]
    return row_df[[f"snap_{state}"]].rename(columns={f"snap_{state}": "snap"})


def fit_one_series(series_id: str, s_train: pd.DataFrame, s_test: pd.DataFrame):
    model = pm.auto_arima(
        s_train["sales"],
        X=snap_column(s_train),
        seasonal=True, m=7,
        max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )
    yhat = model.predict(n_periods=TEST_HORIZON, X=snap_column(s_test))
    yhat = np.clip(np.asarray(yhat), a_min=0, a_max=None)
    actual = s_test["sales"].to_numpy()
    series_mape = mape(actual, yhat)
    return series_id, model, yhat, series_mape


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    long_df = pd.read_csv(f"{PROCESSED_DIR}/subset_long.csv", parse_dates=["date"])
    train, test = train_test_split(long_df)
    series_ids = sorted(long_df["id"].unique())
    print(f"Fitting SARIMA for {len(series_ids)} series in parallel "
          f"({joblib.cpu_count()} CPUs available)...")

    jobs = [
        (sid, train[train["id"] == sid].sort_values("date"),
         test[test["id"] == sid].sort_values("date"))
        for sid in series_ids
    ]
    results = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(fit_one_series)(sid, s_train, s_test) for sid, s_train, s_test in jobs
    )

    per_series_rows = []
    all_forecasts = []
    run_name = f"arima_training_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_series", len(series_ids))
        mlflow.log_param("test_horizon_days", TEST_HORIZON)
        mlflow.log_param("seasonal_period", 7)
        mlflow.log_param("exogenous", "snap")

        for series_id, model, yhat, series_mape in results:
            per_series_rows.append({"id": series_id, "mape": series_mape})
            s_test = test[test["id"] == series_id].sort_values("date")
            out = s_test[["id", "date", "sales"]].copy()
            out["forecast"] = yhat
            all_forecasts.append(out)

            with mlflow.start_run(run_name=series_id, nested=True):
                mlflow.log_param("series_id", series_id)
                mlflow.log_param("order", str(model.order))
                mlflow.log_param("seasonal_order", str(model.seasonal_order))
                mlflow.log_metric("mape", series_mape)
                mlflow.pmdarima.log_model(
                    model, name="model", registered_model_name=f"arima_{series_id}"
                )

        per_series = pd.DataFrame(per_series_rows)
        forecasts = pd.concat(all_forecasts, ignore_index=True)

        mean_per_series_mape = per_series["mape"].mean()
        daily = forecasts.groupby("date")[["sales", "forecast"]].sum().reset_index()
        aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())

        mlflow.log_metric("mean_per_series_mape", mean_per_series_mape)
        mlflow.log_metric("aggregate_mape", aggregate_mape)

        print(f"\nSARIMA -- Mean per-series MAPE: {mean_per_series_mape:.2f}%  "
              f"Aggregate MAPE: {aggregate_mape:.2f}%")

        per_series.to_csv(f"{RESULTS_DIR}/arima_mape_per_series.csv", index=False)
        forecasts.to_csv(f"{RESULTS_DIR}/arima_forecasts.csv", index=False)
        summary = pd.DataFrame([{
            "model": "arima", "mean_per_series_mape": mean_per_series_mape,
            "aggregate_mape": aggregate_mape, "test_horizon_days": TEST_HORIZON,
            "n_series": len(series_ids),
        }])
        summary.to_csv(f"{RESULTS_DIR}/arima_summary.csv", index=False)
        print(f"Saved to {RESULTS_DIR}/arima_summary.csv")


if __name__ == "__main__":
    main()
