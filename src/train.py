"""
Train one Prophet model per series and compare against the naive baseline.

Design choices, and why:
- Same 28-day test holdout as src/baseline.py, so the MAPE comparison is
  apples-to-apples.
- Holidays: built from calendar.csv's event_name_1/date, passed via
  Prophet's `holidays` parameter (its native mechanism for "sales jump/dip
  on this specific date"). EDA showed this effect is weak for our
  FOODS/HOUSEHOLD-heavy subset, but it's nearly free to include and is
  what the workflow doc calls out as the M5-specific feature to use.
- SNAP as an extra regressor: EDA found a real, consistent effect
  (higher sales on SNAP days in every state). Included via
  `add_regressor`, which requires the value to be known for the future
  too -- true here, since SNAP eligibility is a published schedule, not
  something we'd need to forecast.
- Model itself is deliberately simple (Prophet defaults for trend/
  seasonality) -- per the project brief, the pipeline is the point, not
  model sophistication.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
TEST_HORIZON = 28


def train_test_split(long_df: pd.DataFrame):
    dates = sorted(long_df["date"].unique())
    split_date = dates[-TEST_HORIZON]
    train = long_df[long_df["date"] < split_date].copy()
    test = long_df[long_df["date"] >= split_date].copy()
    return train, test


def build_holidays(long_df: pd.DataFrame) -> pd.DataFrame:
    events = long_df[["date", "event_name_1"]].dropna().drop_duplicates()
    return events.rename(columns={"date": "ds", "event_name_1": "holiday"})


def snap_column(row_df: pd.DataFrame) -> pd.Series:
    state = row_df["state_id"].iloc[0]
    return row_df[f"snap_{state}"].astype(int)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def fit_and_forecast(series_train: pd.DataFrame, series_test: pd.DataFrame,
                      holidays: pd.DataFrame) -> np.ndarray:
    train_df = pd.DataFrame({
        "ds": series_train["date"],
        "y": series_train["sales"],
        "snap": snap_column(series_train),
    })

    model = Prophet(holidays=holidays)
    model.add_regressor("snap")
    model.fit(train_df)

    future = pd.DataFrame({
        "ds": series_test["date"],
        "snap": snap_column(series_test),
    })
    forecast = model.predict(future)
    return np.clip(forecast["yhat"].to_numpy(), a_min=0, a_max=None)


def main():
    long_df = pd.read_csv(f"{PROCESSED_DIR}/subset_long.csv", parse_dates=["date"])
    train, test = train_test_split(long_df)
    holidays = build_holidays(long_df)
    print(f"Training Prophet on {train['id'].nunique()} series, "
          f"{len(holidays)} distinct calendar events as holidays")

    series_ids = sorted(long_df["id"].unique())
    per_series_rows = []
    all_forecasts = []

    for i, series_id in enumerate(series_ids, 1):
        s_train = train[train["id"] == series_id].sort_values("date")
        s_test = test[test["id"] == series_id].sort_values("date")

        yhat = fit_and_forecast(s_train, s_test, holidays)

        actual = s_test["sales"].to_numpy()
        series_mape = mape(actual, yhat)
        per_series_rows.append({"id": series_id, "mape": series_mape})

        out = s_test[["id", "date", "sales"]].copy()
        out["forecast"] = yhat
        all_forecasts.append(out)

        if i % 10 == 0 or i == len(series_ids):
            print(f"  [{i}/{len(series_ids)}] fitted -- last series MAPE: {series_mape:.1f}%")

    per_series = pd.DataFrame(per_series_rows)
    forecasts = pd.concat(all_forecasts, ignore_index=True)

    mean_per_series_mape = per_series["mape"].mean()
    daily = forecasts.groupby("date")[["sales", "forecast"]].sum().reset_index()
    aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())

    print(f"\nProphet -- Test horizon: {TEST_HORIZON} days")
    print(f"  Mean per-series MAPE: {mean_per_series_mape:.2f}%")
    print(f"  Aggregate (summed) MAPE: {aggregate_mape:.2f}%")

    per_series.to_csv(f"{RESULTS_DIR}/prophet_mape_per_series.csv", index=False)
    forecasts.to_csv(f"{RESULTS_DIR}/prophet_forecasts.csv", index=False)
    summary = pd.DataFrame([{
        "model": "prophet",
        "mean_per_series_mape": mean_per_series_mape,
        "aggregate_mape": aggregate_mape,
        "test_horizon_days": TEST_HORIZON,
        "n_series": long_df["id"].nunique(),
    }])
    summary.to_csv(f"{RESULTS_DIR}/prophet_summary.csv", index=False)

    baseline_summary = pd.read_csv(f"{RESULTS_DIR}/baseline_summary.csv")
    comparison = pd.concat([baseline_summary, summary], ignore_index=True)
    comparison.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(f"\nSaved comparison table to {RESULTS_DIR}/model_comparison.csv")
    print(comparison[["model", "mean_per_series_mape", "aggregate_mape"]])


if __name__ == "__main__":
    main()
