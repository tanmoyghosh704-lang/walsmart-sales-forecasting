"""
Naive baseline forecast, computed BEFORE any modeling.

Method: seasonal naive (lag-7). We take each series' last 7 observed days
of training data and tile that 7-day pattern across the 28-day test
horizon. This captures weekly seasonality (e.g. weekend spikes) with zero
model fitting -- it's the bar Prophet has to clear to justify its
complexity. If Prophet can't beat this, Prophet isn't earning its place
in the pipeline.

Metric: MAPE (Mean Absolute Percentage Error), per the project's headline
metric. Caveat logged here for the writeup: MAPE is undefined when actual
sales = 0, which happens often in retail data. We exclude zero-actual rows
from the per-row MAPE average (standard practice) and separately report
an aggregate-level MAPE (summed sales across all series per day) as a
more stable number, since summing across series makes exact zeros rare.
"""

import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
TEST_HORIZON = 28


def train_test_split(long_df: pd.DataFrame):
    dates = sorted(long_df["date"].unique())
    split_date = dates[-TEST_HORIZON]
    train = long_df[long_df["date"] < split_date].copy()
    test = long_df[long_df["date"] >= split_date].copy()
    return train, test


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    forecasts = []
    for series_id, series_test in test.groupby("id"):
        series_train = train[train["id"] == series_id].sort_values("date")
        last_week = series_train["sales"].tail(7).to_numpy()
        series_test = series_test.sort_values("date").reset_index(drop=True)
        horizon = len(series_test)
        # tile the last observed 7-day pattern to cover the full horizon
        pred = np.tile(last_week, int(np.ceil(horizon / 7)))[:horizon]
        series_test = series_test.copy()
        series_test["forecast"] = pred
        forecasts.append(series_test)
    return pd.concat(forecasts, ignore_index=True)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def main():
    long_df = pd.read_csv(f"{PROCESSED_DIR}/subset_long.csv", parse_dates=["date"])
    train, test = train_test_split(long_df)
    print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} "
          f"({train['date'].nunique()} days)")
    print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()} "
          f"({test['date'].nunique()} days)")

    result = seasonal_naive_forecast(train, test)

    # Per-series MAPE
    per_series = (
        result.groupby("id")
        .apply(lambda g: mape(g["sales"].to_numpy(), g["forecast"].to_numpy()), include_groups=False)
        .rename("mape")
        .reset_index()
    )
    per_series_mape = per_series["mape"].mean()

    # Aggregate-level MAPE (sum sales across all series per day, then compare)
    daily = result.groupby("date")[["sales", "forecast"]].sum().reset_index()
    aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())

    print(f"\nNaive baseline (seasonal, lag-7) -- Test horizon: {TEST_HORIZON} days")
    print(f"  Mean per-series MAPE: {per_series_mape:.2f}%")
    print(f"  Aggregate (summed) MAPE: {aggregate_mape:.2f}%")

    per_series.to_csv(f"{RESULTS_DIR}/baseline_mape_per_series.csv", index=False)
    summary = pd.DataFrame([{
        "model": "naive_seasonal_lag7",
        "mean_per_series_mape": per_series_mape,
        "aggregate_mape": aggregate_mape,
        "test_horizon_days": TEST_HORIZON,
        "n_series": long_df["id"].nunique(),
    }])
    summary.to_csv(f"{RESULTS_DIR}/baseline_summary.csv", index=False)
    print(f"\nSaved per-series MAPE to {RESULTS_DIR}/baseline_mape_per_series.csv")
    print(f"Saved summary to {RESULTS_DIR}/baseline_summary.csv")


if __name__ == "__main__":
    main()
