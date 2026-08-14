"""
Assemble the full model comparison across every approach tried in this
project, and select a champion.

Selection metric: aggregate MAPE, not mean per-series MAPE. Justified by
a concrete finding from the ARIMA run: one series (a likely stockout/
discontinuation mid-test-period, invisible in training history) produced
an 856% single-series MAPE that dragged ARIMA's *mean* per-series MAPE
up to 85% -- worse than even the naive baseline -- while ARIMA's
*aggregate* MAPE (18.5%) and *median* per-series MAPE (43.9%) told a much
more reasonable story. Aggregate MAPE is far less sensitive to a single
pathological series, which is exactly the property wanted in a model
*selection* metric (median per-series MAPE also resists this and is
reported here too, but aggregate MAPE is what's used to serve the
model-comparison headline number throughout this project).
"""

import pandas as pd

RESULTS_DIR = "results"

SUMMARY_FILES = {
    "naive_seasonal_lag7": f"{RESULTS_DIR}/baseline_summary.csv",
    "prophet": f"{RESULTS_DIR}/prophet_summary.csv",
    "arima": f"{RESULTS_DIR}/arima_summary.csv",
}
PER_SERIES_FILES = {
    "naive_seasonal_lag7": f"{RESULTS_DIR}/baseline_mape_per_series.csv",
    "prophet": f"{RESULTS_DIR}/prophet_mape_per_series.csv",
    "arima": f"{RESULTS_DIR}/arima_mape_per_series.csv",
    "linear_regression": f"{RESULTS_DIR}/linear_regression_mape_per_series.csv",
    "random_forest": f"{RESULTS_DIR}/random_forest_mape_per_series.csv",
    "xgboost": f"{RESULTS_DIR}/xgboost_mape_per_series.csv",
    "lightgbm": f"{RESULTS_DIR}/lightgbm_mape_per_series.csv",
}
ML_SUMMARY_FILE = f"{RESULTS_DIR}/ml_models_summary.csv"


def median_per_series_mape(model: str) -> float:
    path = PER_SERIES_FILES.get(model)
    if path is None:
        return float("nan")
    df = pd.read_csv(path)
    return df["mape"].median()


def main():
    rows = []
    for model, path in SUMMARY_FILES.items():
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row["model"] = model
        row["median_per_series_mape"] = median_per_series_mape(model)
        rows.append(row)

    ml_summary = pd.read_csv(ML_SUMMARY_FILE)
    for _, row in ml_summary.iterrows():
        r = row.to_dict()
        r["median_per_series_mape"] = median_per_series_mape(r["model"])
        rows.append(r)

    comparison = pd.DataFrame(rows)[
        ["model", "mean_per_series_mape", "median_per_series_mape",
         "aggregate_mape", "test_horizon_days", "n_series"]
    ].sort_values("aggregate_mape").reset_index(drop=True)

    champion = comparison.iloc[0]
    print("Full model comparison (sorted by aggregate MAPE):\n")
    print(comparison.to_string(index=False))
    print(f"\nChampion (lowest aggregate MAPE): {champion['model']} "
          f"({champion['aggregate_mape']:.2f}%)")

    comparison.to_csv(f"{RESULTS_DIR}/full_model_comparison.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/full_model_comparison.csv")


if __name__ == "__main__":
    main()
