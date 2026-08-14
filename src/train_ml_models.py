"""
Train four *global* models (one model each, across all 100 series at
once -- not per-series like Prophet/ARIMA) on the leakage-aware
lag_28-based feature set from src/feature_engineering.py:

  - Linear Regression  (simple baseline; categoricals one-hot encoded,
                         since ordinal codes would wrongly imply an order
                         a linear model would try to use)
  - Random Forest
  - XGBoost
  - LightGBM           (categoricals passed as true categorical features,
                         not just integer codes, for better splits)

Same 28-day test horizon and same two MAPE framings (mean per-series,
aggregate) as src/baseline.py and src/train.py, so all five models
(naive, Prophet, and these four) are directly comparable in
results/model_comparison.csv.

Hyperparameters are sensible defaults, not tuned -- per this project's
own stated goal, the pipeline (feature engineering leakage-safety,
tracking, comparison, registry) is the point, not squeezing out the last
percent of MAPE via a hyperparameter search.
"""

import os
import sys

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
TEST_HORIZON = 28

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "m5-model-comparison"

NUMERIC_FEATURES = [
    "lag_28", "rolling_mean_7_lag28", "rolling_mean_28_lag28", "rolling_std_28_lag28",
    "wday", "month", "year", "is_weekend", "has_event", "snap", "sell_price",
]
CATEGORICAL_FEATURES = [
    "item_id_code", "dept_id_code", "cat_id_code", "store_id_code",
    "state_id_code", "event_type_1_code",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_test_split(df: pd.DataFrame):
    dates = sorted(df["date"].unique())
    split_date = dates[-TEST_HORIZON]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def evaluate(test: pd.DataFrame, y_pred: np.ndarray):
    result = test[["id", "date", "sales"]].copy()
    result["forecast"] = np.clip(y_pred, a_min=0, a_max=None)

    per_series = result.groupby("id").apply(
        lambda g: mape(g["sales"].to_numpy(), g["forecast"].to_numpy()), include_groups=False
    ).rename("mape").reset_index()
    mean_per_series_mape = per_series["mape"].mean()

    daily = result.groupby("date")[["sales", "forecast"]].sum().reset_index()
    aggregate_mape = mape(daily["sales"].to_numpy(), daily["forecast"].to_numpy())
    return mean_per_series_mape, aggregate_mape, per_series, result


def log_and_register(model_name: str, params: dict, mean_mape: float, agg_mape: float,
                      per_series: pd.DataFrame, forecasts: pd.DataFrame, log_model_fn):
    per_series.to_csv(f"{RESULTS_DIR}/{model_name}_mape_per_series.csv", index=False)
    forecasts.to_csv(f"{RESULTS_DIR}/{model_name}_forecasts.csv", index=False)
    with mlflow.start_run(run_name=model_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mean_per_series_mape", mean_mape)
        mlflow.log_metric("median_per_series_mape", per_series["mape"].median())
        mlflow.log_metric("aggregate_mape", agg_mape)
        log_model_fn(f"sales_{model_name}")
        print(f"{model_name}: mean_per_series_mape={mean_mape:.2f}%  "
              f"median_per_series_mape={per_series['mape'].median():.2f}%  "
              f"aggregate_mape={agg_mape:.2f}%")


def train_linear_regression(train, test):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", LinearRegression()),
    ])
    model.fit(train[ALL_FEATURES], train["sales"])
    y_pred = model.predict(test[ALL_FEATURES])
    mean_mape, agg_mape, per_series, forecasts = evaluate(test, y_pred)
    log_and_register(
        "linear_regression", {"model_type": "LinearRegression", "one_hot_categoricals": True},
        mean_mape, agg_mape, per_series, forecasts,
        lambda name: mlflow.sklearn.log_model(model, name="model", registered_model_name=name),
    )
    return mean_mape, agg_mape


def train_random_forest(train, test):
    params = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5,
              "n_jobs": -1, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(train[ALL_FEATURES], train["sales"])
    y_pred = model.predict(test[ALL_FEATURES])
    mean_mape, agg_mape, per_series, forecasts = evaluate(test, y_pred)
    log_and_register(
        "random_forest", params, mean_mape, agg_mape, per_series, forecasts,
        lambda name: mlflow.sklearn.log_model(model, name="model", registered_model_name=name),
    )
    return mean_mape, agg_mape


def train_xgboost(train, test):
    params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
    model = xgb.XGBRegressor(**params)
    model.fit(train[ALL_FEATURES], train["sales"])
    y_pred = model.predict(test[ALL_FEATURES])
    mean_mape, agg_mape, per_series, forecasts = evaluate(test, y_pred)
    log_and_register(
        "xgboost", params, mean_mape, agg_mape, per_series, forecasts,
        lambda name: mlflow.xgboost.log_model(model, name="model", registered_model_name=name),
    )
    return mean_mape, agg_mape


def train_lightgbm(train, test):
    params = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
              "num_leaves": 63, "random_state": 42, "verbosity": -1}
    train_X = train[ALL_FEATURES].copy()
    test_X = test[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        train_X[col] = train_X[col].astype("category")
        test_X[col] = test_X[col].astype("category")

    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train["sales"], categorical_feature=CATEGORICAL_FEATURES)
    y_pred = model.predict(test_X)
    mean_mape, agg_mape, per_series, forecasts = evaluate(test, y_pred)
    log_and_register(
        "lightgbm", params, mean_mape, agg_mape, per_series, forecasts,
        lambda name: mlflow.lightgbm.log_model(model, name="model", registered_model_name=name),
    )
    return mean_mape, agg_mape


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = pd.read_csv(f"{PROCESSED_DIR}/ml_features.csv", parse_dates=["date"])
    train, test = train_test_split(df)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    results = []
    for name, fn in [
        ("linear_regression", train_linear_regression),
        ("random_forest", train_random_forest),
        ("xgboost", train_xgboost),
        ("lightgbm", train_lightgbm),
    ]:
        mean_mape, agg_mape = fn(train, test)
        results.append({
            "model": name, "mean_per_series_mape": mean_mape, "aggregate_mape": agg_mape,
            "test_horizon_days": TEST_HORIZON, "n_series": df["id"].nunique(),
        })

    summary = pd.DataFrame(results)
    summary.to_csv(f"{RESULTS_DIR}/ml_models_summary.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/ml_models_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
