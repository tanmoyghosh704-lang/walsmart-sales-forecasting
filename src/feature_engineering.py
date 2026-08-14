"""
Build a feature-engineered, model-ready dataset for the global ML models
(Linear Regression, Random Forest, XGBoost, LightGBM) -- Prophet and
ARIMA work directly off the raw long-format series and don't need this.

Leakage-aware lag design: the task is a 28-day-ahead forecast (same test
horizon as Prophet), not a 1-day-ahead rolling forecast. A naive lag_7
feature would be unusable for most of the test window -- predicting day
15 of the horizon with "sales 7 days ago" would require knowing sales
from day 8 of the horizon, which hasn't happened yet at forecast time.
lag_28 is the largest lag that stays valid for *every* day across the
full 28-day horizon (lag_28 for the last test day still points at the
last training day), so every history-based feature here is built on top
of lag_28, not shorter lags. This mirrors how top M5 competition
solutions actually handled the same constraint.
"""

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SUBSET_PATH = f"{PROCESSED_DIR}/subset_long.csv"
LAG = 28

CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]


def load_prices(subset: pd.DataFrame) -> pd.DataFrame:
    store_items = subset[["store_id", "item_id"]].drop_duplicates()
    prices = pd.read_csv(f"{RAW_DIR}/sell_prices.csv")
    prices = prices.merge(store_items, on=["store_id", "item_id"], how="inner")
    return prices


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id")["sales"]
    df["lag_28"] = g.shift(LAG)
    shifted = g.shift(LAG)
    df["rolling_mean_7_lag28"] = (
        shifted.groupby(df["id"]).rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["rolling_mean_28_lag28"] = (
        shifted.groupby(df["id"]).rolling(28).mean().reset_index(level=0, drop=True)
    )
    df["rolling_std_28_lag28"] = (
        shifted.groupby(df["id"]).rolling(28).std().reset_index(level=0, drop=True)
    )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["event_type_1"] = df["event_type_1"].fillna("none")

    def snap_for_row(row):
        return row[f"snap_{row['state_id']}"]

    df["snap"] = df.apply(snap_for_row, axis=1)
    return df


def add_price_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # A handful of early weeks predate an item's price being listed at a
    # given store; impute with that series' own median price rather than
    # dropping rows (dropping would bias the training set toward items
    # with complete price history).
    df["sell_price"] = df.groupby("id")["sell_price"].transform(
        lambda s: s.fillna(s.median())
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categoricals for tree models. Linear Regression
    one-hot-encodes these itself at train time (see src/train_ml_models.py)
    -- ordinal codes would wrongly imply an order between categories for
    a linear model, but tree models split on thresholds regardless of
    encoding, so label codes are a fine, standard simplification there."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        codes, uniques = pd.factorize(df[col])
        df[f"{col}_code"] = codes
        encoders[col] = list(uniques)
    return df, encoders


def main():
    subset = pd.read_csv(SUBSET_PATH, parse_dates=["date"])
    prices = load_prices(subset)

    df = add_lag_features(subset)
    df = add_calendar_features(df)
    df = add_price_features(df, prices)
    df, encoders = encode_categoricals(df)

    before = len(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_7_lag28", "rolling_mean_28_lag28"])
    print(f"Dropped {before - len(df)} rows with insufficient history for lag_28 features "
          f"(first {LAG + 27} days per series)")

    feature_cols = [
        "lag_28", "rolling_mean_7_lag28", "rolling_mean_28_lag28", "rolling_std_28_lag28",
        "wday", "month", "year", "is_weekend", "has_event", "snap", "sell_price",
        "item_id_code", "dept_id_code", "cat_id_code", "store_id_code",
        "state_id_code", "event_type_1_code",
    ]
    out_cols = ["id", "date", "sales"] + feature_cols
    out = df[out_cols].copy()

    out_path = f"{PROCESSED_DIR}/ml_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows x {len(feature_cols)} features to {out_path}")
    print(f"Feature columns: {feature_cols}")


if __name__ == "__main__":
    main()
