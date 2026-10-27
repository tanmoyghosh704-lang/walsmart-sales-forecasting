"""Feature engineering shared by the training pipeline (feature_engineering.py,
run as a DVC stage) and the live serving path (serving/app.py).

Both call `engineer_features()` so a request scored at serving time goes
through the exact same lag/rolling/calendar/price/categorical transforms as
training data did -- the champion model's feature contract is defined once,
here, instead of being re-derived (and silently drifting) in two places.
"""
import json

import pandas as pd

LAG = 28

CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]

NUMERIC_FEATURES = [
    "lag_28", "rolling_mean_7_lag28", "rolling_mean_28_lag28", "rolling_std_28_lag28",
    "wday", "month", "year", "is_weekend", "has_event", "snap", "sell_price",
]
CATEGORICAL_FEATURES = [f"{col}_code" for col in CATEGORICAL_COLS]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_prices(subset: pd.DataFrame, raw_dir: str = "data/raw") -> pd.DataFrame:
    store_items = subset[["store_id", "item_id"]].drop_duplicates()
    prices = pd.read_csv(f"{raw_dir}/sell_prices.csv")
    return prices.merge(store_items, on=["store_id", "item_id"], how="inner")


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
    df = df.copy()
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["event_type_1"] = df["event_type_1"].fillna("none")

    def snap_for_row(row):
        return row[f"snap_{row['state_id']}"]

    df["snap"] = df.apply(snap_for_row, axis=1)
    return df


def add_price_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["sell_price"] = df.groupby("id")["sell_price"].transform(
        lambda s: s.fillna(s.median())
    )
    return df


def fit_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        codes, uniques = pd.factorize(df[col])
        df[f"{col}_code"] = codes
        encoders[col] = [str(v) for v in uniques]
    return df, encoders


def apply_categoricals(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        lookup = {value: code for code, value in enumerate(encoders[col])}
        df[f"{col}_code"] = df[col].astype(str).map(lookup).fillna(-1).astype(int)
    return df


def save_encoders(encoders: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(encoders, f, indent=2)


def load_encoders(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def engineer_features(
    df: pd.DataFrame, prices: pd.DataFrame, encoders: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run the full lag -> calendar -> price -> categorical pipeline.

    Pass `encoders=None` to fit new category codes (training). Pass a dict
    (loaded via `load_encoders`) to encode with codes learned at training
    time (serving) -- unseen categories map to -1 rather than raising.
    """
    out = add_lag_features(df)
    out = add_calendar_features(out)
    out = add_price_features(out, prices)
    if encoders is None:
        out, encoders = fit_categoricals(out)
    else:
        out = apply_categoricals(out, encoders)
    return out, encoders


def build_future_frame(
    series_id: str,
    horizon: int,
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    encoders: dict,
) -> pd.DataFrame:
    """Build the feature matrix for the next `horizon` days of one series.

    `history` must contain at least the last LAG days of that series'
    `subset_long.csv` rows (so lag/rolling features have real sales to look
    back on) -- forecasting beyond `horizon > LAG` would require feeding
    the model's own prior predictions back in as "sales" (recursive
    forecasting), which this project does not do, so callers must cap
    `horizon` at LAG.
    """
    series_hist = history[history["id"] == series_id].sort_values("date")
    last = series_hist.iloc[-1]
    future_dates = pd.date_range(series_hist["date"].max() + pd.Timedelta(days=1), periods=horizon)

    future = calendar.loc[future_dates].reset_index().rename(columns={"index": "date"})
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        future[col] = last[col]
    future["sales"] = float("nan")

    combined = pd.concat([series_hist, future], ignore_index=True, sort=False)
    engineered, _ = engineer_features(combined, prices, encoders=encoders)
    engineered = engineered[engineered["date"].isin(future_dates)].sort_values("date")
    return engineered.set_index("date")
