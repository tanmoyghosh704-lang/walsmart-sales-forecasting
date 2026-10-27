"""DVC "validate" stage -- schema/quality gate between the ingested sales
history (src/ingest_new_data.py) and data_prep.py. Fails loudly (non-zero
exit) if the data doesn't look like the M5 dataset this pipeline was built
for, instead of letting bad data flow silently into reshaping/feature
engineering.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
REPORT_PATH = "results/data_validation_report.txt"

# Same fallback as data_prep.py: validate the ingested history once
# ingestion has run at least once, otherwise the static seed file.
HISTORY_PATH = f"{PROCESSED_DIR}/sales_history.csv"
SEED_PATH = f"{RAW_DIR}/sales_train_validation.csv"
SALES_PATH = HISTORY_PATH if os.path.exists(HISTORY_PATH) else SEED_PATH

SALES_REQUIRED_COLS = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
CALENDAR_REQUIRED_COLS = {
    "date", "d", "wday", "month", "year", "weekday", "wm_yr_wk",
    "snap_CA", "snap_TX", "snap_WI",
}
PRICES_REQUIRED_COLS = {"store_id", "item_id", "wm_yr_wk", "sell_price"}


class ValidationError(Exception):
    pass


def require(condition: bool, message: str):
    if not condition:
        raise ValidationError(message)


def validate_sales(path: str):
    df = pd.read_csv(path)
    require(len(df) > 0, f"{path}: no rows")
    require(SALES_REQUIRED_COLS.issubset(df.columns), f"{path}: missing id columns")
    day_cols = [c for c in df.columns if c.startswith("d_")]
    require(len(day_cols) > 0, f"{path}: no d_* day columns found")
    require(df["id"].notna().all(), f"{path}: null values in 'id'")
    require(df[day_cols].isna().sum().sum() == 0, f"{path}: null values in sales day columns")
    require((df[day_cols] >= 0).all().all(), f"{path}: negative sales values")
    return f"OK  {path}: {len(df)} series x {len(day_cols)} days"


def validate_calendar(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    require(len(df) > 0, f"{path}: no rows")
    require(CALENDAR_REQUIRED_COLS.issubset(df.columns), f"{path}: missing required columns")
    require(df["date"].notna().all(), f"{path}: null dates")
    require(df["date"].is_monotonic_increasing, f"{path}: dates not sorted ascending")
    for col in ("snap_CA", "snap_TX", "snap_WI"):
        require(df[col].isin([0, 1]).all(), f"{path}: '{col}' has values outside {{0,1}}")
    return f"OK  {path}: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}"


def validate_prices(path: str):
    df = pd.read_csv(path)
    require(len(df) > 0, f"{path}: no rows")
    require(PRICES_REQUIRED_COLS.issubset(df.columns), f"{path}: missing required columns")
    require(df["sell_price"].notna().all(), f"{path}: null sell_price values")
    require((df["sell_price"] > 0).all(), f"{path}: non-positive sell_price values")
    return f"OK  {path}: {len(df)} (store, item, week) price rows"


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        lines = [
            validate_sales(SALES_PATH),
            validate_calendar(f"{RAW_DIR}/calendar.csv"),
            validate_prices(f"{RAW_DIR}/sell_prices.csv"),
        ]
    except ValidationError as exc:
        with open(REPORT_PATH, "w") as f:
            f.write(f"{timestamp}  VALIDATION FAILED: {exc}\n")
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    for line in lines:
        print(line)
    with open(REPORT_PATH, "w") as f:
        f.write(f"{timestamp}  All raw data validation checks passed.\n")
        f.write("\n".join(lines) + "\n")
    print("All raw data validation checks passed.")


if __name__ == "__main__":
    main()
