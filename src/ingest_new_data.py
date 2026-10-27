"""First DAG step: simulate new sales data arriving and append it to the
working historical dataset the rest of the pipeline builds on.

The M5 Kaggle dataset is static, so there's no real live feed to pull from.
`sales_train_evaluation.csv` (already in data/raw/) has the exact same
series as `sales_train_validation.csv` plus 28 more real days
(d_1914..d_1941) -- that's used here as a stand-in "new data" source.
Each run absorbs the next INGEST_CHUNK_DAYS-day chunk into
data/processed/sales_history.csv, which is what data_prep.py reads from
(falling back to the static seed file if ingestion has never run yet).
Once all 28 extra days are absorbed, further runs are a no-op -- this is
a simulation of incremental ingestion, not a real one, and it can't
manufacture days that don't exist in the source.
"""
import os

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
HISTORY_PATH = f"{PROCESSED_DIR}/sales_history.csv"
SEED_PATH = f"{RAW_DIR}/sales_train_validation.csv"
NEW_DATA_SOURCE_PATH = f"{RAW_DIR}/sales_train_evaluation.csv"

# NOT "id": M5's own convention gives it a different suffix in each file
# ("..._validation" vs "..._evaluation") even though it's the same series,
# so merging on it directly would never match anything. The other 5
# columns are suffix-free and jointly identify a series just as uniquely.
MERGE_KEY_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

CHUNK_DAYS = int(os.environ.get("INGEST_CHUNK_DAYS", "7"))


def day_cols(columns) -> list:
    return sorted((c for c in columns if c.startswith("d_")), key=lambda c: int(c.split("_")[1]))


def main():
    current_path = HISTORY_PATH if os.path.exists(HISTORY_PATH) else SEED_PATH
    current_header = pd.read_csv(current_path, nrows=0)
    last_day = int(day_cols(current_header.columns)[-1].split("_")[1])

    source = pd.read_csv(NEW_DATA_SOURCE_PATH)
    available_days = day_cols(source.columns)
    max_available_day = int(available_days[-1].split("_")[1])

    target_day = min(last_day + CHUNK_DAYS, max_available_day)
    if target_day <= last_day:
        print(f"No new data to ingest -- history already spans through d_{last_day}, "
              f"source has through d_{max_available_day}.")
        return

    new_cols = [f"d_{d}" for d in range(last_day + 1, target_day + 1)]
    history = pd.read_csv(current_path)
    increment = source[MERGE_KEY_COLS + new_cols]
    merged = history.merge(increment, on=MERGE_KEY_COLS, how="left")

    missing = merged[new_cols].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(
            f"{missing} series had no matching row in {NEW_DATA_SOURCE_PATH} for the new day "
            f"columns -- merge key {MERGE_KEY_COLS} didn't uniquely match; refusing to write "
            f"a history file with gaps."
        )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(HISTORY_PATH, index=False)
    print(f"Ingested {len(new_cols)} new day(s): d_{last_day + 1}..d_{target_day}. "
          f"History now spans through d_{target_day} ({len(merged)} series), saved to {HISTORY_PATH}.")


if __name__ == "__main__":
    main()
