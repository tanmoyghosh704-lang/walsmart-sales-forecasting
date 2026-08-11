"""
Scope the full M5 dataset down to a manageable subset and reshape it into
long format (one row per series-day) joined with calendar features.

Why a subset: M5 has ~30,490 store-item series. Training/evaluating a
model per series at that scale is slow to iterate on and unnecessary to
demonstrate the pipeline. We pick the top-N series by total historical
volume — high-volume series have less intermittent (zero-heavy) demand,
which makes MAPE a meaningful metric instead of being dominated by
divide-by-zero noise from slow-moving items.
"""

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
N_SERIES = 100


def load_raw_sales() -> pd.DataFrame:
    return pd.read_csv(f"{RAW_DIR}/sales_train_validation.csv")


def load_calendar() -> pd.DataFrame:
    return pd.read_csv(f"{RAW_DIR}/calendar.csv")


def select_top_series(sales_wide: pd.DataFrame, n: int = N_SERIES) -> pd.DataFrame:
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    totals = sales_wide[day_cols].sum(axis=1)
    top_ids = sales_wide.loc[
        totals.sort_values(ascending=False).index[:n], "id"
    ]
    return sales_wide[sales_wide["id"].isin(top_ids)].reset_index(drop=True)


def to_long_format(sales_subset: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_subset.columns if c.startswith("d_")]

    long_df = sales_subset.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )

    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    long_df = long_df.merge(calendar[cal_cols], on="d", how="left")
    long_df["date"] = pd.to_datetime(long_df["date"])

    # d_1, d_2, ... sorts lexicographically wrong (d_10 before d_2) — sort by date instead.
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
    return long_df


def main():
    print(f"Loading raw sales from {RAW_DIR}/sales_train_validation.csv ...")
    sales_wide = load_raw_sales()
    print(f"Full dataset: {sales_wide.shape[0]} series x {sales_wide.shape[1]} columns")

    subset = select_top_series(sales_wide, N_SERIES)
    print(f"Selected top {N_SERIES} series by total historical volume")

    calendar = load_calendar()
    long_df = to_long_format(subset, calendar)
    print(f"Long-format subset: {long_df.shape[0]} rows ({N_SERIES} series x "
          f"{long_df['d'].nunique()} days)")

    out_path = f"{PROCESSED_DIR}/subset_long.csv"
    long_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
