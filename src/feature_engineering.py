import pandas as pd

from features import ALL_FEATURES, engineer_features, load_prices, save_encoders

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SUBSET_PATH = f"{PROCESSED_DIR}/subset_long.csv"
ENCODERS_PATH = f"{PROCESSED_DIR}/encoders.json"


def main():
    subset = pd.read_csv(SUBSET_PATH, parse_dates=["date"])
    prices = load_prices(subset, raw_dir=RAW_DIR)

    df, encoders = engineer_features(subset, prices)

    before = len(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_7_lag28", "rolling_mean_28_lag28"])
    print(f"Dropped {before - len(df)} rows with insufficient history for lag features")

    out_cols = ["id", "date", "sales"] + ALL_FEATURES
    out = df[out_cols].copy()

    out_path = f"{PROCESSED_DIR}/ml_features.csv"
    out.to_csv(out_path, index=False)
    save_encoders(encoders, ENCODERS_PATH)
    print(f"Saved {len(out)} rows x {len(ALL_FEATURES)} features to {out_path}")
    print(f"Saved category encoders to {ENCODERS_PATH}")
    print(f"Feature columns: {ALL_FEATURES}")


if __name__ == "__main__":
    main()
