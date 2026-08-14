"""
Streamlit demo UI for the M5 forecasting project.

Two views:
1. Per-series backtest comparison -- pick one of the 100 series, see
   actual sales (history + held-out test window) plotted against every
   model's test-period predictions, with each model's MAPE for that
   specific series.
2. Overall model comparison -- the same headline numbers from
   results/full_model_comparison.csv (aggregate MAPE across all 100
   series), so the per-series view can't be mistaken for the full
   picture. The champion model registered in MLflow is called out.

This reads pre-computed backtest results (results/*_forecasts.csv) rather
than generating live forecasts for arbitrary future dates. Reason: the
champion model (LightGBM) is a *global* model that needs the full
feature set (lag_28, rolling stats, price, calendar) to score new rows --
building that feature pipeline live for arbitrary future dates is a
different, larger piece of work than this demo needs. What's genuinely
useful here -- and what an interviewer would actually want to see -- is
"how well did each approach do on the same held-out test window,"
visualized per series, not a black-box future forecast.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "data/processed"
RESULTS_DIR = "results"

MODELS = {
    "Naive (seasonal lag-7)": "baseline",
    "Prophet": "prophet",
    "ARIMA": "arima",
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM (champion)": "lightgbm",
}
COMPARISON_KEY = {
    "baseline": "naive_seasonal_lag7", "prophet": "prophet", "arima": "arima",
    "linear_regression": "linear_regression", "random_forest": "random_forest",
    "xgboost": "xgboost", "lightgbm": "lightgbm",
}
COLORS = {
    "baseline": "#999999", "prophet": "#4C72B0", "arima": "#C44E52",
    "linear_regression": "#8172B2", "random_forest": "#937860",
    "xgboost": "#DA8BC3", "lightgbm": "#55A868",
}


@st.cache_data
def load_history():
    return pd.read_csv(f"{DATA_DIR}/subset_long.csv", parse_dates=["date"])


@st.cache_data
def load_forecasts(file_key: str):
    try:
        return pd.read_csv(f"{RESULTS_DIR}/{file_key}_forecasts.csv", parse_dates=["date"])
    except FileNotFoundError:
        return None


@st.cache_data
def load_per_series_mape(file_key: str):
    try:
        return pd.read_csv(f"{RESULTS_DIR}/{file_key}_mape_per_series.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_comparison():
    return pd.read_csv(f"{RESULTS_DIR}/full_model_comparison.csv")


st.set_page_config(page_title="M5 Forecasting Demo", layout="wide")
st.title("Walmart M5 Demand Forecasting")
st.caption(
    "Prophet, ARIMA, Linear Regression, Random Forest, XGBoost, and LightGBM, "
    "all backtested on the same 28-day held-out window across 100 store-item series."
)

history = load_history()
series_ids = sorted(history["id"].unique())

with st.sidebar:
    st.header("Controls")
    selected_series = st.selectbox("Series (item at store)", series_ids)
    selected_models = st.multiselect(
        "Models to compare", list(MODELS.keys()),
        default=["Naive (seasonal lag-7)", "Prophet", "LightGBM (champion)"],
    )
    history_days = st.slider("History window (days before test)", 30, 365, 90)

tab1, tab2 = st.tabs(["Per-series backtest", "Overall model comparison"])

with tab1:
    series_history = history[history["id"] == selected_series].sort_values("date")
    test_start = series_history["date"].max() - pd.Timedelta(days=27)
    plot_start = test_start - pd.Timedelta(days=history_days)
    plot_df = series_history[series_history["date"] >= plot_start]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["sales"], mode="lines",
        name="Actual", line=dict(color="black", width=2),
    ))
    fig.add_vrect(x0=test_start, x1=series_history["date"].max(),
                   fillcolor="lightgray", opacity=0.3, line_width=0,
                   annotation_text="test window", annotation_position="top left")

    mape_rows = []
    for label in selected_models:
        key = MODELS[label]
        forecasts = load_forecasts(key)
        if forecasts is None:
            st.warning(f"No forecast file found for {label} yet -- run its training script first.")
            continue
        series_forecast = forecasts[forecasts["id"] == selected_series].sort_values("date")
        fig.add_trace(go.Scatter(
            x=series_forecast["date"], y=series_forecast["forecast"], mode="lines+markers",
            name=label, line=dict(color=COLORS[key], dash="dot"),
        ))

        per_series = load_per_series_mape(key)
        if per_series is not None:
            row = per_series[per_series["id"] == selected_series]
            if len(row):
                mape_rows.append({"model": label, "MAPE (this series)": f"{row['mape'].iloc[0]:.1f}%"})

    fig.update_layout(
        height=500, xaxis_title="Date", yaxis_title="Units sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if mape_rows:
        st.subheader(f"MAPE for {selected_series} (this series only)")
        st.dataframe(pd.DataFrame(mape_rows), hide_index=True, use_container_width=True)
        st.caption(
            "A single series' MAPE is noisy (see results/writeup.md) -- "
            "check the Overall comparison tab for the metric actually used to pick the champion."
        )

with tab2:
    st.subheader("Aggregate MAPE across all 100 series (sorted best to worst)")
    comparison = load_comparison().sort_values("aggregate_mape")
    comparison_display = comparison.copy()
    comparison_display["model"] = comparison_display["model"].apply(
        lambda m: f"{m}  \U0001f3c6" if m == "lightgbm" else m
    )
    st.dataframe(
        comparison_display[["model", "aggregate_mape", "mean_per_series_mape", "median_per_series_mape"]]
        .rename(columns={
            "aggregate_mape": "Aggregate MAPE (%)",
            "mean_per_series_mape": "Mean per-series MAPE (%)",
            "median_per_series_mape": "Median per-series MAPE (%)",
        }),
        hide_index=True, use_container_width=True,
    )

    bar = go.Figure(go.Bar(
        x=comparison["model"], y=comparison["aggregate_mape"],
        marker_color=[COLORS.get(m, "#333333") for m in comparison["model"]],
    ))
    bar.update_layout(height=400, yaxis_title="Aggregate MAPE (%)",
                       title="Lower is better -- selection metric for the registered champion")
    st.plotly_chart(bar, use_container_width=True)

    st.info(
        "**Champion: LightGBM** (5.82% aggregate MAPE), registered in the MLflow Model Registry "
        "as `sales_lightgbm` with alias `champion`. Selected on **aggregate MAPE**, not mean "
        "per-series MAPE -- ARIMA's mean per-series MAPE (85%) was dragged up by a single series "
        "with an 856% error (a mid-test-window stockout no history-only model could have predicted), "
        "while its aggregate MAPE (18.5%) told a more representative story. See results/writeup.md."
    )
