# Walmart M5 Demand Forecasting — MLOps Pipeline

A production-shaped demand forecasting pipeline built on Kaggle's M5
Forecasting - Accuracy dataset. Seven forecasting approaches are
benchmarked head-to-head on 100 high-volume store-item series, tracked
and registered in MLflow with conditional champion promotion, versioned
with DVC, served via FastAPI in Docker, orchestrated with Airflow, and
monitored with Prometheus/Grafana including a prediction-drift check.
The models themselves are intentionally simple; the pipeline around
them is the point.

## Results

100 series (top 100 by historical volume), 28-day held-out test
horizon, MAPE reported three ways since it's sensitive to outliers on
low-volume days (see Methodology).

| Model | Mean per-series MAPE | Median per-series MAPE | Aggregate MAPE |
|---|---|---|---|
| **LightGBM** 🏆 (champion) | 73.45% | 33.60% | **5.82%** |
| Linear Regression | 80.73% | 36.09% | 6.45% |
| XGBoost | 74.71% | 33.55% | 6.48% |
| Random Forest | 87.17% | 32.79% | 7.29% |
| SVM | 60.37% | 31.87% | 7.83% |
| Naive (seasonal lag-7) | 78.58% | 40.86% | 10.75% |
| ARIMA | 85.21% | 43.87% | 18.46% |

LightGBM is registered as the MLflow champion (`sales_lightgbm@champion`),
selected on aggregate MAPE. That's a 45.8% improvement over the naive
baseline.

## Methodology

**Scope.** The full M5 dataset covers ~30,490 store-item series. This
project uses the top 100 by total historical volume, which keeps
zero-sales days rare enough for MAPE to stay meaningful. That selection
skews heavily toward the `FOODS` category (grocery items sell far more
often than hobbies or household goods), which is a real bias in the
subset, not an artifact of sampling error.

**Models.** SVM and ARIMA are trained per series (100 independent
models each), since neither has a natural way to share information
across series. Linear Regression, Random Forest, XGBoost, and LightGBM
are trained as single global models on all 100 series stacked together,
with store/item/department/category/state as categorical features —
the same approach used by the top solutions in the actual M5
competition. Every model is evaluated on the identical 28-day held-out
window.

**Features.** The global models use engineered features built around a
28-day lag rather than a 1-day lag, because the task is a 28-day-ahead
forecast: a lag shorter than the horizon would require knowing sales
that haven't happened yet by the time later days in the horizon are
predicted. On top of `lag_28`: 7- and 28-day rolling mean/std, calendar
features (weekday, month, is-weekend), SNAP eligibility, and price
(joined from `sell_prices.csv`).

**Metric.** MAPE is reported as mean per-series, median per-series, and
aggregate (sum of actual and forecast sales across all series before
computing one MAPE). Aggregate MAPE is used for champion selection
because it's far less sensitive to a single bad series: one series in
the ARIMA run had an 856% single-series MAPE from a real stockout mid
test-window, which pulled ARIMA's *mean* per-series MAPE above the
naive baseline even though its aggregate MAPE told a more representative
story.

**Champion promotion.** A candidate model only replaces the current
MLflow-registered champion if it actually beats it on aggregate MAPE.
Promotion is restricted to the four global models, since SVM and ARIMA
are each 100 separately-registered models with no single name an alias
could represent.

## Architecture

```
data/raw (DVC -> DagsHub)
   -> src/data_prep.py -> data/processed/subset_long.csv
   -> src/baseline.py             (naive)        --+
   -> src/train_svm.py            (SVR x100)     --+--> MLflow tracking + registry
   -> src/train_arima.py          (SARIMA x100)  --+
   -> src/feature_engineering.py (lag_28-based)
       -> src/train_ml_models.py (LinReg/RF/XGB/LightGBM, global) --+
   -> src/compare_models.py -> results/full_model_comparison.csv
   -> src/promote_champion.py -> MLflow champion alias
   -> serving/app.py (FastAPI) -> Docker
   -> Prometheus (/metrics: latency, request count, prediction drift) -> Grafana
   -> ui/app.py (Streamlit): backtest comparison across all approaches

airflow/dags/retrain_dag.py (m5_full_retrain) -> weekly retrain of every
   approach + conditional champion promotion (Docker Compose, LocalExecutor)
```

## Repository structure

```
data/                  DVC-tracked raw + processed M5 data
notebooks/eda.ipynb    Weekly seasonality, holiday/SNAP effects
src/
  data_prep.py           Scope full M5 down to top-100-by-volume subset
  baseline.py            Naive seasonal MAPE reference
  train_svm.py           Per-series SVR + MLflow tracking/registry
  train_arima.py         Per-series SARIMA, parallelized
  feature_engineering.py Leakage-safe lag_28-based features for the global models
  train_ml_models.py     Linear Regression / Random Forest / XGBoost / LightGBM
  compare_models.py      Full comparison table, sorted by aggregate MAPE
  promote_champion.py    Conditional MLflow champion alias promotion
ui/
  app.py                 Streamlit: per-series + overall backtest comparison
serving/
  app.py                 FastAPI /predict, /health, /metrics
  Dockerfile
airflow/
  dags/retrain_dag.py    m5_full_retrain: multi-task weekly retraining DAG
  Dockerfile              Airflow image + isolated task venv
monitoring/
  docker-compose.yaml    serving-app + Prometheus + Grafana
  prometheus.yml
  grafana/               Datasource + dashboard provisioning
docker-compose.yaml      Airflow stack (Postgres + webserver + scheduler)
results/                 MAPE tables, forecasts, EDA plots
ROADMAP.txt              Phase-by-phase build guide
```

## Prerequisites

- Python 3.10
- Docker Desktop
- A Kaggle account (to download the M5 dataset) and a DagsHub account
  (free tier, used as the DVC remote)

## Reproducing this project

### 1. Environment

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux
```

### 2. Data

Requires a Kaggle API token at `~/.kaggle/kaggle.json` and having
joined the M5 competition on kaggle.com.

```bash
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
cd data/raw && unzip m5-forecasting-accuracy.zip && rm m5-forecasting-accuracy.zip && cd ../..

python src/data_prep.py     # builds data/processed/subset_long.csv
python src/baseline.py      # naive MAPE reference
```

The DVC remote is already configured in `.dvc/config`. To pull the
already-processed data instead of rebuilding it: `dvc pull`.

### 3. Train and track

Start an MLflow tracking server (not a raw `sqlite:///` path, which
breaks once a client runs inside Docker):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 \
  --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
```

Then, in another terminal:

```bash
python src/train_svm.py           # SVM x100
python src/train_arima.py         # SARIMA x100 (parallelized across CPU cores)
python src/feature_engineering.py # builds data/processed/ml_features.csv
python src/train_ml_models.py     # Linear Regression, Random Forest, XGBoost, LightGBM
python src/compare_models.py      # builds results/full_model_comparison.csv
python src/promote_champion.py    # sets the MLflow champion alias
```

MLflow UI: http://127.0.0.1:5000

### 4. Explore the comparison

```bash
streamlit run ui/app.py
```

### 5. Serve

```bash
uvicorn serving.app:app --reload
```

- `GET /health`
- `GET /predict?series_id=FOODS_1_218_TX_2_validation&horizon=7`
- `GET /metrics`

Or containerized (the MLflow server must already be running on the host):

```bash
docker build -f serving/Dockerfile -t m5-forecast-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 m5-forecast-api
```

### 6. Orchestrate retraining

```bash
docker compose up airflow-init      # one-off: migrate DB, create admin user
docker compose up -d                # webserver + scheduler
```

Airflow UI: http://localhost:8080 (`airflow` / `airflow`). Trigger
`m5_full_retrain` manually or let it run on its weekly schedule.

### 7. Monitoring

```bash
docker compose -f monitoring/docker-compose.yaml up -d --build
```

- Grafana: http://localhost:3000 (`admin` / `admin`)
- Prometheus: http://localhost:9090

## Scope

Everything runs locally/in containers on a single machine; cloud
deployment, autoscaling, and high availability are out of scope. The
prediction-drift check compares recent served predictions against the
pooled training distribution across all 100 series rather than a
per-series baseline. No hyperparameter tuning was performed on any
model.
