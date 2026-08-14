# Walmart M5 Demand Forecasting — MLOps Pipeline

An end-to-end, production-shaped demand forecasting pipeline built on
Kaggle's M5 Forecasting - Accuracy dataset: **seven forecasting
approaches compared head-to-head** (naive baseline, Prophet, SARIMA,
Linear Regression, Random Forest, XGBoost, LightGBM), tracked and
registered in MLflow (champion selected on aggregate MAPE), versioned
with DVC, served via FastAPI, containerized with Docker, orchestrated
with Airflow, monitored with Prometheus/Grafana (including a
prediction-drift check), and explored interactively via a Streamlit
comparison UI.

See [`results/writeup.md`](results/writeup.md) for the full methodology
and results, and [`results/LEARNING_LOG.md`](results/LEARNING_LOG.md) for
a detailed build log of every tool, decision, and bug hit along the way.

## Results

| Model | Mean per-series MAPE | Median per-series MAPE | Aggregate MAPE |
|---|---|---|---|
| **LightGBM** 🏆 (champion) | 73.45% | 33.60% | **5.82%** |
| Linear Regression | 80.73% | 36.09% | 6.45% |
| XGBoost | 74.71% | 33.55% | 6.48% |
| Prophet | 68.58% | 33.83% | 6.57% |
| Random Forest | 87.17% | 32.79% | 7.29% |
| Naive (seasonal lag-7) | 78.58% | 40.86% | 10.75% |
| ARIMA | 85.21% | 43.87% | 18.46% |

100 series (top-100 by volume), 28-day held-out test horizon. **LightGBM
is registered as the MLflow champion** (`sales_lightgbm@champion`),
selected on aggregate MAPE — see the write-up for why that metric (not
mean per-series MAPE) was used for selection, including a concrete case
where a single outlier series inflated ARIMA's mean per-series MAPE
above even the naive baseline.

## Architecture

```
data/raw (DVC → DagsHub)
   → src/data_prep.py → data/processed/subset_long.csv
   → src/baseline.py            (naive)         --+
   → src/train.py                (Prophet ×100)  --+
   → src/train_arima.py          (SARIMA ×100)   --+--> MLflow tracking + registry
   → src/feature_engineering.py (lag_28-based)     |
       → src/train_ml_models.py (LinReg/RF/XGB/LightGBM, global) --+
   → src/compare_models.py → results/full_model_comparison.csv → champion alias
   → serving/app.py (FastAPI, Prophet) → Docker
   → Prometheus (/metrics: latency, request count, prediction drift) → Grafana
   → ui/app.py (Streamlit): backtest comparison across all 7 approaches

airflow/dags/retrain_dag.py → weekly-scheduled retrain (Docker Compose, LocalExecutor)
```

## Repository structure

```
data/                  DVC-tracked raw + processed M5 data
notebooks/eda.ipynb    Weekly seasonality, holiday/SNAP effects
src/
  data_prep.py          Scope full M5 down to top-100-by-volume subset
  baseline.py           Naive seasonal MAPE reference
  train.py              Prophet training (per-series) + MLflow tracking/registry
  train_arima.py        SARIMA training (per-series, parallelized)
  feature_engineering.py  Leakage-safe lag_28-based features for the global ML models
  train_ml_models.py    Linear Regression / Random Forest / XGBoost / LightGBM (global)
  compare_models.py     Full comparison table + champion selection
ui/
  app.py                 Streamlit: per-series + overall backtest comparison
serving/
  app.py                FastAPI /predict, /health, /metrics (serves Prophet)
  Dockerfile
airflow/
  dags/retrain_dag.py    Weekly retraining DAG
  Dockerfile             Airflow image + isolated task venv
monitoring/
  docker-compose.yaml    serving-app + Prometheus + Grafana
  prometheus.yml
  grafana/                Datasource + dashboard provisioning
docker-compose.yaml      Airflow stack (Postgres + webserver + scheduler)
results/
  writeup.md              Methodology, results, architecture, scope notes
  LEARNING_LOG.md          Full build log with every difficulty and fix
  full_model_comparison.csv
```

## Prerequisites

- Python 3.10
- Docker Desktop
- A Kaggle account (to download the M5 dataset) and a DagsHub account
  (free tier — used as the DVC remote)

## Reproducing this project

### 1. Environment

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux
```

### 2. Data

Requires a Kaggle API token at `~/.kaggle/kaggle.json` (see
[Kaggle's API docs](https://github.com/Kaggle/kaggle-api)) and having
joined the M5 competition on kaggle.com.

```bash
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
cd data/raw && unzip m5-forecasting-accuracy.zip && rm m5-forecasting-accuracy.zip && cd ../..

python src/data_prep.py     # builds data/processed/subset_long.csv
python src/baseline.py      # naive MAPE reference -> results/baseline_summary.csv
```

DVC remote (DagsHub) is already configured in `.dvc/config`; to pull the
already-processed data instead of rebuilding it:

```bash
dvc pull
```

### 3. Train + track

Start a real MLflow tracking server first (not a raw `sqlite:///` path —
see the write-up for why that matters once Docker enters the picture):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 \
  --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
```

Then, in another terminal:

```bash
python src/train.py               # Prophet x100
python src/train_arima.py         # SARIMA x100 (parallelized across CPU cores, ~10-15 min)
python src/feature_engineering.py # builds data/processed/ml_features.csv
python src/train_ml_models.py     # Linear Regression, Random Forest, XGBoost, LightGBM
python src/compare_models.py      # builds results/full_model_comparison.csv, prints the champion
```

Registers all 7 approaches in the MLflow registry. MLflow UI:
http://127.0.0.1:5000. `compare_models.py` picks the champion by
aggregate MAPE but doesn't set the registry alias itself — set it
explicitly once you've looked at the comparison table:

```python
from mlflow import MlflowClient
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
latest = max(client.search_model_versions("name='sales_lightgbm'"), key=lambda v: int(v.version))
client.set_registered_model_alias("sales_lightgbm", "champion", latest.version)
```

### 4. Explore the comparison (Streamlit)

```bash
streamlit run ui/app.py
```

Pick any of the 100 series and any subset of the 7 models to see actual
vs. predicted sales over the test window, plus an overall aggregate-MAPE
comparison tab.

### 5. Serve

```bash
uvicorn serving.app:app --reload
```

- `GET /health`
- `GET /predict?series_id=FOODS_1_218_TX_2_validation&horizon=7`
- `GET /metrics` (Prometheus format)

Or containerized (MLflow server must already be running on the host):

```bash
docker build -f serving/Dockerfile -t m5-forecast-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 m5-forecast-api
```

### 6. Orchestrate retraining (Airflow)

```bash
docker compose up airflow-init      # one-off: migrate DB, create admin user
docker compose up -d                # webserver + scheduler
```

Airflow UI: http://localhost:8080 (`airflow` / `airflow`). Unpause and
trigger `m5_prophet_retrain` to run a retrain on demand, or let it fire
on its weekly schedule.

### 7. Monitoring (Prometheus + Grafana)

```bash
docker compose -f monitoring/docker-compose.yaml up -d --build
```

- Grafana: http://localhost:3000 (`admin` / `admin`) — dashboard is
  auto-provisioned, no manual setup
- Prometheus: http://localhost:9090
- API (this compose file's own instance): http://localhost:8000

## Notes

- AWS/cloud deployment is explicitly out of scope for this project —
  see `results/writeup.md` for the full list of deliberate
  simplifications.
- `results/LEARNING_LOG.md` documents every non-obvious decision and bug
  encountered building this, including several genuine debugging
  sessions (an MLflow artifact-path bug that only surfaces inside a
  container, a Windows console encoding crash, and a hardcoded config
  value that silently broke Airflow-triggered training).
