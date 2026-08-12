# Walmart M5 Demand Forecasting — MLOps Pipeline

An end-to-end, production-shaped demand forecasting pipeline built on
Kaggle's M5 Forecasting - Accuracy dataset: Prophet models, tracked and
registered in MLflow, versioned with DVC, served via FastAPI, containerized
with Docker, orchestrated with Airflow, and monitored with
Prometheus/Grafana (including a prediction-drift check).

The model itself (Prophet) is intentionally the simplest part of this
project — the pipeline around it is the point. See
[`results/writeup.md`](results/writeup.md) for the full methodology and
results, and [`results/LEARNING_LOG.md`](results/LEARNING_LOG.md) for a
detailed build log of every tool, decision, and bug hit along the way.

## Results

| Model | Mean per-series MAPE | Aggregate MAPE |
|---|---|---|
| Naive (seasonal, lag-7) | 78.58% | 10.75% |
| **Prophet** | **68.58%** | **6.57%** |

100 series (top-100 by volume), 28-day held-out test horizon. See the
write-up for why both MAPE framings are reported.

## Architecture

```
data/raw (DVC → DagsHub)
   → src/data_prep.py → data/processed/subset_long.csv
   → src/baseline.py (naive MAPE reference)
   → src/train.py (Prophet ×100 → MLflow tracking + Model Registry)
   → serving/app.py (FastAPI, loads from MLflow registry) → Docker
   → Prometheus (/metrics: latency, request count, prediction drift)
   → Grafana (auto-provisioned dashboard)

airflow/dags/retrain_dag.py → weekly-scheduled retrain (Docker Compose, LocalExecutor)
```

## Repository structure

```
data/                  DVC-tracked raw + processed M5 data
notebooks/eda.ipynb    Weekly seasonality, holiday/SNAP effects
src/
  data_prep.py         Scope full M5 down to top-100-by-volume subset
  baseline.py           Naive seasonal MAPE reference
  train.py              Prophet training + MLflow tracking/registry
serving/
  app.py                FastAPI /predict, /health, /metrics
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
python src/train.py
```

Trains and registers all 100 Prophet models. MLflow UI: http://127.0.0.1:5000

### 4. Serve

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

### 5. Orchestrate retraining (Airflow)

```bash
docker compose up airflow-init      # one-off: migrate DB, create admin user
docker compose up -d                # webserver + scheduler
```

Airflow UI: http://localhost:8080 (`airflow` / `airflow`). Unpause and
trigger `m5_prophet_retrain` to run a retrain on demand, or let it fire
on its weekly schedule.

### 6. Monitoring (Prometheus + Grafana)

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
