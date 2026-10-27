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
| **LightGBM** (champion) | 73.45% | 33.60% | **5.82%** |
| KNN (k=7, distance-weighted) | 74.42% | 30.92% | 5.90% |
| Linear Regression | 80.73% | 36.09% | 6.45% |
| XGBoost | 74.71% | 33.55% | 6.48% |
| Random Forest | 87.17% | 32.79% | 7.29% |
| Naive (seasonal lag-7) | 78.58% | 40.86% | 10.75% |
| ARIMA | 85.21% | 43.87% | 18.46% |

LightGBM is registered as the MLflow champion (`sales_lightgbm@champion`),
selected on aggregate MAPE. That's a 45.8% improvement over the naive
baseline -- though KNN came within 0.08 points of dethroning it, which is
a good reminder that a dead-simple model (no gradient boosting, no
hyperparameter search) can be surprisingly competitive once the feature
engineering is doing the real work.

## Methodology

**Scope.** The full M5 dataset covers ~30,490 store-item series. This
project uses the top 100 by total historical volume, which keeps
zero-sales days rare enough for MAPE to stay meaningful. That selection
skews heavily toward the `FOODS` category (grocery items sell far more
often than hobbies or household goods), which is a real bias in the
subset, not an artifact of sampling error.

**Models.** KNN and ARIMA are trained per series (100 independent
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
Promotion is restricted to the four global models, since KNN and ARIMA
are each 100 separately-registered models with no single name an alias
could represent.

## Architecture

Three images, one Docker Compose stack (`mlops-network`), one DVC pipeline:

```
data/raw/sales_train_validation.csv (DVC remote: DagsHub, static Kaggle seed)
data/raw/sales_train_evaluation.csv (same series, 28 more real days --
                                      stands in for "new data arriving",
                                      since M5 itself is a static dataset)
   |
   v  (Airflow DAG m5_full_retrain, weekly)
src/ingest_new_data.py  -- absorb the next 7-day chunk into
                            data/processed/sales_history.csv (runs directly
                            in the airflow container; stops once all 28
                            days are absorbed)
   -> dvc repro -s validate              (src/validate_raw.py: schema/quality gate)
   -> dvc repro -s prepare               (src/data_prep.py -> subset_long.csv)
   -> dvc repro -s feature_engineering   (src/feature_engineering.py -> ml_features.csv, encoders.json)
   -> src/commit_artifacts.py -- commit + push the resulting dvc.lock to
      GitHub (runs in parallel with training below, not blocking it)
   -> src/baseline.py / train_knn.py / train_arima.py / train_ml_models.py
      -- one-shot m5-training containers --> MLflow (DagsHub-hosted
      tracking server + registry; not a local container)
   -> src/compare_models.py -> results/full_model_comparison.csv
   -> src/promote_champion.py -> sets the "champion" alias on exactly one
      registered model (and clears it from the others)
   -> reload the running serving-app container

serving/app.py (FastAPI, its own container)
   -> resolves whichever model currently carries the "champion" alias from
      MLflow, builds the same engineered features live via src/features.py
   -> GET /predict -> records the request's feature vector into a
      300-sample-per-feature rolling buffer, KS-tests it against the last
      feature_engineering snapshot -> Prometheus gauges
   -> Prometheus -> Grafana dashboard (latency, request rate, drift score,
      drift alert, per-feature KS breakdown)

ui/app.py (Streamlit): backtest comparison across all approaches, run ad hoc
```

**Three Dockerfiles**, composed together in `docker-compose.yaml`:
- `airflow/Dockerfile` -- a single `airflow standalone` container. Has
  git/dvc/pandas baked in and the whole repo bind-mounted at
  `/opt/airflow/project`, so ingestion, `dvc repro`, and the git-commit
  step all run as plain `BashOperator` commands directly in this
  container. It talks to the host Docker daemon (socket-mounted) only to
  launch one-shot training containers for the actual retrain/compare/
  promote steps -- it never installs the heavy ML libraries itself.
- `training/Dockerfile` -- the ML toolchain (pandas, scikit-learn,
  LightGBM, XGBoost, statsmodels, pmdarima, MLflow). Launched as one-shot
  containers per DAG task via `DockerOperator`, project directory
  bind-mounted in. No dvc/git -- those stay in the airflow container.
- `serving/Dockerfile` -- FastAPI + whichever ML libraries are needed to
  load any of the 4 promotable model flavors from the registry.

**No MLflow container.** MLflow is DagsHub-hosted: every DagsHub repo gets
a free MLflow-compatible tracking server + model registry (and UI) at
`https://dagshub.com/<owner>/<repo>.mlflow`. Every component that talks to
MLflow (`src/common.py`'s `MLFLOW_TRACKING_URI_DEFAULT`, every training
script, `serving/app.py`) points there instead of a self-hosted
`mlflow server` + SQLite container -- one less compose service, one less
local backend store to keep consistent across containers. Auth is HTTP
basic via `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` (a DagsHub
access token), read automatically by the `mlflow` client from `.env`.

## Repository structure

```
data/                  DVC-tracked raw + processed M5 data
dvc.yaml               DVC pipeline: validate -> prepare -> feature_engineering
notebooks/eda.ipynb    Weekly seasonality, holiday/SNAP effects
src/
  common.py              Shared train/test split, MAPE, champion-alias constants
  features.py            Shared feature engineering (training AND live serving)
  ingest_new_data.py     DAG step 1: absorb new data into sales_history.csv
  validate_raw.py        DVC "validate" stage: raw-data schema/quality gate
  data_prep.py           DVC "prepare" stage: scope to top-100-by-volume subset
  baseline.py            Naive seasonal MAPE reference
  train_knn.py           Per-series KNN regressor + MLflow tracking/registry, joblib-parallel
  train_arima.py         Per-series SARIMA, joblib-parallel
  feature_engineering.py DVC "feature_engineering" stage (wraps src/features.py)
  train_ml_models.py     Linear Regression / Random Forest / XGBoost / LightGBM
  compare_models.py      Full comparison table, sorted by aggregate MAPE
  promote_champion.py    Conditional MLflow champion alias promotion
  commit_artifacts.py    Commit + push dvc.lock after each retrain's data pipeline
ui/
  app.py                 Streamlit: per-series + overall backtest comparison
serving/
  app.py                 FastAPI /predict, /health, /metrics -- loads the
                          live "champion" alias, not a hardcoded model family
  Dockerfile
  requirements.txt
training/
  Dockerfile             ML toolchain only (no dvc/git), launched per Airflow task
  requirements.txt
airflow/
  dags/retrain_dag.py    m5_full_retrain: ingest -> dvc repro -> commit ->
                          retrain -> compare -> promote -> reload serving
  Dockerfile              Single `airflow standalone` image: git+dvc+pandas
  requirements.txt
monitoring/
  prometheus.yml
  grafana/               Datasource + dashboard provisioning
docker-compose.yaml      The whole stack: airflow (standalone), serving-app,
                          Prometheus, Grafana (training is build-only; MLflow
                          is DagsHub-hosted, not a compose service)
.env.example             Template for HOST_PROJECT_DIR / GITHUB_TOKEN /
                          MLFLOW_TRACKING_USERNAME / PASSWORD (copy to .env)
results/                 MAPE tables + forecasts written by the scripts above (gitignored)
ROADMAP.txt              Phase-by-phase build guide
```

## Prerequisites

- Python 3.10
- Docker Desktop
- A Kaggle account (to download the M5 dataset) and a DagsHub account
  (free tier) -- DagsHub does double duty here: it's both the DVC remote
  *and* the MLflow tracking server + model registry. You'll need a
  DagsHub access token (Settings -> Tokens) for both -- DVC reads it from
  `.dvc/config.local`, MLflow reads it from `.env` (see `.env.example`);
  they authenticate independently, so the same token goes in both places
  but isn't automatically shared between them.

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
```

The DVC remote is already configured in `.dvc/config` (DagsHub). To pull
the already-processed data instead of rebuilding it: `dvc pull`.

### 3. Run the DVC pipeline

```bash
dvc repro
```

Runs the three stages defined in `dvc.yaml` in order -- `validate`
(`src/validate_raw.py`, a schema/quality gate on the raw CSVs), `prepare`
(`src/data_prep.py` -> `data/processed/subset_long.csv`), and
`feature_engineering` (`src/feature_engineering.py` -> `ml_features.csv` +
`encoders.json`). `dvc repro -s <stage>` runs just one.

### 4. Train and track

MLflow is DagsHub-hosted -- there's no server to start. Everything below
talks to it via `MLFLOW_TRACKING_URI` (defaults to
`https://dagshub.com/<owner>/<repo>.mlflow`, set in `src/common.py`).
Outside Docker, export the two auth env vars yourself first (inside the
compose stack, `.env` + `env_file` does this for you):

```bash
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-access-token>

python src/baseline.py            # naive MAPE reference
python src/train_knn.py           # KNN x100, joblib-parallel
python src/train_arima.py         # SARIMA x100, joblib-parallel
python src/train_ml_models.py     # Linear Regression, Random Forest, XGBoost, LightGBM
python src/compare_models.py      # builds results/full_model_comparison.csv
python src/promote_champion.py    # sets the MLflow champion alias
```

MLflow UI: `https://dagshub.com/<owner>/<repo>` -> **Experiments** tab
(same DagsHub repo used as the DVC remote).

### 5. Explore the comparison

```bash
streamlit run ui/app.py
```

### 6. Serve

```bash
uvicorn serving.app:app --reload
```

- `GET /health` -- also reports which model currently holds the champion alias
- `GET /predict?series_id=FOODS_1_218_TX_2_validation&horizon=7`
- `GET /metrics`

`/predict` resolves whichever model currently carries the MLflow
`champion` alias (cached for `MODEL_CACHE_TTL_SECONDS`, default 300s) and
builds its full engineered feature vector live via `src/features.py` --
the same code path used at training time.

### 7. Run the whole stack

```bash
cp .env.example .env   # fill in HOST_PROJECT_DIR, MLFLOW_TRACKING_USERNAME/PASSWORD
                        # (and GITHUB_TOKEN if you want auto-push)
docker compose build
docker compose up -d
```

This brings up one `airflow` container (runs `airflow standalone` --
self-initializes its own metadata DB in a named volume and creates an
admin user on first boot; the generated password is in
`docker compose logs airflow` or at
`/opt/airflow/standalone_admin_password.txt` inside the container), the
FastAPI serving app, Prometheus, and Grafana, all on one `mlops-network`
bridge network. There's no MLflow container to start -- every service
reaches DagsHub's hosted MLflow instance directly.

- Airflow: http://localhost:8080 (user `admin`, password from the logs
  above) -- trigger `m5_full_retrain` manually or let it run on its
  weekly schedule:
  - `ingest_new_data`, `dvc_validate`, `dvc_prepare`,
    `dvc_feature_engineering`, `commit_dvc_lineage` run directly inside
    the airflow container (`BashOperator`) -- it already has git, dvc,
    and pandas, and the whole repo mounted at `/opt/airflow/project`.
  - `retrain_baseline`, `retrain_knn`, `retrain_arima`,
    `retrain_ml_models`, `compare_models`, `promote_champion` each run in
    their own one-shot `m5-training` container (`DockerOperator`) --
    these need the heavier ML libraries, kept out of the airflow image.
  - `reload_serving` restarts the running `serving-app` container
    (`PythonOperator` + Docker SDK) so it picks up the new champion
    immediately instead of waiting out its own cache TTL.
- MLflow: `https://dagshub.com/<owner>/<repo>` -> **Experiments** tab (DagsHub-hosted, not local)
- API: http://localhost:8000
- Grafana: http://localhost:3000 (`admin` / `admin`)
- Prometheus: http://localhost:9090

## Scope

Everything runs locally/in containers on a single machine; cloud
deployment, autoscaling, and high availability are out of scope. No
hyperparameter tuning was performed on any model. The champion model's
feature-based forecast horizon is capped at 28 days (`LAG` in
`src/features.py`) since lag/rolling features beyond that would require
feeding the model's own predictions back in as "sales" (recursive
forecasting), which this project doesn't implement.
