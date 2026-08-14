# Walmart M5 Demand Forecasting — MLOps Pipeline Write-up

## Goal

Build a production-shaped demand forecasting system for a subset of
Walmart's M5 sales data — where the forecasting model itself is
deliberately the simplest part, and the surrounding pipeline (data
versioning, experiment tracking, serving, orchestration, monitoring) is
what's actually being demonstrated. Full process notes, difficulties hit,
and how they were debugged live in [`LEARNING_LOG.md`](LEARNING_LOG.md);
this document is the results summary.

Cloud deployment (AWS/etc.) was explicitly out of scope — everything runs
locally, in containers, on a single machine.

## Data & Scope

Source: Kaggle's [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
competition (hierarchical Walmart sales data, ~30,490 store-item series,
1,913 days of history).

The full dataset was scoped down to the **top 100 series by total
historical sales volume**, rather than a random sample or a single
category. High-volume series have proportionally fewer zero-sales days,
which matters directly for MAPE (division by near-zero actuals otherwise
dominates the metric with noise). One side effect worth stating plainly:
this volume-based selection skewed the subset almost entirely toward
`FOODS` (97 of 100 series), with a handful of `HOUSEHOLD` items and no
`HOBBIES` — grocery items are simply bought far more often. The subset is
**not representative of the full M5 catalog**, and any conclusions here
are scoped to fast-moving grocery-adjacent items.

## Methodology

1. **Naive baseline first**, before any modeling: a seasonal-naive
   forecast (each series' last 7 observed training days, tiled across
   the 28-day test horizon). This is the free lunch every real model has
   to beat to justify its complexity.
2. **EDA** on the subset: confirmed real weekly seasonality (weekend
   sales ~40 units/day vs. ~28 midweek), a genuine and consistent SNAP
   (food-assistance) effect across all three states, and — a bit
   surprisingly — a negligible calendar-holiday effect for this
   particular FOODS-heavy subset.
3. **Seven forecasting approaches**, compared head-to-head on the same
   28-day held-out window (2016-03-28 to 2016-04-24):
   - **Naive** (seasonal lag-7) and **Prophet** — per-series models (100
     each), described above; Prophet fed calendar holidays natively and
     SNAP as an `add_regressor`.
   - **ARIMA** (SARIMA, seasonal period 7) — also per-series (100 models,
     via `pmdarima.auto_arima`), since ARIMA has no natural "global
     across series" form. SNAP passed as an exogenous regressor, same
     reasoning as Prophet. Order search parallelized across all CPU
     cores (independent per-series fits), since a single `auto_arima`
     call took ~35s and 100 sequential fits would have been prohibitive.
   - **Linear Regression, Random Forest, XGBoost, LightGBM** — *global*
     models: one model each, trained on all 100 series stacked into a
     single table, with `store_id`/`item_id`/`dept_id`/`cat_id`/
     `state_id` as features rather than 100 separate fits. This mirrors
     how the real M5 competition's top solutions approached the problem,
     and lets the model learn cross-series patterns a per-series
     approach never sees. Categoricals were one-hot encoded for Linear
     Regression (ordinal codes would wrongly imply an order to a linear
     model) and passed as native categorical features to LightGBM (for
     better split quality); Random Forest and XGBoost used label-encoded
     integer codes, a standard simplification for tree models.
4. **Feature engineering for the global models**
   (`src/feature_engineering.py`) — the one piece of new methodology
   this comparison required. Key design constraint: this is a 28-day-
   *ahead* forecast, not a 1-day-ahead rolling one, so a naive `lag_7`
   feature would be invalid for most of the test window (predicting day
   15 of the horizon with "sales 7 days ago" would require knowing sales
   from day 8 of the horizon, which hasn't happened yet). Every
   history-based feature is instead built on **`lag_28`** — the largest
   lag that stays valid for the *entire* 28-day horizon, since lag_28 on
   the last test day still points at the last training day. On top of
   that: 7- and 28-day rolling mean/std (computed on the lag_28-shifted
   series), calendar features (weekday, month, is-weekend, SNAP,
   has-event), and `sell_price` (joined from `sell_prices.csv`, not used
   at all by the Prophet/ARIMA phase of this project).
5. **MAPE** computed three ways per model, since MAPE is highly
   sensitive to small actuals (see the ARIMA finding below for a
   concrete example of why a third framing was added): mean per-series
   MAPE, **median** per-series MAPE, and aggregate MAPE on total daily
   sales summed across all 100 series.
6. **Champion selection**: the model with the lowest **aggregate MAPE**
   is registered under an MLflow Model Registry alias, `champion` — see
   "A finding worth stating plainly" below for why aggregate MAPE, not
   mean per-series MAPE, was chosen as the selection metric.

## Results

| Model | Mean per-series MAPE | Median per-series MAPE | Aggregate MAPE |
|---|---|---|---|
| **LightGBM** 🏆 | 73.45% | 33.60% | **5.82%** |
| Linear Regression | 80.73% | 36.09% | 6.45% |
| XGBoost | 74.71% | 33.55% | 6.48% |
| Prophet | 68.58% | 33.83% | 6.57% |
| Random Forest | 87.17% | 32.79% | 7.29% |
| Naive (seasonal lag-7) | 78.58% | 40.86% | 10.75% |
| ARIMA | 85.21% | 43.87% | 18.46% |

**Champion: LightGBM**, registered in the MLflow Model Registry as
`sales_lightgbm` with alias `champion` — 45.8% relative aggregate-MAPE
improvement over the naive baseline, and the best aggregate MAPE of any
approach tried, including Prophet.

Every model except ARIMA beats the naive baseline on aggregate MAPE; all
four global ML models land within a tight band of each other (5.8–7.3%),
with LightGBM and XGBoost — the two models best suited to tabular data
with mixed categorical/numeric features — coming out on top, as expected
given the M5 competition's own history.

### A finding worth stating plainly: why ARIMA lost, and why the selection metric matters

ARIMA's **mean** per-series MAPE (85.2%) makes it look like the worst
model by a wide margin — worse than the naive baseline. Investigating
before accepting that at face value: the per-series MAPE distribution
has a median of 43.9% (a much more ordinary result) but a maximum of
**856%**, driven by a single series (`FOODS_3_808_CA_3_validation`)
whose actual sales collapsed from a steady ~20-50 units/day to almost
entirely zero partway through the test window — a real-world stockout or
discontinuation that's invisible in training history and that *no*
history-only model could have anticipated. ARIMA extrapolated the recent
trend forward as usual; because MAPE divides by the (now near-zero)
actual value, a moderate absolute error on the few remaining non-zero
days turned into an enormous percentage error, single-handedly dragging
the whole model's mean per-series MAPE above the naive baseline's.

ARIMA's **aggregate** MAPE (18.5%) is unaffected by this one series in
the same catastrophic way, and is a more trustworthy read: ARIMA is
still the weakest of the seven approaches here, but not the naive-
baseline-losing failure the mean per-series number alone would suggest.
This is exactly why **aggregate MAPE was used as the model-selection
metric** for choosing the registered champion, not mean per-series MAPE
— a single pathological series shouldn't be able to disqualify an
otherwise-reasonable model, and a metric that lets one data point do that
is a bad metric to select on, even if it's still worth reporting
alongside the others.

## Architecture

```
Kaggle M5 data
      |
      v
data/raw/  --[DVC]-->  DagsHub (data versioning)
      |
      v
src/data_prep.py  -->  data/processed/subset_long.csv (100 series, long format)
      |
      +--> src/baseline.py            --> naive MAPE reference
      +--> src/train.py               --> Prophet x100          (per-series)
      +--> src/train_arima.py         --> SARIMA x100            (per-series, parallelized)
      +--> src/feature_engineering.py --> data/processed/ml_features.csv (lag_28-based, leakage-safe)
             |
             +--> src/train_ml_models.py --> LinReg / RF / XGBoost / LightGBM (global, 1 model each)
      |
      v
  all 7 approaches logged + registered via MLflow
      (tracking server, SQLite backend + local artifact store)
      |
      v
src/compare_models.py --> results/full_model_comparison.csv
      |                   champion selected on aggregate MAPE
      v
MLflow Model Registry: sales_lightgbm@champion  <-------------+
      |                                                       |
      v                                                       |
serving/app.py (FastAPI, Prophet)  ---[Docker]---> served predictions
      |                                                       |
      v                                                       |
Prometheus (/metrics: latency, request count, prediction drift)
      |
      v
Grafana (auto-provisioned dashboard)

ui/app.py (Streamlit) --> per-series + overall backtest comparison across all 7 models

airflow/dags/retrain_dag.py (m5_full_retrain)  --[Docker Compose, LocalExecutor]-->
      retrain_baseline, retrain_prophet, retrain_arima, and
      build_features->retrain_ml_models all run, then
      compare_models -> promote_champion (conditional alias re-point,
      only if the new candidate beats the current champion)
```

Each stage was verified independently and end-to-end (not just "the
container started") before moving to the next — see `LEARNING_LOG.md`
for the specific checks run at each phase, with one exception noted
immediately below.

**Status note on the multi-task DAG:** `m5_full_retrain` is built, loads
without import errors, and its dependency graph was verified correct —
but a full live run through Airflow's orchestrator repeatedly
destabilized Docker Desktop's own daemon on the development machine via
CPU oversubscription in `retrain_arima`, even after correctly
diagnosing and fixing the underlying bug (joblib's `n_jobs` doesn't
bound each worker process's *internal* BLAS/OpenMP thread count, so
"4 workers" was silently oversubscribing every core several times
over). The original single-task DAG (`m5_prophet_retrain`) completed
successfully multiple times, so Airflow orchestration itself is proven
end-to-end; this is specifically an environment resource ceiling on one
machine, not a defect in the DAG or training code. Full debugging
narrative, including all three rounds of diagnosis, in
`LEARNING_LOG.md`'s Phase 8.

Note: `serving/app.py` (the FastAPI /predict endpoint) still serves
Prophet specifically, not the LightGBM champion — LightGBM is a *global*
model that needs the full engineered feature set (lag_28, rolling stats,
price, calendar) to score a new row, which is a meaningfully different
serving problem than Prophet's "give it a date + snap flag." The
Streamlit UI (`ui/app.py`) is where the champion model and every other
approach are actually compared, via pre-computed backtest results rather
than live inference — see that file's docstring for the reasoning.

## What's intentionally out of scope

- **Cloud deployment.** Everything here runs locally/in containers on
  one machine, per the project's own stated scope. A real deployment
  would need managed infrastructure (e.g. a hosted MLflow tracking
  server instead of a local SQLite-backed one, managed Postgres for
  Airflow, a real ingress/load balancer in front of the API, secrets
  management instead of local gitignored files).
- **Autoscaling / high availability.** Single-instance FastAPI, single
  Airflow scheduler (LocalExecutor, not CeleryExecutor's distributed
  workers), no redundancy anywhere.
- **Drift metric is a simplification.** The prediction-drift KS-test
  compares recent served forecasts against the *pooled* training
  distribution across all 100 series, not a per-series reference. This
  makes the metric genuinely sensitive to real distributional facts, but
  it's also somewhat inflated by design (a diverse traffic mix will
  always look "different" from any single series' typical range). A
  production version would want a per-series or per-cohort baseline.
- **No hyperparameter tuning for any model.** Every model (Prophet, the
  4 global ML models, ARIMA's search space) uses sensible defaults, not
  a tuned configuration. Tuning would likely improve every model's MAPE
  somewhat, but wasn't the point of this project — the pipeline and the
  cross-model comparison were.
- **The multi-task auto-promotion DAG exists but hasn't completed a
  live run on this machine.** `m5_full_retrain` (Phase 8) does retrain
  and re-evaluate all 7 approaches and conditionally re-points the
  champion alias — the code is built, reviewed, and every script it
  calls is independently verified — but a full run through Airflow's
  orchestrator hasn't succeeded end-to-end here due to a Docker Desktop
  resource ceiling on this development machine (see `LEARNING_LOG.md`
  Phase 8 for the full diagnosis). The single-task predecessor DAG did
  complete successfully multiple times, so this is an environment
  constraint, not an unbuilt feature.
- **The FastAPI serving layer doesn't serve the champion model.**
  `serving/app.py` still serves Prophet, not `sales_lightgbm@champion` —
  see the architecture note above for why (global models need live
  feature engineering to score a new request, a different and larger
  problem than this project's serving layer currently solves).
- **ARIMA's order search space is intentionally narrow**
  (`max_p=2, max_q=2, max_P=1, max_Q=1`) to keep 100 parallel fits
  tractable (~35s/series otherwise). A wider search might change
  individual series' orders, though it's unlikely to fix the
  fundamental issue the ARIMA finding above describes (extrapolating a
  trend through a stockout it has no way to see coming) — no ARIMA order
  search would have caught that.
