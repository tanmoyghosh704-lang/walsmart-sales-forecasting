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
   the 28-day test horizon). This is the free lunch any real model has
   to beat to justify its complexity.
2. **EDA** on the subset: confirmed real weekly seasonality (weekend
   sales ~40 units/day vs. ~28 midweek), a genuine and consistent SNAP
   (food-assistance) effect across all three states, and — a bit
   surprisingly — a negligible calendar-holiday effect for this
   particular FOODS-heavy subset.
3. **Prophet**, one model per series (100 total), trained on data through
   2016-03-27 and evaluated on the following 28 days (2016-03-28 to
   2016-04-24), matching M5's own competition horizon. Fed two features
   based on what EDA actually found, not just what the source data
   offered: calendar events via Prophet's native `holidays` parameter,
   and SNAP eligibility as an `add_regressor` (a real signal, and its
   future values are a published schedule, not something that has to be
   forecast).
4. **MAPE** computed two ways per model, since MAPE is highly sensitive
   to small actuals: mean per-series MAPE (excluding zero-actual rows),
   and an aggregate MAPE on total daily sales summed across all 100
   series (where near-zero days are rare, making it a more stable
   number).

## Results

| Model | Mean per-series MAPE | Aggregate MAPE |
|---|---|---|
| Naive (seasonal, lag-7) | 78.58% | 10.75% |
| **Prophet** | **68.58%** | **6.57%** |
| Relative improvement | **12.7%** | **38.9%** |

Prophet beats the naive baseline on both framings, with a much larger
gap at the aggregate level. That's expected: Prophet explicitly models
trend and seasonality (the *shape* of the curve over time), which shows
up clearly once per-series noise is summed away — but per-series MAPE
stays noisier regardless of model quality, since a single bad day on a
low-volume series can still dominate that series' own MAPE. Both numbers
are reported deliberately, rather than leading with whichever looks
better: the per-series number is the more honest one if the real use
case is "forecast this specific item at this specific store."

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
      v
src/baseline.py  -->  naive MAPE reference
src/train.py     -->  Prophet x100, logged + registered via MLflow
      |                (tracking server, SQLite backend + local artifact store)
      v
MLflow Model Registry  <---------------------+
      |                                       |
      v                                       |
serving/app.py (FastAPI)  ---[Docker]---> served predictions
      |                                       |
      v                                       |
Prometheus (/metrics: latency, request count, prediction drift)
      |
      v
Grafana (auto-provisioned dashboard)

airflow/dags/retrain_dag.py  --[Docker Compose, LocalExecutor]-->
      weekly-scheduled re-run of src/train.py
```

Each stage was verified independently and end-to-end (not just "the
container started") before moving to the next — see `LEARNING_LOG.md`
for the specific checks run at each phase.

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
- **No per-series hyperparameter tuning.** All 100 Prophet models use
  the same default configuration (plus the holidays/SNAP features found
  during EDA). Tuning changepoint priors or seasonality strength
  per-series would likely improve MAPE further but wasn't the point of
  this project — the pipeline was.
- **No multi-task / auto-promotion DAG.** The Airflow DAG is
  single-task (retrain on a schedule) per the project's own suggested
  build order; a validation-and-conditional-promotion step (only
  registering a new model version if it beats the current one on
  held-out MAPE) is a natural next step, not yet built.
