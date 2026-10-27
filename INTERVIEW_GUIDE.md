# Interview Guide: This Project's MLOps Pipeline

This is a study document, not project documentation — it exists so you can
explain and defend every design decision in this repo out loud. For the
historical build log (phase-by-phase, with gotchas as they were hit), see
`ROADMAP.txt`. For results/methodology, see `README.md`. This file answers
"why does it work this way" and "what will they ask me about it."

## 1. The one-paragraph pitch

This is an end-to-end MLOps pipeline for demand forecasting on Kaggle's M5
dataset (Walmart sales). Seven forecasting approaches are benchmarked on
100 high-volume series. The point isn't the models (they're intentionally
simple) — it's the pipeline around them: DVC for data versioning and a
real multi-stage pipeline, MLflow for experiment tracking and a model
registry with alias-based promotion, Airflow for weekly retrain
orchestration, Docker for reproducible environments split by concern,
FastAPI for serving whichever model is currently promoted, and
Prometheus/Grafana for latency and data-drift monitoring. Every piece
exists to answer a specific production question: *where did this model
come from, is it better than what's live, is it actually what's serving
traffic, and is production data still what it was trained on?*

## 2. Architecture at a glance

```
            ┌─────────────────────────────────────────────┐
            │  airflow (single "standalone" container)     │
            │  git + dvc + pandas baked in, repo mounted    │
            │                                                │
            │  ingest_new_data → dvc_validate → dvc_prepare │
            │       → dvc_feature_engineering               │
            │       → commit_dvc_lineage (git push)         │
            │       → [4x DockerOperator training tasks]    │
            │       → compare_models → promote_champion     │
            │       → reload_serving (restarts serving-app) │
            └───────────────┬────────────────────────────────┘
                             │ DockerOperator (docker socket)
                             ▼
            ┌─────────────────────────────┐
            │  m5-training (one-shot)      │   ──→  MLflow: DagsHub-hosted
            │  baseline / knn / arima /     │         tracking server +
            │  ml_models / compare /       │         model registry (NOT
            │  promote_champion            │         a local container)
            └─────────────────────────────┘
                             │ champion alias
                             ▼
            ┌─────────────────────────────┐        ┌───────────────┐
            │  serving-app (FastAPI)       │ ──────▶│ Prometheus    │──▶ Grafana
            │  resolves live champion,     │  /metrics
            │  builds features live,      │
            │  tracks drift                │
            └─────────────────────────────┘
```

Three purpose-built Docker images (`airflow/`, `training/`, `serving/`),
one `docker-compose.yaml`, one `dvc.yaml` pipeline, one Airflow DAG
(`m5_full_retrain`). No MLflow image — see 3.2.

## 3. Component-by-component: what, why, and the alternative you'd be asked about

### 3.1 Data: DVC + the ingestion simulation

**What's there:** `dvc.yaml` defines three stages — `validate` → `prepare`
→ `feature_engineering` — each with explicit `deps`/`outs`, so `dvc repro`
only reruns a stage if its inputs actually changed. `data/raw.dvc` is a
directory-level pointer versioning the raw Kaggle CSVs against a DagsHub
remote (`dvc push`/`dvc pull`).

Ahead of the DVC pipeline sits `src/ingest_new_data.py`, which is
**deliberately not a DVC stage**. M5 is a static historical dataset — there's
no real live feed — so this script simulates one: `sales_train_evaluation.csv`
(already in `data/raw/`) has the exact same 100 series as
`sales_train_validation.csv` plus 28 more real days. Each DAG run absorbs a
7-day chunk of those into `data/processed/sales_history.csv`, which is what
the rest of the pipeline actually reads. Once all 28 days are absorbed,
further runs are a no-op.

**Why not make ingestion a DVC stage?** DVC caches a stage by hashing its
`deps`. Ingestion's two inputs (the static seed and source CSVs) never
change, so DVC would hash them once, cache the result, and skip the stage
forever — exactly wrong for something that's supposed to make incremental
progress each run. DVC's model assumes stages are pure functions of their
deps; this isn't one, by design.

**The bug you should be ready to explain:** the first version of ingestion
merged the new day-columns onto history using the `id` column — and
silently produced an all-NaN result. M5's own convention gives `id` a
*different suffix* in each file (`..._validation` vs `..._evaluation`) for
the same series, so a merge on `id` matches nothing. Fixed by merging on
the five other identifier columns (`item_id`/`dept_id`/`cat_id`/`store_id`/
`state_id`), which are suffix-free and jointly unique — verified that before
shipping the fix, and added a hard-fail guard (`RuntimeError` if any row
comes out with a null new-day value) so a similar mismatch can never again
pass silently. This is a good "tell me about a bug you found" story: it
wasn't caught by a type system or a test, it was caught by actually running
the code against real data and checking the output, not just the exit code.

**Why DVC at all, instead of just re-running scripts on raw files?** Two
separate jobs: (1) version large data files outside git (git can't
diff/store 450MB CSVs sanely); (2) make the data *pipeline* itself
reproducible and cacheable — `dvc repro` skips work that's already done,
and `dvc.yaml` + `dvc.lock` together are a record of "this exact file,
produced by this exact command, from this exact input hash."

### 3.2 Experiment tracking + model registry: MLflow, hosted on DagsHub

**What's there:** no MLflow container. Every DagsHub repo gets a free
MLflow-compatible tracking server + model registry (and UI) at
`https://dagshub.com/<owner>/<repo>.mlflow` — this project was already
using DagsHub as its DVC remote, so `MLFLOW_TRACKING_URI_DEFAULT`
(`src/common.py`) just points there too, one repo doing double duty rather
than a second service to run and keep healthy. Auth is HTTP basic
(`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD`, the latter a
DagsHub access token) — the `mlflow` client reads both automatically from
the environment, so no code changed when this moved off a self-hosted
server, only the URI and two new env vars threaded through `.env` →
`docker-compose.yaml` → the DAG's training-container launches →
`serving/app.py`. Every training run logs params/metrics and registers a
model exactly as it would against a self-hosted server. Promotion uses
**registry aliases**, not the deprecated "stages" concept:
`client.set_registered_model_alias(name, "champion", version)`.

**Why not self-host (`mlflow server` + SQLite), which is what this project
did initially?** A bare SQLite backend store works until more than one
process/container needs to resolve model artifacts — concurrent writers to
one SQLite file across containers is a correctness risk, so a self-hosted
setup still needs a real server process, its own volume-mounted backend
store and artifact store, a healthcheck, and a compose service to keep
alive. DagsHub's hosted instance is the same MLflow server concept, just
operated for you — no backend store to provision, no artifact storage to
manage, no extra container in the stack. The trade-off going the other
way: a self-hosted server keeps everything on one machine with no external
dependency, which matters if you need to work offline or avoid any
third-party service — a legitimate reason to choose it instead.

**The two-auth-flows gotcha, worth mentioning unprompted:** DVC's DagsHub
credentials (`.dvc/config.local`) and MLflow's (`.env`) authenticate
*independently* against the same DagsHub repo, even though the same access
token value can be reused for both. Setting one up doesn't set up the
other — easy to assume otherwise since it's "the same DagsHub account,"
but DVC and the `mlflow` client never actually share credentials.

**Why aliases over stages?** MLflow's old "stage" model (`Staging`/
`Production`/`Archived`) is a fixed global enum — every model in the
registry fights over the same four buckets. Aliases are just named
pointers you define yourself (`champion`, `canary`, whatever), scoped per
registered model name, and that's the mechanism MLflow itself now
recommends.

**Why only 4 of 7 models are "promotable":** `promote_champion.py` only
considers `linear_regression`/`random_forest`/`xgboost`/`lightgbm` — the
four *global* models (one registered model each, trained on all 100 series
stacked). KNN and ARIMA are trained **per series** — 100 independent
registered models each (`knn_<series_id>`, `arima_<series_id>`) — so
there's no single name a `champion` alias could point to. This is a direct
design trade-off of "100 specialized models" vs "1 model that generalizes
across entities," and you should be able to say which one you'd pick for a
real production system (global models, because they're the ones that are
actually operable — one alias, one promotion decision, one thing to
monitor, instead of 100).

**Promotion logic, in one sentence:** a new candidate only overwrites the
current champion if its aggregate MAPE is strictly better — a regressing
retrain cannot silently demote a good model.

**A bug worth bringing up unprompted:** the first version of promotion set
the `champion` alias on the new best model but never *removed* it from
whichever model held it before, if the winner changed families between
retrains (e.g. XGBoost champion this week, LightGBM next week). Since
alias lookup only checked "does this one model have the alias," and nothing
ever cleared the old one, two models could simultaneously carry `champion`
— and which one `get_current_champion()` returned depended on dict
iteration order, not on which was actually most recent. Fixed by explicitly
clearing the alias from every other promotable model on each promotion, and
added a warning if more than one is ever found. This is the kind of bug
that only shows up once you actually trace "what does the *reader* of this
alias see," not from looking at the writer in isolation.

### 3.3 Training: per-series vs. global, and why the split

- **KNN / ARIMA** (`train_knn.py`, `train_arima.py`): 100 independent
  models, one per series, `joblib`-parallelized across CPU cores (fit
  outside any MLflow run context, log sequentially afterward, since
  concurrent writes inside a `joblib.Parallel` worker would need
  per-worker MLflow run handling). KNN's slot was originally SVM
  (`SVR`, RBF kernel) — swapped out deliberately, not because SVM was
  broken: the global-model side already covers linear + two flavors of
  boosted trees + bagged trees, so the more informative per-series
  "non-time-series ML" pick was something with a genuinely *different*
  inductive bias than another kernel/margin method — pure instance-based
  local averaging. `KNeighborsRegressor(n_neighbors=7, weights="distance")`,
  same `StandardScaler` pipeline (KNN's Euclidean distance is exactly as
  scale-sensitive as SVR was), same features, same MLflow logging
  pattern — effectively a drop-in swap of the estimator inside an
  otherwise-unchanged script. `n_neighbors=7` isn't sklearn's default
  (5) — it's one seasonal week, deliberately echoing the `dow_sin`/
  `dow_cos` weekly-seasonality encoding already in the feature set.
  Result: 74.42% mean / **30.92% median** / 5.90% aggregate per-series
  MAPE — second place by aggregate MAPE (0.08 points behind the LightGBM
  champion), and its median per-series MAPE is the *best of any model in
  the comparison*, including the champion. Good talking point: mean and
  median per-series MAPE can rank models differently, which is exactly
  why this project reports both instead of collapsing to one number.
- **Linear Regression / Random Forest / XGBoost / LightGBM**
  (`train_ml_models.py`): single global model each, trained on all 100
  series stacked together with entity IDs (item/dept/cat/store/state) as
  categorical features — the same approach the top M5 competition
  solutions used, because it lets the model borrow statistical strength
  across similar series instead of learning each one from scratch.
- **Naive seasonal (lag-7)** (`baseline.py`): the reference point everything
  else has to beat. Champion beats it by 45.8% on aggregate MAPE.

**Feature leakage guard you should be able to explain cold:** features are
built on a **28-day lag**, not a 1-day lag, because the forecast horizon is
28 days. A shorter lag would mean that by day 20 of the forecast, the
model's own "feature" would require knowing sales that haven't happened
yet. Always use the largest lag that stays valid across the *entire*
horizon, not just the first step of it.

**Why three MAPE framings (mean per-series / median per-series /
aggregate), and why aggregate for champion selection:** one ARIMA series
had an 856% single-series MAPE from a real stockout mid-test-window — that
single outlier pulled ARIMA's *mean* per-series MAPE above the naive
baseline, even though ARIMA was doing fine everywhere else. Aggregate MAPE
(sum actuals and forecasts across all series *first*, then compute one
MAPE) is far less sensitive to one bad series, which is why it's the
promotion metric — but reporting all three is what makes that outlier
visible in the first place instead of hidden.

### 3.4 Serving: FastAPI, and the bug that mattered most

**What's there:** `serving/app.py` resolves whichever registered model
currently carries the `champion` alias (cached for `MODEL_CACHE_TTL_SECONDS`,
default 300s, so a new promotion is picked up without a restart), then
builds that model's exact feature vector **live**, at request time, using
`src/features.py` — the same module `feature_engineering.py` uses at
training time. `GET /predict?series_id=...&horizon=...` returns a forecast;
`GET /health` reports which model is currently live; `/metrics` exposes
Prometheus data.

**The bug that was the actual headline issue in this whole project:** the
original serving code always loaded per-series SVM models by `/latest`
version — completely independent of whatever `promote_champion.py` had
just promoted. The model that got promoted (a global model like LightGBM)
was *never* the model answering requests. Two systems that both looked
"done" in isolation, silently disconnected. If you get one question right
in an interview about this project, it should be this one: **a model
registry promotion is only meaningful if the serving layer actually reads
from it** — which sounds obvious stated directly, and is exactly the kind
of thing that's invisible when you build serving and promotion as two
separate, both-individually-correct pieces of work.

**Second bug, subtler, found while fixing the first:** `feature_engineering.py`
computed category encoders (`pd.factorize`) but never persisted them. Fine
for one offline run where the whole dataset goes through factorize
together — but once serving needs to encode a *single* incoming request,
there's no way to reproduce the same category→code mapping training used.
Worse: for LightGBM specifically, a naive `.astype("category")` on a
single-row request would only see that one row's category value, re-number
its code to 0, and silently feed the booster the wrong split value — since
LightGBM's categorical split points are the *codes*, not the labels. Fixed
by persisting `encoders.json` and, for LightGBM specifically, reconstructing
the *full* training-time category universe (`pd.Categorical(codes,
categories=range(n))`) before prediction, not just casting the dtype.
Verified this preserved the exact training-time code for a single-row
input before considering it fixed, not just that it ran without error.

**Why cap the forecast horizon at 28 days (`LAG` in `features.py`):** the
`lag_28` feature needs real sales from 28 days before the forecast date.
That only holds while every forecast day is within 28 days of the last
known actuals. Beyond that you'd need to feed the model's own predictions
back in as if they were real sales (recursive forecasting) — a real
technique, just not one this project implements, and the horizon cap makes
that limitation explicit instead of silently producing degraded forecasts.

### 3.5 Drift monitoring: the deque + KS-test design

**What's there:** per numeric feature (`lag_28`, rolling means/std, `wday`,
`month`, `year`, `is_weekend`, `has_event`, `snap`, `sell_price`), a
`collections.deque(maxlen=300)` buffer of the feature values actually fed
into recent `/predict` calls. Once a buffer has ≥30 samples, a two-sample
Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) compares it against the
same feature's column in the most recent `feature_engineering` snapshot
(`ml_features.csv`, reloaded on the same TTL as the champion model, so a
fresh retrain's reference distribution gets picked up too). The mean KS
statistic across features becomes `prediction_drift_score`; it crossing
`DRIFT_KS_THRESHOLD` (default 0.3) sets `prediction_drift_alert` to 1.
Grafana visualizes the aggregate score, the alert, and the per-feature
breakdown (so you can see *which* feature is actually drifting, not just
that something is).

**Why KS test specifically:** it's distribution-free (no assumption of
normality), it's sensitive to differences anywhere in the distribution (not
just the mean), and it gives both an effect-size-like statistic (0 = no
drift, 1 = fully disjoint) and a p-value. For a judgment call like "has
production data meaningfully diverged from training data," a
nonparametric, two-sample test is the standard first tool.

**Why a bounded deque instead of accumulating forever:** drift is about
*recent* behavior, not all-time behavior — an unbounded buffer would mean
one week of drift five months ago still half-weighs today's comparison
forever. A rolling window of a fixed size keeps the comparison meaningful
("have the last ~300 requests looked different"), and bounding it also
bounds memory.

**Why feature-level drift, not output-level drift:** the earlier version of
this check compared the *pooled sales values the API predicted* against the
*pooled training sales distribution* — a reasonable first idea, but it
conflates "the model's predictions changed" (could just mean a different
champion got promoted) with "the input data distribution changed" (the
thing you actually want an early warning for). Comparing input features
against the input features the current champion was trained on is the more
direct signal, and breaking it out per-feature tells you *which* input
changed, which is what you'd actually act on.

**What this doesn't do, and you should say so unprompted:** there's no
ground-truth accuracy check in production (that would need waiting for
actual sales to arrive, which happens offline via each retrain's MAPE
comparison). Drift is a *leading, proxy* signal you can check continuously
without ground truth; accuracy is the *lagging, authoritative* signal you
check periodically once ground truth exists. Good pipelines have both, for
exactly that reason — drift tells you something might be wrong *now*;
accuracy tells you whether it actually was, later.

### 3.6 Orchestration: Airflow, and why BashOperator here but DockerOperator there

**What's there:** one DAG, `m5_full_retrain`, `@weekly`, `catchup=False`,
`max_active_runs=1`. Airflow itself runs as a single `airflow standalone`
container (webserver + scheduler + triggerer in one process, its own SQLite
metadata store in a named volume) — not four containers
(postgres/init/webserver/scheduler). The whole repo is bind-mounted into it
at `/opt/airflow/project`.

**Task-to-operator mapping, and the reasoning behind each choice:**

| Task | Operator | Why |
|---|---|---|
| `ingest_new_data`, `dvc_validate`, `dvc_prepare`, `dvc_feature_engineering`, `commit_dvc_lineage` | `BashOperator` | Only need git/dvc/pandas, which are installed directly in the airflow image; the project is already mounted, so these are just shell commands against it — no reason to pay for a second container launch |
| `retrain_baseline`/`retrain_knn`/`retrain_arima`/`retrain_ml_models`, `compare_models`, `promote_champion` | `DockerOperator` | Need the heavy ML stack (scikit-learn, LightGBM, XGBoost, statsmodels, pmdarima, mlflow) — deliberately **not** installed in the airflow image, so these launch one-shot containers from a separate `m5-training` image instead |
| `reload_serving` | `PythonOperator` + Docker SDK | Restarting an *already-running, compose-managed* long-lived service doesn't fit DockerOperator's model (which always creates a brand-new one-shot container and removes it after) — this just talks to the Docker daemon directly to restart the existing `serving-app` container |

**The underlying principle, if asked "why isn't everything just one
operator type":** match the tool to what the task actually needs. A task
that only needs lightweight, already-present dependencies and the already-
mounted repo doesn't need a new container; a task that needs a large,
specific dependency set that shouldn't bloat every other task's image does;
and an action that's really "control an existing service" rather than "run
a batch job" doesn't fit a one-shot-container abstraction at all, so
forcing it into `DockerOperator` just to make every task the same operator
type would be consistency for its own sake, not for a real reason.

**DAG dependency graph, and why it's shaped this way:**

```
ingest_new_data → dvc_validate → dvc_prepare ─┬→ retrain_baseline ─┐
                                               ├→ retrain_knn ──────┤
                                               ├→ retrain_arima ────┤
                                               └→ dvc_feature_engineering ─┬→ retrain_ml_models ─┤
                                                                            └→ commit_dvc_lineage  │
                                                                                                    ▼
                                                                                          compare_models
                                                                                                    │
                                                                                                    ▼
                                                                                          promote_champion
                                                                                                    │
                                                                                                    ▼
                                                                                            reload_serving
```

Baseline/KNN/ARIMA only need `subset_long.csv` (the `prepare` stage's
output), so they fan out and run in parallel with `feature_engineering`
rather than waiting on it — `train_ml_models` is the only one that actually
needs the engineered features. `commit_dvc_lineage` branches off
`feature_engineering` too and runs in parallel with the training fan-out,
not serialized after it: the git commit only concerns the data pipeline's
state, so there's no reason for it to block or wait on model training.
Everything converges at `compare_models` because that's the first step that
needs every model's results.

**Why `max_active_runs=1` and `catchup=False`:** this is a weekly retrain
over a monotonically growing dataset — two overlapping runs could ingest
the same chunk twice or promote off half-written results, and backfilling
every missed weekly interval since `start_date` would mean replaying
history that's no longer relevant (and would re-run `ingest_new_data`
repeatedly before you want it to).

### 3.7 Containerization: 3 images + 1 compose file, and the Windows wrinkle

**The three images and what's deliberately NOT in each:**
- `airflow/Dockerfile` — orchestration + git/dvc/pandas. No ML libraries.
- `training/Dockerfile` — the ML stack. No dvc, no git (moved out in the
  second hardening pass, once BashOperator took over the dvc/git tasks —
  the training image literally never calls either).
- `serving/Dockerfile` — FastAPI + every ML library needed to *load* (not
  train) any of the 4 promotable model flavors, since it doesn't know in
  advance which one is champion.

**Why split instead of one image with everything:** image size/build time
(training's dependency list is large and has no reason to be in the
always-running serving or airflow containers), blast radius (a dependency
bump for training can't accidentally break serving), and conceptual
clarity (each image's dependency list *is* a statement of what that
component actually needs to do its job).

**Docker-outside-of-docker, and the real limitation to be upfront about:**
`DockerOperator` tasks launch *sibling* containers by talking to the host's
Docker daemon through a mounted socket (`/var/run/docker.sock`) — the
airflow container isn't nesting Docker inside itself, it's asking the one
real daemon (on the host) to start another container next to it. The
catch: when that bind-mount specifies a path for the new container, the
daemon resolves it against *the host's* filesystem, not against the
airflow container's. So `HOST_PROJECT_DIR` has to be the repo path as the
*Docker daemon* sees it — on Linux/macOS that's just the repo's absolute
path; on Windows with Docker Desktop's WSL2 backend, it's the `/mnt/c/...`
form. This is a real, known rough edge of the docker-outside-of-docker
pattern, not something papered over — the project's `.env.example`
documents it explicitly rather than assuming it "just works."

**One network, pinned name:** every service sits on one bridge network,
`mlops-network`, with its name pinned explicitly (`networks.mlops-network.name`)
rather than left to Compose's default project-prefixed naming — because the
DAG's `DockerOperator` calls need to pass that literal string as
`network_mode` so the training containers they launch land on the same
network as `serving-app` and can resolve it by service name (MLflow
itself is DagsHub-hosted, not on this network at all — reached over the
public internet instead, see 3.2).

## 4. A walkthrough of one full DAG run (narrate this out loud)

1. **`ingest_new_data`** absorbs the next 7-day chunk of "new" sales data
   into `data/processed/sales_history.csv` (or no-ops if there's nothing
   new left).
2. **`dvc_validate`** runs schema/quality checks (`src/validate_raw.py`) —
   non-null, non-negative sales, valid SNAP flags, positive prices — and
   fails the whole run loudly if the data doesn't look like M5 data should.
3. **`dvc_prepare`** reshapes wide-format sales into the long-format 100-series
   subset (`subset_long.csv`).
4. **`dvc_feature_engineering`** builds the lag/rolling/calendar/price/
   categorical feature matrix (`ml_features.csv`) and persists the category
   encoders (`encoders.json`).
5. In parallel: **`commit_dvc_lineage`** commits `dvc.lock` (the record of
   exactly what data produced this run) and pushes it to GitHub, while
   **`retrain_baseline`/`retrain_knn`/`retrain_arima`** retrain against
   `subset_long.csv` and **`retrain_ml_models`** retrains the 4 global
   models against `ml_features.csv` — each in its own one-shot container,
   each logging to DagsHub's hosted MLflow instance.
6. **`compare_models`** pulls every model's results into one table, sorted
   by aggregate MAPE.
7. **`promote_champion`** only overwrites the `champion` alias if the best
   *promotable* candidate actually beats the current champion's logged
   metric — and clears the alias from every other promotable model, so
   exactly one model ever holds it.
8. **`reload_serving`** restarts the serving container so it picks up the
   new champion immediately rather than waiting out its cache TTL.
9. From here on, every `/predict` call resolves the new champion, builds
   its features live, and feeds the drift buffers — Grafana reflects it
   within seconds.

## 5. Hard questions to be ready for

**"Why not just retrain and redeploy on every request, skip the batch
job?"** Training here takes minutes (ARIMA's 100 per-series SARIMA fits
alone can take ~40), and the whole point of a registry + alias is
decoupling "when a model gets trained" from "when it gets served" — serving
should be fast and stateless per request, not re-fitting anything.

**"What happens if two retrains overlap?"** They can't —
`max_active_runs=1` on the DAG.

**"What if the training container crashes halfway through a model
family?"** That task fails, Airflow marks it failed, and downstream tasks
depending on it don't run — `compare_models` would only see the models
that actually finished and logged, and a human has to look at the DAG run
in Airflow before `promote_champion` ever gets a chance to run off
incomplete data (it's not auto-retried blindly past a hard failure).

**"How would you scale this past 100 series / past one machine?"** The
honest answer: per-series models (KNN/ARIMA) don't scale well as written —
100 sequential MLflow registrations is already the long pole. The global
models scale much better (one model, any number of entities as a
categorical feature) — that's the actual production-shaped approach here,
and it's a deliberate point of comparison in the project, not an oversight.
Orchestration-wise, `DockerOperator` on a single host maps fairly directly
to `KubernetesPodOperator` on a cluster — the "one-shot container per task"
shape doesn't change, just where it schedules.

**"Why Prometheus/Grafana and not something like Evidently or
WhyLabs for drift?"** Prometheus/Grafana were already the metrics stack for
latency/request-rate; running the KS test inside the serving process and
exposing it as a couple of Prometheus gauges reuses that stack instead of
adding a second monitoring system. A dedicated drift-detection library
would give you more statistical tests and dataset-level reports out of the
box — a reasonable alternative, traded here for "one less moving part."

**"Single point of failure?"** DagsHub's hosted MLflow instance is one: if
it (or your network path to it) is down, no retrain can log/register
anything, and serving can't resolve a champion if its own TTL-cached one
expires mid-outage. Self-hosting wouldn't remove this risk, just relocate
it — a single self-hosted `mlflow server` is exactly as much of a SPOF as
a single hosted instance; you'd need redundant instances behind a load
balancer either way to actually remove it. That's an accepted trade-off
for a project this size, not something hidden.

## 6. Glossary (say these correctly, unprompted, and you'll sound fluent)

- **MAPE, three ways** — mean per-series (sensitive to outliers), median
  per-series (robust to them), aggregate (sum actuals/forecasts across
  series first, then one MAPE — most stable, used for promotion).
- **Global vs. per-series models** — per-series: each entity's own model,
  own history only. Global: one model, every entity stacked, entity ID as
  a feature. Global models share statistical strength across entities;
  per-series models can't leak information between entities even if you
  wanted them to.
- **Model registry alias vs. stage** — alias: a named pointer you define
  (`champion`), scoped to one registered model, reassignable. Stage: a
  fixed global enum (deprecated in MLflow).
- **Leakage-safe lag** — use the largest lag that stays valid across the
  *entire* forecast horizon, not just its first step.
- **Drift vs. accuracy monitoring** — drift is a leading proxy signal,
  checkable continuously without ground truth; accuracy is the lagging,
  authoritative signal, only checkable once ground truth (actual sales)
  arrives.
- **Docker-outside-of-docker** — a container asks the *host's* Docker
  daemon (via a mounted socket) to start a sibling container, rather than
  running a nested Docker daemon inside itself. Bind-mount paths in that
  sibling resolve against the host, not the caller.
- **DVC stage vs. plain script** — a DVC stage is cacheable and
  reproducible because it's framed as a pure function of its `deps`; a
  step whose output depends on accumulated state across runs (like this
  project's ingestion simulation) doesn't fit that model and should stay a
  plain step outside `dvc.yaml`.
- **DagsHub-hosted MLflow** — every DagsHub repo gets a free MLflow-
  compatible tracking server + model registry + UI at
  `https://dagshub.com/<owner>/<repo>.mlflow`. Same `mlflow` client API as
  a self-hosted server; only the URI and HTTP basic auth
  (`MLFLOW_TRACKING_USERNAME`/`PASSWORD`) differ.
- **Two auth flows, one DagsHub repo** — DVC's credentials
  (`.dvc/config.local`) and MLflow's (`MLFLOW_TRACKING_USERNAME`/
  `PASSWORD` in `.env`) authenticate independently against the same
  DagsHub repo. The same access token value can be reused for both, but
  setting up one does not set up the other.
