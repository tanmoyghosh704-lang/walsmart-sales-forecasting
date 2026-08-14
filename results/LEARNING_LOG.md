# Build Log — Walmart M5 MLOps Forecasting Project

> **Status (as of 2026-08-12): all 6 phases complete.** Data → naive
> baseline → Prophet + MLflow tracking/registry → DVC/DagsHub → FastAPI
> serving → Docker → Airflow retraining DAG → Prometheus/Grafana
> monitoring, each independently verified end-to-end. See
> [`writeup.md`](writeup.md) for the results summary and
> [`../README.md`](../README.md) for reproduction steps.
>
> To resume working on this locally: the `mlflow server` process doesn't
> persist between sessions. Restart it before touching `serving/app.py`
> or `src/train.py`:
> `mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
> --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"`
> — the registered models and run history are still there in `mlflow.db`
> (gitignored but still on disk locally), nothing needs retraining.
>
> Natural next steps if extending this further: multi-task Airflow DAG
> with validation + conditional model promotion, per-series drift
> reference distributions (see Phase 6's honest limitation note), and
> per-series Prophet hyperparameter tuning.

Running log of what we did, which tool it involved, why we did it, and any
difficulties hit along the way. Written to be re-read before an interview.

---

## 2026-08-11 — Project kickoff

### Repo scaffolding
- Created the directory structure from the workflow doc (`data/`, `src/`,
  `serving/`, `airflow/dags/`, `monitoring/`, `results/`).
- **Why this structure:** separates concerns the same way a real MLOps repo
  does — data pipeline (`src/`), serving layer (`serving/`), orchestration
  (`airflow/`), and observability (`monitoring/`) are independently
  deployable pieces, not one monolithic script. Interview talking point:
  this mirrors how a model moves from training code to a served, monitored
  system.

### Git
- Ran `git init` — turns the folder into a git repository (creates a
  hidden `.git/` folder that tracks history). Nothing is committed yet;
  `git init` alone doesn't save any files, it just starts tracking.
- Added `.gitignore` — tells git which files/folders to never track. We're
  excluding:
  - `data/raw/*` and `data/processed/*` (actual dataset files) — because
    **DVC**, not git, will version the data (git is bad at large binary
    files; DVC stores them separately and git only tracks small pointer
    files). This is the standard git+DVC split.
  - `.venv/` — the virtual environment is machine-specific and rebuildable
    from `requirements.txt`; never commit it.
  - `mlruns/`, `mlartifacts/` — MLflow's local tracking data; regenerable,
    and can get large.
  - `kaggle.json`, `.env` — credentials; must never be committed.

### Python virtual environment
- Ran `python -m venv .venv` — creates an isolated Python environment
  inside the project folder (`.venv/`). Why isolate: this project needs
  specific versions of many libraries (Prophet, MLflow, Airflow, FastAPI);
  installing them globally would pollute your system Python and could
  conflict with other projects.
- `requirements.txt` lists every dependency the project needs, pinned
  loosely for now. This file is what makes the project reproducible on
  another machine (`pip install -r requirements.txt`).
- **Note:** Airflow is deliberately *not* installed in the same pass as
  everything else. Airflow publishes "constraints files" per Python/Airflow
  version because its dependency tree conflicts easily with unrelated
  packages. We'll install it separately in Phase 5 using the official
  constraints URL to avoid breaking the rest of the environment.
- **Difficulty:** the first install attempt got silently killed when the
  background shell session was torn down mid-run (only a handful of DVC's
  transitive dependencies had landed). Lesson: for long-running background
  installs, verify with `pip list` after the fact rather than assuming
  "it ran, so it finished."

### Kaggle authentication
- The M5 dataset lives behind Kaggle's competition download, which
  requires (a) an authenticated API token tied to your account, and
  (b) having clicked "Join/Late Submission" on the competition page to
  accept its rules — a plain public download link doesn't work.
- Kaggle's newer auth scheme uses a single opaque token (`KGAT_...`)
  instead of the older `kaggle.json` (`{"username":..., "key":...}`)
  format. The client looks for it at `~/.kaggle/access_token` (or the
  `KAGGLE_API_TOKEN` env var).
- We saved the token to `~/.kaggle/access_token` and added `access_token`
  to `.gitignore` (on top of the pre-existing `kaggle.json` entry) so it
  can never accidentally get committed.
- **Difficulty / hygiene note:** the token was captured in a screenshot
  saved to the Downloads folder — a plaintext credential sitting in an
  image file is a real leak vector (synced to cloud photo backup, shared
  by accident, etc). Recommended deleting that file after setup. General
  interview-relevant point: secrets belong in credential stores or
  environment variables, never in files that get casually shared.
- **Difficulty:** that `KGAT_...` token turned out to be Kaggle's *new*
  auth scheme, but the current PyPI `kaggle` package (v1.7.4.5 — latest
  available) only understands the classic `kaggle.json`
  (`{"username":..., "key":...}`) format via HTTP Basic Auth. Confirmed by
  reading the installed package's `authenticate()` source directly rather
  than guessing. Had to go back to Kaggle settings and use the separate
  **"Create New Token"** button, which downloads the old-style
  `kaggle.json`. Moved it to `~/.kaggle/kaggle.json` and `chmod 600` it
  (restrict to owner-read/write — same "don't leave secrets
  world-readable" principle as above).
- Also: invoking the CLI as `python -m kaggle ...` fails (`kaggle` is a
  plain package with no `__main__.py`); the correct entry point is the
  installed console script `kaggle` (or `.venv/Scripts/kaggle.exe` on
  Windows when not activating the venv first).

### Downloading the M5 dataset
- `kaggle competitions download -c m5-forecasting-accuracy -p data/raw`
  pulled a 45.8 MB zip in under a second (Kaggle serves these from a CDN).
  Unzipped into `data/raw/`, then deleted the zip to avoid keeping the
  data twice on disk.
- **What's actually in the M5 files** (worth knowing cold for an
  interview):
  - `sales_train_validation.csv` — daily unit sales per store-item series
    (`d_1` … `d_1913`), the core training data. Each row is one
    (item_id, store_id) series; each column after the id fields is one
    day.
  - `sales_train_evaluation.csv` — same shape but extended 28 more days
    (`d_1` … `d_1941`); used for the competition's final scoring. We'll
    mainly use `sales_train_validation.csv` for iterative dev.
  - `calendar.csv` — maps `d_1`, `d_2`, ... day-index columns to real
    dates, weekday, month, year, and event/holiday flags (`event_name_1`,
    `event_type_1`, SNAP indicators). This is the join table that gives
    us actual holiday features for Prophet.
  - `sell_prices.csv` — weekly price per (store, item), needed if we ever
    add price as a regressor (not required for the baseline Prophet
    model).
  - `sample_submission.csv` — Kaggle's expected submission format; not
    needed since we're not submitting to the leaderboard, just using the
    data.
  - Files total ~430 MB uncompressed — confirms why "scope the dataset
    down" (per the workflow doc) matters before training anything.

---

## Phase 0 — Data subset + naive baseline

### Scoping the dataset (`src/data_prep.py`)
- Loaded the full wide-format `sales_train_validation.csv` (30,490 series
  x 1,913 days), summed each series' total historical units, and kept
  the **top 100 series by total volume**.
- Why top-N by volume rather than a random sample or "one category":
  high-volume series have proportionally fewer zero-sales days, which
  matters a lot for MAPE (division by zero / near-zero actuals blows the
  metric up — see below). A random sample would drag in slow-moving,
  mostly-zero series and make the headline MAPE meaningless.
- Interesting side effect: the top 100 by volume turned out to be almost
  entirely `FOODS` (97 series) with a handful of `HOUSEHOLD` (3 series) —
  no `HOBBIES`. Makes sense once you think about it: grocery items are
  bought far more frequently/in higher unit counts than household goods,
  so volume-based ranking naturally skews the category mix. Worth
  mentioning in the writeup as an explicit, deliberate bias in the
  subset (not representative of the full catalog).
- Reshaped from wide (`d_1`...`d_1913` columns) to long format
  (`id`, `date`, `sales`, ...) via `pandas.melt` — long format is what
  Prophet requires (`ds`, `y` columns) and what almost every plotting /
  groupby operation downstream wants. Joined in `calendar.csv` to attach
  real dates and holiday/SNAP event flags to each row.
- Saved to `data/processed/subset_long.csv` (191,300 rows = 100 series x
  1,913 days).

### Naive baseline (`src/baseline.py`)
- Held out the **last 28 days** as a test window (28 days = 4 weeks,
  matches M5's own competition horizon, so it's a defensible, standard
  choice rather than an arbitrary one).
- Baseline method: **seasonal naive, lag-7** — take each series' last 7
  training days and tile that pattern across the 28-day horizon. This is
  "assume next month looks like last week, repeated" — deliberately dumb,
  which is the point: it's the free lunch any model must beat to justify
  existing.
- **Result:**
  - Mean per-series MAPE: **78.58%**
  - Aggregate (summed-across-series) MAPE: **10.75%**
  - **Why those two numbers are so different — important interview
    point:** MAPE is a ratio, so it's extremely sensitive to small
    actuals. A series with actual=1 and forecast=3 contributes 200% to
    the per-series average even though the absolute error is trivial.
    Summing sales across all 100 series before computing MAPE smooths
    out that per-series noise (the day's *total* demand is never near
    zero), so the aggregate number is far more stable and representative
    of "how good is this forecast in practice." We're reporting both,
    but the per-series number is the more honest one if the real use
    case is "forecast this specific item at this specific store" — the
    aggregate number would look great even if individual series were
    forecast poorly.
  - This also foreshadows a known weakness of plain MAPE for intermittent
    retail demand — WMAPE (weighted by actual volume) or sMAPE are common
    fixes in industry, worth naming even though we're sticking with MAPE
    per the project's stated headline metric.
- Output: `results/baseline_mape_per_series.csv` (one row per series) and
  `results/baseline_summary.csv` (the two headline numbers above) — these
  are the numbers every later Prophet MAPE claim gets compared against.
- **Difficulty:** `DataFrameGroupBy.apply` threw a `FutureWarning` about
  including grouping columns in the applied function in future pandas
  versions — fixed by passing `include_groups=False`. Minor, but a good
  habit: don't ignore deprecation warnings, they're pandas telling you
  the API is about to change under you.

---

## Phase 1 — EDA (`notebooks/eda.ipynb`)

Ran as a Jupyter notebook (per your preference, so you can re-open and
poke at it interactively) rather than a script. Executed non-interactively
once via `jupyter nbconvert --to notebook --execute --inplace` to confirm
it runs clean top-to-bottom and to pull numbers into this log — you can
still open it in Jupyter/VS Code and re-run cells by hand.

- **What `nbconvert --execute` does:** launches a real Jupyter kernel,
  runs every cell in order, and writes the outputs (including saved
  matplotlib figures) back into the `.ipynb` file itself — that's why
  opening the notebook file now shows results without you having to run
  it first. Useful for CI ("does this notebook still run") as well as
  for generating a reviewable artifact.

### Weekly seasonality
- Weekend sales are clearly higher: Saturday ~40.6, Sunday ~39.5 units/day
  vs. midweek Tue/Wed ~28. Friday sits in between (~33). Confirms weekly
  seasonality is real and worth Prophet's default weekly Fourier term.
- Plot saved to `results/eda_weekly_seasonality.png`.

### Calendar events (`event_name_1`/`event_type_1`)
- **Surprising result:** mean sales on event days (32.30) vs. non-event
  days (32.66) are essentially the same — if anything slightly *lower* on
  event days. Breaking down by event type, Sporting (35.3) and Cultural
  (34.5) events show a mild bump, National (30.4) events show none.
- **Why this matters / interview point:** the workflow doc assumed
  holiday effects would be a clear win to hand Prophet as regressors —
  the data says otherwise for *this* subset (FOODS/HOUSEHOLD-heavy, not
  e.g. toys or seasonal goods where holiday spikes would be obvious).
  Good example of "check the assumption, don't just implement it" — we'll
  still pass `event_name_1` to Prophet's holidays parameter in Phase 1
  modeling since it's cheap and Prophet handles irrelevant regressors
  gracefully, but we won't oversell it in the writeup.

### SNAP effect
- Clear, consistent, and non-trivial: every state shows higher mean sales
  on its own SNAP days vs. non-SNAP days (CA: 30.6→33.7, TX: 35.9→38.2,
  WI: 27.0→34.7 — WI's ~28% jump is the largest). Makes sense given the
  subset is dominated by FOODS items, which is exactly what SNAP benefits
  cover.
- Implementation note: SNAP is state-specific (`snap_CA`/`snap_TX`/
  `snap_WI` are three separate columns), so it had to be looked up per
  row using that row's own `state_id` — a flat "is today a SNAP day"
  column would be wrong for a multi-state subset.
- This is a stronger, more defensible feature to hand Prophet than the
  calendar events above.

### Individual series shape
- Sampled 3 individual series (28-day rolling mean) to sanity-check for
  intermittency/outliers before committing to per-series Prophet models —
  saved to `results/eda_sample_series.png`. Reasoning: aggregate plots
  (like the trend plot) hide per-series pathologies that would break a
  per-series model but wash out in a 100-series sum.

---

## Phase 1 — Prophet modeling (`src/train.py`)

### What Prophet actually does
- Prophet fits an **additive model**: `y(t) = trend(t) + seasonality(t) +
  holidays(t) + error`. Trend is piecewise-linear (or logistic) with
  automatically-detected changepoints; seasonality terms (weekly, yearly)
  are Fourier series; holidays are one-off date-indexed offsets. It
  optimizes this via `cmdstanpy`, a Python interface to Stan (a
  probabilistic-programming/optimization engine written in C++) — that's
  why the first Prophet install compiled a separate binary.
- Trained **one Prophet model per series** (100 total, not one shared
  model) — same as how the baseline was computed per series, and
  standard practice for M5-style hierarchical forecasting where each
  store-item combination has its own trend/seasonality shape.

### Features fed to Prophet
- `holidays`: built from calendar.csv's `event_name_1`/date pairs (154
  distinct events across the training window), passed via Prophet's
  native `holidays` parameter — this is its designed mechanism for "this
  specific date behaves differently."
- `snap` as an **extra regressor** (`model.add_regressor("snap")`) — the
  one EDA flagged as having a real effect. Important mechanical detail:
  `add_regressor` must be called *before* `model.fit()`, and any column
  registered this way must also be supplied for the future/test period at
  prediction time (Prophet doesn't forecast the regressor itself, you
  give it the known-in-advance value). SNAP eligibility is a published
  government schedule, so this is legitimate — we're not leaking future
  sales data, just a known future calendar fact, same category as
  "is this a Sunday."
- Negative predictions clipped to 0 post-hoc (`yhat` can go negative
  since Prophet's additive model has no non-negativity constraint, but
  unit sales can't be negative).

### Result — Prophet vs. naive baseline

| model | mean per-series MAPE | aggregate MAPE |
|---|---|---|
| naive_seasonal_lag7 (baseline) | 78.58% | 10.75% |
| prophet | 68.52% | 6.60% |

- **Per-series: 12.8% relative improvement. Aggregate: 38.6% relative
  improvement.** Prophet clears the baseline on both metrics, but the gap
  is much bigger at the aggregate level — makes sense given Prophet
  explicitly models trend/seasonality (smooths the *shape* of the curve),
  which shows up clearly once you sum away the per-series noise. The
  per-series number stays noisier because MAPE's small-denominator problem
  (discussed in the baseline section) doesn't go away just because the
  model got better — a single bad day on a low-volume series still
  dominates that series' MAPE regardless of which model produced the
  forecast.
- These are the numbers behind the project's headline CV claim ("improved
  MAPE by XX% over naive baseline") — worth having both the per-series and
  aggregate framing ready in an interview, since which one you lead with
  changes the story.
- Outputs: `results/prophet_mape_per_series.csv`, `results/prophet_forecasts.csv`,
  `results/prophet_summary.csv`, and `results/model_comparison.csv` (the
  side-by-side table above).
- **Runtime note:** 100 sequential Prophet fits took a few minutes
  (each fit + Stan optimization is roughly 1-2s). At this subset size
  that's fine for iteration; at full M5 scale (30k+ series) this would
  need parallelization (e.g. joblib/multiprocessing per series) — worth
  naming as a known scaling limitation if asked.

---

## Phase 2 — DVC/DagsHub versioning + MLflow tracking/registry

### DVC + DagsHub
- `dvc init` turns the repo into a DVC project (adds `.dvc/config`,
  `.dvc/.gitignore`, `.dvcignore`) — analogous to `git init`, but for
  large-data version control. DVC's core trick: it stores the *actual*
  large files in a content-addressed cache (keyed by MD5 hash) and
  commits only a small `.dvc` pointer file (a few lines of YAML/JSON
  with the hash + size + file count) to git. Git stays fast and small;
  DVC's remote (here, DagsHub) holds the real bytes.
- `dvc remote add origin <dagshub-url>.dvc` registers DagsHub as the
  storage backend. Credentials (username + token) were set with
  `dvc remote modify origin --local ...` — the `--local` flag writes to
  `.dvc/config.local` instead of the committed `.dvc/config`, and DVC's
  own `.dvc/.gitignore` already excludes `config.local` by default. Same
  "secrets never touch a committed file" pattern as the Kaggle/API-token
  setup earlier — verified this explicitly (`cat .dvc/.gitignore`) rather
  than assuming.
- `dvc add data/raw data/processed` — this is the actual "version this
  data" step. It hashes the directory contents, moves them into DVC's
  local cache, and writes `data/raw.dvc` / `data/processed.dvc` pointer
  files plus a `data/.gitignore` that tells git to ignore the real
  `raw/`/`processed/` folder contents (git only ever sees the `.dvc`
  pointer files).
  - **Difficulty:** this failed the first time with `output 'data\raw'
    is already tracked by SCM` — because `data/raw/.gitkeep` and
    `data/processed/.gitkeep` (placeholders from the very first repo
    scaffolding step) were already committed to git. DVC refuses to
    take over a directory git already has a stake in. Fixed by
    `git rm --cached` on the two `.gitkeep` files (they're now
    redundant anyway — DVC's pointer file keeps the directory
    "present" in the repo's history without needing a placeholder).
- `dvc push -r origin` uploaded the actual data (6 files: 5 raw CSVs +
  1 processed CSV, ~450MB raw) to DagsHub's storage. This is the step
  that makes data "pushed to DagsHub" true, not just configured.
- **Bug found and fixed while doing this:** the original `.gitignore`
  (written in the very first setup step, before any data existed) had a
  blanket `*.csv` rule intended to stop the *raw M5 data* from being
  git-tracked. It was too broad — it was silently also excluding
  `results/*.csv` (the baseline and Prophet MAPE tables from Phases 0-1),
  so those outputs were sitting in the working directory but never
  actually entering git history. Caught by reading `git status` closely
  instead of assuming a big commit succeeded fully. Fixed by removing the
  blanket rule now that DVC's own auto-generated `data/.gitignore`
  handles excluding the real data directories precisely.
- **Also noticed:** a GitHub remote (`origin`) and an earlier commit
  ("data processing done") already existed in this repo, made directly
  through VS Code's git UI outside of this working session. Verified via
  `git show --stat` and `git log` before touching anything, per the
  general rule of investigating unfamiliar repo state before acting on
  it — turned out to be the user's own prior commit, not a conflict.

### MLflow tracking + Model Registry
- **Key gotcha:** MLflow's default tracking store is a flat local
  `mlruns/` folder (a "file store"). The file store can log runs,
  params, and metrics fine, but it **cannot support the Model Registry**
  — registering a model silently requires a database-backed store
  (SQLite, Postgres, MySQL, etc). Switched the tracking URI to
  `sqlite:///mlflow.db` specifically to unlock the registry, since the
  project brief calls out "use the Model Registry specifically, not just
  experiment tracking."
- **Run structure:** one parent run per training invocation
  (`prophet_training_<timestamp>`) logging the shared config (n_series,
  test horizon, date ranges, holiday source, regressors used) plus the
  two headline metrics (mean per-series MAPE, aggregate MAPE) — and one
  **nested run per series** (`mlflow.start_run(..., nested=True)`)
  logging that series' own params (item/dept/cat/store/state id) and its
  individual MAPE. This mirrors a real per-entity forecasting setup:
  parent run = "this training job," child runs = "this one model."
- **Registry pattern:** registered each series' fitted Prophet model
  under its *own* registered-model name (`prophet_<series_id>`), via
  `mlflow.prophet.log_model(model, name="model",
  registered_model_name=f"prophet_{series_id}")`. This is what makes the
  Phase 3 FastAPI design work cleanly: given a store-item id, load
  `models:/prophet_<series_id>/latest` directly — no loose pickle files,
  no custom lookup table. `mlflow.prophet` is a built-in MLflow "flavor"
  (like `mlflow.sklearn`, `mlflow.pytorch`) that knows how to
  serialize/deserialize a Prophet model specifically, so
  `mlflow.prophet.load_model(...)` hands back a working Prophet object
  ready to `.predict()`.
- **Verification before the full run:** rather than firing off the full
  100-series retrain with untested logging code, first did a throwaway
  smoke test — fit one tiny Prophet model, log it, register it, then
  immediately load it back via `models:/.../<version>` and call
  `.predict()` — to confirm the register→load→predict round trip
  actually works end to end before spending the ~several minutes on the
  real run. Caught a deprecation warning this way too (`artifact_path`
  param renamed to `name` in this MLflow version) and fixed it before
  the real run instead of after.
- **Result:** all 100 series registered as 100 distinct MLflow registered
  models (confirmed via `MlflowClient().search_registered_models()`),
  each with 1 version so far. MAPE numbers matched the earlier
  non-MLflow run exactly (68.52% / 6.60%) — expected, since Prophet's
  fit is deterministic and MLflow only wraps logging around the same
  computation, it doesn't change it.
- `mlflow.db` and `mlruns/` (where artifact files like the serialized
  model actually live, even with a SQLite *metadata* backend) added to
  `.gitignore` — regenerable from `src/train.py`, shouldn't bloat git.

---

## Phase 3 — FastAPI serving (`serving/app.py`)

### Design
- `GET /health` — trivial liveness probe returning tracking URI, how many
  series are servable, and the valid forecast date window. Cheap to
  build, but a real signal a reviewer/interviewer will specifically look
  for (load balancers and orchestrators poll this, not `/predict`).
- `GET /predict?series_id=...&horizon=...` — loads the model for that
  series via `mlflow.pyfunc.load_model(f"models:/prophet_{series_id}/latest")`
  (the registry URI scheme, not a file path), builds the future dataframe
  the model needs (dates + the `snap` regressor, looked up from
  `calendar.csv`), predicts, clips negative `yhat` to 0, and returns
  JSON.
- **`mlflow.pyfunc.load_model` vs `mlflow.prophet.load_model`:** used
  the generic `pyfunc` loader deliberately (per the project brief's
  explicit instruction) rather than the Prophet-specific one. `pyfunc` is
  MLflow's flavor-agnostic interface — every registered model, regardless
  of what ML library trained it, exposes the same `.predict(df)` method.
  This is what lets a serving layer stay decoupled from "what
  library was this model trained with," which matters if the modeling
  approach ever changes later without needing to touch serving code.
- **Model caching:** wrapped the loader in `functools.lru_cache` so the
  registry is only hit once per series per process lifetime, not on
  every request. Verified empirically — first call to a fresh series
  took ~2x longer than the cached repeat call.
- **`/latest` alias:** confirmed `models:/<name>/latest` resolves without
  needing to know/hardcode a version number (tested this against the
  registry before wiring it into the endpoint, same "verify before
  building on it" habit as the MLflow smoke test in Phase 2).

### The forecast-window constraint (and why it's not a bug)
- Models were trained on data through 2016-03-27 and evaluated through
  2016-04-24. The `snap` regressor they need is only genuinely *known*
  (not fabricated) through `calendar.csv`'s actual coverage, which
  extends to 2016-06-19 — 56 more days past evaluation.
- `/predict` computes this window at startup (`_forecast_start`,
  `_forecast_end`) and rejects out-of-range horizons with a clear 400,
  rather than silently guessing a future SNAP schedule. Tested this path
  explicitly (`horizon=9999` → 400 with the exact valid range in the
  error message; `horizon=0` → same). Also tested an unknown `series_id`
  → clean 404 instead of a raw MLflow stack trace leaking to the client.
- **Interview point:** this is a real constraint any regressor-based
  forecasting API has — you can only forecast as far as your "known
  future" exogenous inputs actually extend. A naive implementation would
  either crash past that point or quietly extrapolate garbage; this one
  fails loudly and explains why.

### Verification
- Ran the app locally with `uvicorn serving.app:app` and hit it with
  `curl` rather than trusting it compiles — confirmed `/health`,
  `/predict` (valid input), `/docs` (FastAPI's free auto-generated
  Swagger UI, another built-in production-awareness signal), and all
  three error paths above, before calling this phase done.
- **What `uvicorn` is:** the ASGI server that actually runs a FastAPI
  app — FastAPI defines routes/handlers, uvicorn is the process that
  listens on a socket, parses HTTP, and calls into that app. FastAPI
  by itself isn't runnable; it needs uvicorn (or another ASGI server)
  underneath it, same relationship as Flask+gunicorn or Django+wsgi.

---

## Phase 4 — Docker (`serving/Dockerfile`, `.dockerignore`)

### Basic setup
- `serving/requirements.txt` is a **separate, slimmer** dependency list
  from the root `requirements.txt` (fastapi/uvicorn/mlflow/prophet/
  pandas/numpy only) — the root file includes Airflow and Jupyter, which
  have no business in a serving image (bigger image, slower build, and
  Airflow's dependency tree is fussy enough that it's not worth risking
  a conflict here for packages the API never imports).
- `.dockerignore` matters even though the Dockerfile only `COPY`s a
  handful of specific paths: Docker sends the **entire build context**
  (everything not excluded) to the daemon before any `COPY` line runs,
  regardless of what actually gets used. Without excluding `.venv/`
  (huge), `.git/`, and the 3 large raw M5 CSVs the app doesn't need,
  every `docker build` would tar up and transfer ~500MB+ for nothing.
- Built with `docker build -f serving/Dockerfile -t m5-forecast-api .`
  — note the context is the **repo root** (`.`), not `serving/`, because
  the Dockerfile needs to reach `data/raw/calendar.csv` etc. from there.

### Problem 1: baked-in MLflow artifacts don't survive the trip into a container
- First attempt `COPY`d `mlflow.db` and `mlruns/` straight into the
  image, reasoning "then the container is fully standalone, no external
  dependency." Built fine, `/health` worked, but `/predict` failed with
  `No such artifact: ''`.
- **Root cause, found by inspecting the sqlite DB directly** (not
  guessing): MLflow's local file-based artifact store records each
  experiment's `artifact_location` as an **absolute host filesystem
  path** (`file:///C:/python3_10_11/walsmart sales forecasting/mlruns/1`)
  at the moment the experiment is first created. That path is baked into
  the database and is immutable per-experiment. Copying `mlruns/` into
  `/app/mlruns/` inside a Linux container doesn't help — the DB still
  says to look for artifacts at the old Windows path, which doesn't
  exist there.
- **Real fix, not a workaround:** stood up an actual `mlflow server`
  process (`mlflow server --backend-store-uri sqlite:///mlflow.db ...`)
  instead of pointing clients directly at `sqlite:///mlflow.db`. With a
  tracking server, clients (both `src/train.py` on the host and
  `serving/app.py` in a container) talk to a **network address**
  (`MLFLOW_TRACKING_URI=http://...`), and the server — which always runs
  on the same machine as its own artifact storage — resolves artifact
  paths locally on its own filesystem. The client never needs to know or
  care about a filesystem path at all. This is the actual, intended way
  to run MLflow across more than one machine/environment; local
  `sqlite:///` + bare `mlruns/` is documented as single-machine-only for
  exactly this reason.
  - Wiped the old `mlflow.db`/`mlruns/` (safe — both gitignored, fully
    regenerable) and reran `src/train.py` against the server to get a
    fresh, portable experiment.
  - Reworked `serving/Dockerfile` to stop copying MLflow state into the
    image entirely; `serving/app.py` now reads `MLFLOW_TRACKING_URI`
    from an environment variable (default `http://127.0.0.1:5000` for
    local dev), set at `docker run` time to
    `http://host.docker.internal:5000` — Docker Desktop's built-in DNS
    alias for reaching the host machine from inside a container.

### Problem 2: MLflow's server crashed mid-training on a Windows console encoding error
- Retraining against the new server died after registering exactly 1 of
  100 models, with `UnicodeEncodeError: 'charmap' codec can't encode
  character '\U0001f3c3'` (the 🏃 emoji MLflow prints when a run ends).
- **Root cause:** Windows' default console/stdout encoding is `cp1252`
  (not UTF-8), which has no representation for that emoji. MLflow's own
  `_log_url()` doesn't guard the `sys.stdout.write()` call, so this
  crashes the whole client process, not just a warning.
  - Confirmed this by counting how many `"Successfully registered
    model"` lines existed in the log before the crash (exactly 1) rather
    than assuming the whole run had failed or partially worked.
- **Fix:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
  at the top of `src/train.py`, gated on `sys.platform == "win32"`.
  Forces UTF-8 stdout regardless of the terminal's codepage, which is
  the standard fix for this entire class of Windows console Unicode
  issue in Python (also shows up with print statements involving
  checkmarks, arrows, etc. from other libraries).

### Problem 3: MLflow's DNS-rebinding protection blocked the container
- With the server fix in place, `/predict` from inside the container
  still failed: `403 ... 'Invalid Host header - possible DNS rebinding
  attack detected'`.
- **What this is:** MLflow 3.x ships built-in protection against DNS
  rebinding attacks (a browser-based attack where a malicious page
  tricks your browser into hitting `localhost` services). By default it
  only accepts requests whose `Host` header matches `localhost`/private
  IPs. The container's requests arrive with `Host:
  host.docker.internal:5000`, which isn't on that default allowlist.
- **Fix:** started the server with
  `--allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"`
  — note the **port had to be included** in each entry; the bare
  hostname without a port was silently not enough (first attempt with
  `--allowed-hosts host.docker.internal` alone still 403'd; adding the
  `:5000`-qualified variant fixed it). Also needed `--host 0.0.0.0` on
  the server itself so it accepts connections from outside `localhost`
  in the first place, not just `127.0.0.1`.
  - This is a legitimate security feature, not something to disable
    wholesale (`--dev` mode removes all of it) — worth naming as "found
    a real security control, configured it narrowly for the one
    legitimate caller, didn't turn it off" if this comes up in an
    interview.

### Verification
- Ran the **rebuilt** image standalone (`docker run -p 8001:8000 -e
  MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ...`) and
  re-tested everything: `/health`, `/predict` with a real series (numbers
  matched the non-container run exactly), the unknown-series 404 path,
  and `/docs`. Didn't call this phase done on "the image built" alone —
  building was the easy 80%, the artifact-resolution and host-header
  issues were the part that actually needed debugging.

---

## Phase 5 — Airflow (`docker-compose.yaml`, `airflow/`)

Airflow doesn't officially support running natively on Windows (only
Linux, or Windows via WSL2/Docker), so this ran via Docker Compose from
the start rather than a native `pip install apache-airflow`.

### LocalExecutor instead of the official CeleryExecutor setup
- Apache's official `docker-compose.yaml` defaults to `CeleryExecutor`:
  webserver + scheduler + a separate worker + triggerer + Redis +
  Postgres (6+ containers). That's real infrastructure for running many
  concurrent tasks across distributed workers — overkill for "get a
  single-task DAG working," which is explicitly how the project doc says
  to start.
- Built a slimmer stack instead: Postgres (Airflow's metadata DB, not to
  be confused with the MLflow tracking server) + webserver + scheduler,
  using `LocalExecutor` (tasks run as subprocesses of the scheduler
  process itself, no separate worker needed). Fewer moving parts, same
  DAG-authoring experience -- worth being able to name explicitly *why*
  this is a legitimate simplification and not just "skipped the real
  thing," if asked.

### Problem 1: `.dockerignore` silently broke the Airflow image build
- First `docker compose build` failed: `COPY airflow/requirements.txt
  /requirements.txt` — file not found in build context.
- Root cause: Phase 4's `.dockerignore` excluded the entire `airflow/`
  folder (reasonable at the time — the FastAPI image had no use for it).
  Since Docker Compose builds from the same repo-root context, that
  exclusion silently applied here too. Fixed by narrowing the ignore
  rule to just `airflow/logs/` (runtime-generated, shouldn't be sent as
  build context) instead of the whole directory.

### Problem 2: Airflow's own dependency constraints conflict with mlflow/prophet
- Following the officially recommended pattern
  (`pip install -r requirements.txt --constraint <airflow-constraints-url>`)
  to add prophet/mlflow/pandas into the Airflow image failed twice in a
  row with `ResolutionImpossible` — first over `pandas` (constraint
  pinned `2.1.4`, our requirements.txt hard-pinned `2.3.3`), then again
  over `cryptography` after loosening the pandas pin.
- Rather than chase an ever-moving conflict pinning one package at a
  time, switched approach: installed prophet/mlflow/pandas/numpy into an
  **isolated venv** (`/opt/airflow/task_venv`) inside the image, entirely
  separate from Airflow's own Python environment. Nothing installed
  there can ever break Airflow's packages, because they don't share an
  environment. The DAG's `BashOperator` invokes
  `/opt/airflow/task_venv/bin/python src/train.py` directly instead of
  the system `python`. Clean build on the first try afterward.

### Problem 3: `catchup=False` still ran one extra, concurrent DAG run
- On unpausing the DAG, Airflow started **two** runs at once: the manual
  trigger, and an automatic `scheduled__2026-08-02...` run. `catchup=False`
  stops Airflow from backfilling *every* missed interval since
  `start_date`, but it still runs the single most recent past interval
  on activation — a real, slightly surprising Airflow default worth
  knowing before it causes confusing double-execution during testing.
  Not a bug in this DAG; just something to expect.

### Problem 4: the real bug — hardcoded `MLFLOW_TRACKING_URI`, and how it was actually found
- Both concurrent runs failed after ~4 minutes. This took a long,
  methodical debugging pass to pin down, and the process is worth
  recording as much as the fix:
  1. **First (wrong) hypothesis: it's hung.** `docker stats` showed ~3%
     CPU on the scheduler container — far below what 100 concurrent
     Prophet/Stan fits should consume — which looked like a stall.
  2. Tested every piece of `train.py` in isolation directly inside the
     container via `docker compose exec`: `mlflow.set_experiment`
     (0.3s), a bare `Prophet().fit()` (0.1s), reading the real
     `subset_long.csv` over the Docker bind mount (0.5s), the full
     `fit_and_forecast` with holidays + the snap regressor (0.4s), even
     `mlflow.prophet.log_model(..., registered_model_name=...)` with the
     nested parent/child run structure `main()` uses. **Every single
     piece worked, fast, no hang.** This was genuinely confusing — if
     every component works, why does the whole script not?
  3. **Second (also wrong) hypothesis: stdout buffering.** Python fully
     buffers stdout when it's not a TTY (as under a subprocess pipe),
     while the `logging`-module lines from cmdstanpy/mlflow flush
     immediately — which explained why we saw "Importing plotly failed"
     but none of `train.py`'s own `print()` progress lines. Reasonable
     theory, but didn't explain everything on its own.
  4. **The actual breakthrough:** stopped trusting `stdout` entirely and
     queried the MLflow tracking server directly —
     `mlflow.search_runs(experiment_names=[...])` — to check whether any
     of the 100 expected nested child runs actually existed. Zero did,
     after 15+ minutes. That's a real network-visible fact, immune to
     any local buffering theory, and it proved the process wasn't just
     "slow to log" — no work was happening at all.
  5. Went back to the **full** task log (not just the tail) once the run
     had actually failed (Airflow's `list-runs` had been showing `running`
     because I kept checking mid-retry-backoff, not because it was stuck
     forever) — and the real error was sitting there the whole time:
     `HTTPConnectionPool(host='127.0.0.1', port=5000): ... Connection
     refused`.
  6. **Root cause:** `src/train.py`'s `MLFLOW_TRACKING_URI` was a
     **hardcoded string constant** (`"http://127.0.0.1:5000"`), left over
     from Phase 4's fix. Only `serving/app.py` had been updated to read
     it from an environment variable — I never applied the same change
     to `train.py`. So no matter what `MLFLOW_TRACKING_URI` the
     `docker-compose.yaml` environment block set, the script ignored it
     and always tried `127.0.0.1` — which is the *container itself*
     inside Docker, not the host machine. Every one of my manual
     debugging tests had "worked" only because each one explicitly
     called `mlflow.set_tracking_uri('http://host.docker.internal:5000')`
     by hand — I had never actually run the unmodified script itself
     inside the container until the DAG did it for me.
  7. Fix: made `MLFLOW_TRACKING_URI` in `train.py` read from
     `os.environ.get(...)`, matching the pattern already used in
     `serving/app.py`, with `docker-compose.yaml` setting it to
     `http://host.docker.internal:5000` for the Airflow containers.
     Since `src/` is bind-mounted (not baked into the image), the fix
     applied immediately without a rebuild.
- **Lesson worth remembering:** isolated tests that each explicitly
  hardcode the "correct" value can all pass while the actual
  unmodified code path still fails — because the isolated tests never
  exercised the exact configuration surface (an environment variable
  silently not being read) that the real script depends on. The fix was
  found by checking observable server-side state (real network fact)
  over trusting either "it looks hung" or "my simplified reproduction
  passed," and by reading the complete log rather than a tail snippet.

### Verification
- After the fix, triggered a fresh run: **succeeded in 1m38s** for all
  100 series (real nested MLflow runs appearing at roughly 1/second,
  confirmed via `mlflow.search_runs` mid-run rather than assumed).
  Final MAPE: 68.58% / 6.57% (matches earlier runs within Prophet's
  normal fit-to-fit variance) — `results/model_comparison.csv` was
  correctly written back to the host through the bind mount, proving the
  full loop (container → training → MLflow registry → results file back
  on host) works end to end, not just that the container started.

---

## Phase 6 — Prometheus + Grafana (`monitoring/`)

Separate `monitoring/docker-compose.yaml` (not merged into the Airflow
one) so this phase stays independently runnable/verifiable, same
principle as every earlier phase.

### Request latency + count (`serving/app.py`)
- `prometheus-fastapi-instrumentator` -- one line
  (`Instrumentator().instrument(app).expose(app)`) adds a `/metrics`
  endpoint with request-count and latency-histogram metrics for every
  route, labeled by handler/method/status, with zero custom
  instrumentation code. Exactly the "low effort, real signal" the
  project brief describes -- confirmed by inspecting the raw
  `/metrics` output for `http_requests_total` and
  `http_request_duration_seconds_bucket` (the labeled version, as
  opposed to `..._highr_seconds` which has more buckets but no labels --
  used the labeled one in Grafana so panels can break down by endpoint).

### Drift metric -- what "drift" means for a forecast-serving API
- The project brief's drift example (KS-test vs. training distribution)
  implicitly assumes a live feature-scoring API receiving fresh feature
  vectors to compare. `/predict` doesn't work that way -- it takes a
  `series_id` + `horizon`, not a feature vector. Had to translate the
  concept rather than copy it literally: instead compare the
  distribution of **recently-served predictions (yhat)** against the
  **full historical training sales distribution** those models were fit
  on. If the model starts forecasting values that look statistically
  different from what it learned, that's the same underlying signal
  (something's off, worth a look) even though the mechanics differ from
  a textbook feature-drift check.
- Implementation: `collections.deque(maxlen=200)` as a rolling window of
  recent `yhat` values, `scipy.stats.ks_2samp` against the training
  distribution (`_subset["sales"]`, all 100 series pooled), exposed as
  two Prometheus `Gauge`s (`prediction_drift_ks_statistic`,
  `prediction_drift_ks_pvalue`) recomputed on each `/predict` call once
  the window has >=30 samples.
- **Observed and worth naming honestly:** querying a single low-volume
  series repeatedly produced a very high KS statistic (~0.84, p~1e-31)
  against the pooled 100-series training distribution -- not because
  anything is wrong, but because one item's typical sales range is
  naturally narrower than the full catalog's. Mixing traffic across
  several series brought it down (~0.34) but still statistically
  "different" by a formal KS test, since even a mixed sample of *recent*
  short-horizon forecasts is a different shape than five years of full
  history. This is a real limitation of the pooled-reference-distribution
  design worth stating plainly in the writeup: the metric is genuinely
  sensitive (it reacts to real distributional facts), but a
  production version would want a per-series reference distribution,
  or a baseline captured over a comparable serving window, not "all
  historical data across every series" -- otherwise the alarm is
  always at least mildly triggered by design, which teaches operators to
  ignore it. Naming a metric's limitation instead of just shipping it is
  itself worth surfacing in an interview.

### Grafana dashboard
- Auto-provisioned via two YAML files Grafana reads on startup:
  `provisioning/datasources/datasource.yml` (points it at the Prometheus
  container by service name, `http://prometheus:9090` -- Docker Compose's
  built-in DNS resolves service names automatically within the same
  compose network, no `host.docker.internal` needed here since both
  containers are on the same Docker network) and
  `provisioning/dashboards/dashboard.yml` (tells it to load dashboard
  JSON files from a mounted folder). This means the dashboard exists the
  moment the stack starts -- nobody has to click through the UI to
  recreate it, and it's versioned in git like any other config.
- Hand-wrote `dashboards/m5-forecasting.json` (6 panels: request rate by
  endpoint, p95 latency by endpoint, total request count, `/predict`
  count, drift KS statistic as a gauge with color thresholds, drift
  p-value as a stat panel) rather than building it in the UI first --
  faster to iterate on directly as text, and Grafana's dashboard JSON
  schema is stable enough to write by hand for a panel count this small.

### Verification
- Didn't stop at "containers started." Checked each layer of the actual
  data path independently:
  1. Prometheus's own target-health API
     (`/api/v1/targets`) -- confirmed `job=m5-forecast-api` status `up`,
     i.e. Prometheus is actually successfully scraping the app, not just
     configured to try.
  2. Generated real `/predict` traffic across 3 different series (15
     requests, 105 individual predictions) and re-checked `/metrics` --
     confirmed both the request counters and the drift gauges updated
     with real numbers, not placeholder zeros.
  3. Fetched the dashboard definition back via Grafana's own API
     (`/api/dashboards/uid/m5-forecasting`) to confirm all 6 panels
     provisioned correctly, not just that the YAML didn't error.
  4. Queried the drift metric **through Grafana's own datasource proxy**
     (`/api/datasources/proxy/.../query?query=prediction_drift_ks_statistic`)
     and confirmed the returned value matched what `/metrics` reported
     directly -- proof the full chain (app -> Prometheus -> Grafana)
     carries the same real number end to end, not just that each piece
     independently "looks fine."

---

## Phase 7 — Full model comparison (ARIMA, Linear Regression, Random
## Forest, XGBoost, LightGBM) + champion registration + Streamlit UI

After the original 6-phase pipeline was complete, decided to turn this
into a fuller data-science portfolio piece: compare Prophet against
classical (ARIMA) and traditional ML (Linear Regression, Random Forest,
XGBoost, LightGBM) approaches on the same held-out window, register
whichever wins as an MLflow "champion," and build a Streamlit UI to
explore the comparison. Also dropped Prophet as the CV headline (kept it
in the pipeline, since it's a legitimate comparison point) in favor of
whichever model actually won -- you shouldn't put a tool on a CV you
can't defend in an interview.

### Global vs. per-series -- the key design fork
- Prophet and ARIMA are inherently **per-series**: each fits only on one
  series' own history, with no natural way to see other series at all.
- Linear Regression, Random Forest, XGBoost, and LightGBM were instead
  trained as **one global model each**, on all 100 series stacked into a
  single table with `store_id`/`item_id`/`dept_id`/`cat_id`/`state_id` as
  features. This is how the real M5 competition's top solutions actually
  approached the problem (not a shortcut) -- it lets the model learn
  cross-series patterns ("TX stores sell more," "FOODS_3 spikes on
  weekends") that a per-series model structurally cannot see, and it
  means 4 models total instead of 400.

### Feature engineering for the global models
  (`src/feature_engineering.py`)
- **Leakage constraint, and why `lag_28` specifically:** this is a
  28-day-*ahead* forecast, not a 1-day-ahead rolling one. A `lag_7`
  feature would be invalid for most of the test window -- predicting day
  15 of the horizon with "sales 7 days ago" requires knowing sales from
  day 8 of the horizon, which hasn't happened yet at forecast time.
  `lag_28` is the largest lag that stays valid for *every* day across the
  full 28-day horizon (lag_28 on the last test day still points at the
  last training day). Every history-based feature -- the lag itself,
  plus 7-/28-day rolling mean/std computed on top of the lag-28-shifted
  series -- is built on this constraint. This mirrors how top M5
  competition solutions actually handled the exact same problem, not an
  invented workaround.
- Also joined `sell_prices.csv` (never used by the Prophet/ARIMA phase)
  on `(store_id, item_id, wm_yr_wk)`, and encoded categoricals two ways:
  integer/label codes for the tree models (fine -- trees split on
  thresholds regardless of encoding), one-hot for Linear Regression via
  a `ColumnTransformer` (label codes would wrongly imply an ordering a
  linear model would try to use).
- 5,500 rows (100 series x 55 days) dropped for insufficient lag_28
  history at the start of each series -- same shape as the earlier
  Prophet-training row-count math, a good sanity check that the join/
  shift logic was right before training anything on it.

### Training the 4 global models (`src/train_ml_models.py`)
- Straightforward compared to Prophet/ARIMA: one `.fit()` / `.predict()`
  call each (no per-series loop), so one MLflow run per model (not
  nested runs). Registered each under its own name (`sales_linear_
  regression`, `sales_random_forest`, `sales_xgboost`, `sales_lightgbm`).
- LightGBM given true categorical columns (`categorical_feature=`,
  values cast to pandas `category` dtype) rather than raw integer codes,
  for better split quality than the label-encoded columns RF/XGBoost got
  -- a deliberate difference, not an oversight.
- Deliberately no hyperparameter tuning (sensible defaults only) -- per
  this project's own stated goal, the comparison and pipeline are the
  point, not squeezing out the last percent of MAPE.

### Training ARIMA (`src/train_arima.py`)
- Per-series SARIMA (`pmdarima.auto_arima`, seasonal period 7 to match
  the weekly seasonality EDA found), with `snap` passed as an exogenous
  regressor -- same reasoning as Prophet's `add_regressor`.
- **Runtime problem and fix:** a single `auto_arima` call took ~35s
  (measured directly before committing to all 100). Sequential, that's
  ~1hr+. Since each series' fit is fully independent, parallelized with
  `joblib.Parallel(n_jobs=-1)` across all 12 available CPU cores --
  fitting dropped to a few minutes. MLflow logging stayed sequential in
  the main process afterward (concurrent writers to nested runs across
  separate processes is asking for trouble), so the real wall-clock cost
  became parallel-fit-time + sequential-log-time, not
  parallel-fit-time alone.

### A genuine, surprising result: ARIMA scored *worse than the naive
### baseline* on mean per-series MAPE -- investigated before reporting it
- First result: ARIMA's mean per-series MAPE was 85.2% (worse than
  naive's 78.6%) and aggregate MAPE 18.5% (worse than naive's 10.75% and
  every other model's). Rather than accept "ARIMA is just bad here" at
  face value or quietly drop the result, checked whether this was a real
  finding or a bug:
  1. Pulled the per-series MAPE distribution: mean 85.2%, but **median
     43.9%**, std 139.5, max **856%**. A distribution whose std exceeds
     its mean is a strong tell that a few extreme outliers are doing the
     damage, not a broad, even weakness.
  2. Traced the single worst series (`FOODS_3_808_CA_3_validation`,
     856% MAPE) back to its actual raw data: steady ~20-57 units/day
     through training, then a collapse to almost entirely zero for most
     of the 28-day test window -- a real stockout or discontinuation,
     invisible in training history, that *no* history-only model could
     have anticipated.
  3. Confirmed ARIMA's selected order for that series
     (`(2,1,2)(1,0,0,7)`) via `search_runs` -- it extrapolated the
     pre-collapse trend forward as expected. Because MAPE divides by the
     (now near-zero) actual value, an ordinary absolute miss on the
     handful of remaining non-zero test days turned into a triple-digit
     percentage error that, averaged into a 100-series mean, dragged the
     whole model below the naive baseline.
- **This is not a bug -- it's the same MAPE small-denominator problem
  already documented in Phase 0**, now demonstrated with a concrete,
  traced-through example instead of an abstract caveat. Reported
  honestly rather than hidden, and it directly motivated the next
  decision.

### Champion selection: aggregate MAPE, not mean per-series MAPE -- and why
- The ARIMA finding is exactly why **aggregate MAPE was used as the
  model-*selection* metric**, not mean per-series MAPE: a selection
  metric that lets one pathological series disqualify an otherwise-
  reasonable model is a bad selection metric, even though that same
  per-series number is still worth reporting for transparency (which
  `results/full_model_comparison.csv` and the Streamlit UI both do).
- `src/compare_models.py` assembles every model's summary + per-series
  files into one table, sorted by aggregate MAPE. **LightGBM won**
  (5.82% aggregate MAPE, beating Prophet's 6.57%).
- Registered the champion via MLflow's **alias** mechanism (not the
  older, now-deprecated "stage" concept):
  `client.set_registered_model_alias("sales_lightgbm", "champion",
  version)`. Verified by loading it back through the alias URI
  (`models:/sales_lightgbm@champion`) rather than assuming the call
  succeeded -- same "verify, don't assume" habit as every earlier phase.
  Had to re-point the alias after later reruns bumped the model to v2,
  then v3 -- a real reminder that an alias needs to be re-set after every
  retrain, it doesn't follow "whatever is newest" automatically.

### Streamlit UI (`ui/app.py`)
- Two tabs: a **per-series backtest view** (actual sales history +
  test-window predictions from any subset of the 7 models, overlaid on
  one Plotly chart, with that series' own MAPE per model) and an
  **overall comparison view** (the full aggregate-MAPE table + bar
  chart, with the champion called out).
- **Deliberately visualizes pre-computed backtest results, not live
  future forecasts.** The champion (LightGBM) is a global model that
  needs the full engineered feature set (lag_28, rolling stats, price,
  calendar) to score a new row -- building that feature pipeline live
  for arbitrary future dates is a meaningfully different, larger piece of
  work than a comparison demo needs. What's actually useful to show is
  "how did each approach do on the same held-out window," which the
  existing `results/*_forecasts.csv` files already answer directly.
  Named this as a deliberate scope decision (in both the code's
  docstring and `results/writeup.md`), not silently limited it.
- Required adding row-level forecast CSV exports
  (`results/{model}_forecasts.csv`, `id`/`date`/`sales`/`forecast`) to
  `baseline.py`, `train_arima.py`, and `train_ml_models.py` -- only
  Prophet had saved these already from Phase 1. Reran all three after
  adding the export lines (ARIMA's ~15-20 min rerun was the expensive
  one, parallelized fitting again).
- Verified past "the server responds": loaded every CSV the app depends
  on directly (bypassing Streamlit's caching layer) to confirm shapes
  and columns before trusting the UI, then hit the running app with
  `curl` and grepped its log for tracebacks after triggering a real page
  load -- the same "check the actual data path, not just that a process
  started" habit used throughout this project.

### Final comparison

| model | mean per-series MAPE | median per-series MAPE | aggregate MAPE |
|---|---|---|---|
| **lightgbm** | 73.45% | 33.60% | **5.82%** |
| linear_regression | 80.73% | 36.09% | 6.45% |
| xgboost | 74.71% | 33.55% | 6.48% |
| prophet | 68.58% | 33.83% | 6.57% |
| random_forest | 87.17% | 32.79% | 7.29% |
| naive_seasonal_lag7 | 78.58% | 40.86% | 10.75% |
| arima | 85.21% | 43.87% | 18.46% |

---
