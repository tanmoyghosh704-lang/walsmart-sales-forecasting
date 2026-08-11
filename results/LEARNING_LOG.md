# Build Log — Walmart M5 MLOps Forecasting Project

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
