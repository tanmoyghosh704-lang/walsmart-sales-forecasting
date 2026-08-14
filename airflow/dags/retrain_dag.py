"""
Multi-task weekly retraining DAG: retrains every one of the 7 forecasting
approaches compared in this project, rebuilds the comparison table, and
conditionally promotes a new MLflow registry champion.

    retrain_baseline  ---\
    retrain_prophet   ----\
    retrain_arima     -----+--> compare_models --> promote_champion
    build_features -> retrain_ml_models --/

Supersedes the original single-task version (which only retrained
Prophet) -- that was a deliberate "start minimal, prove the orchestration
mechanics work" first step (see results/LEARNING_LOG.md Phase 5). This
is the follow-up once there were actually multiple approaches worth
retraining and comparing.

Runtime note: retrain_arima dominates the DAG's wall-clock time
(~15-20 min -- parallelized across CPU cores via joblib, but capped to
4 workers regardless of host core count; see src/train_arima.py for why
a higher cap twice made Docker Desktop's own daemon unresponsive) --
the other tasks finish in well under a minute each.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

VENV_PYTHON = "/opt/airflow/task_venv/bin/python"
PROJECT_DIR = "/opt/airflow/project"


def run(script: str) -> str:
    return f"cd {PROJECT_DIR} && {VENV_PYTHON} {script}"


with DAG(
    dag_id="m5_full_retrain",
    description="Weekly retrain + compare all 7 forecasting approaches, auto-promote champion",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # catchup=False still lets Airflow auto-run the single most recent past
    # interval on DAG activation, which can overlap with a manual trigger.
    # retrain_arima parallelizes across all CPU cores (see src/train_arima.py) --
    # two concurrent runs each grabbing every core oversubscribed the host badly
    # enough to make Docker Desktop's own daemon unresponsive. max_active_runs=1
    # makes that structurally impossible: Airflow queues a second run instead of
    # starting it alongside the first.
    max_active_runs=1,
    tags=["m5", "retraining", "multi-model"],
) as dag:
    # execution_timeout on every task: without this, a task that hangs
    # (as retrain_prophet and retrain_arima did during the Docker-daemon
    # incident this DAG was debugged against -- stuck on a DB heartbeat
    # failure with no progress) blocks the whole DAG indefinitely instead
    # of failing loudly. Sized generously above each task's normal runtime.
    retrain_baseline = BashOperator(
        task_id="retrain_baseline",
        bash_command=run("src/baseline.py"),
        execution_timeout=timedelta(minutes=5),
    )
    retrain_prophet = BashOperator(
        task_id="retrain_prophet",
        bash_command=run("src/train.py"),
        execution_timeout=timedelta(minutes=15),
    )
    retrain_arima = BashOperator(
        task_id="retrain_arima",
        bash_command=run("src/train_arima.py"),
        execution_timeout=timedelta(minutes=40),
    )
    build_features = BashOperator(
        task_id="build_features",
        bash_command=run("src/feature_engineering.py"),
        execution_timeout=timedelta(minutes=5),
    )
    retrain_ml_models = BashOperator(
        task_id="retrain_ml_models",
        bash_command=run("src/train_ml_models.py"),
        execution_timeout=timedelta(minutes=10),
    )
    compare_models = BashOperator(
        task_id="compare_models",
        bash_command=run("src/compare_models.py"),
        execution_timeout=timedelta(minutes=5),
    )
    promote_champion = BashOperator(
        task_id="promote_champion",
        bash_command=run("src/promote_champion.py"),
        execution_timeout=timedelta(minutes=5),
    )

    build_features >> retrain_ml_models
    [retrain_baseline, retrain_prophet, retrain_arima, retrain_ml_models] >> compare_models
    compare_models >> promote_champion
