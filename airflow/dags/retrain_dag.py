from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

VENV_PYTHON = "/opt/airflow/task_venv/bin/python"
PROJECT_DIR = "/opt/airflow/project"


def run(script: str) -> str:
    return f"cd {PROJECT_DIR} && {VENV_PYTHON} {script}"


with DAG(
    dag_id="m5_full_retrain",
    description="Weekly retrain + compare all forecasting approaches, auto-promote champion",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["m5", "retraining", "multi-model"],
) as dag:
    retrain_baseline = BashOperator(
        task_id="retrain_baseline",
        bash_command=run("src/baseline.py"),
        execution_timeout=timedelta(minutes=5),
    )
    retrain_svm = BashOperator(
        task_id="retrain_svm",
        bash_command=run("src/train_svm.py"),
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
    [retrain_baseline, retrain_svm, retrain_arima, retrain_ml_models] >> compare_models
    compare_models >> promote_champion
