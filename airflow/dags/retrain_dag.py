"""
Single-task retraining DAG: runs src/train.py (the same Prophet training
+ MLflow logging/registration script used interactively) on a weekly
schedule.

Deliberately minimal per the project's build order: get one task working
end-to-end on a schedule before adding fetch/validate/promote steps.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="m5_prophet_retrain",
    description="Weekly retrain of all 100 series' Prophet models",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["m5", "prophet", "retraining"],
) as dag:
    retrain = BashOperator(
        task_id="retrain_prophet_models",
        # /opt/airflow/task_venv is a venv isolated from Airflow's own
        # Python env (see airflow/Dockerfile) -- prophet/mlflow live
        # there, not in Airflow's environment.
        bash_command="cd /opt/airflow/project && /opt/airflow/task_venv/bin/python src/train.py",
    )
