# Build from repo root: docker build -f airflow/Dockerfile -t mlops-airflow:local .
#
# Runs as a single `airflow standalone` service (see docker-compose.yaml) --
# no separate postgres/init/webserver/scheduler containers. The whole repo
# is bind-mounted at /opt/airflow/project, so ingestion, `dvc repro`, and
# the git-commit step (see airflow/dags/retrain_dag.py) run as plain
# BashOperator commands directly in this container -- that's why git, dvc,
# and pandas/numpy are installed here. The heavier ML libraries
# (scikit-learn, LightGBM, XGBoost, statsmodels, pmdarima, mlflow) are NOT
# installed here; those stay in the separate m5-training image that
# DockerOperator launches for the actual retrain/compare/promote tasks.
FROM apache/airflow:2.10.4-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
