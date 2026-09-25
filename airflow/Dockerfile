# Extends the official Airflow image with this project's training
# dependencies so the DAG can run the retraining scripts.
#
# Build from repo root: docker build -f airflow/Dockerfile -t m5-airflow .
FROM apache/airflow:2.10.4-python3.10

# LightGBM needs the GNU OpenMP runtime at import time; the base image
# doesn't ship it.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY airflow/requirements.txt /requirements.txt

# Installed into an isolated venv rather than Airflow's own environment,
# since mlflow/scikit-learn's dependency requirements conflict with
# versions Airflow's own constraints file pins. The DAG invokes this
# venv's python directly.
RUN python -m venv /opt/airflow/task_venv && \
    /opt/airflow/task_venv/bin/pip install --no-cache-dir -r /requirements.txt
