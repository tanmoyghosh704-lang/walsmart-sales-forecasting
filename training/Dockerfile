# Build from the repo root: docker build -f training/Dockerfile -t m5-training .
#
# This image is the ML toolchain only -- no dvc, no git. Those run in the
# airflow container directly (see airflow/Dockerfile); this image only
# ever runs src/baseline.py, train_knn.py, train_arima.py,
# train_ml_models.py, compare_models.py, and promote_champion.py, none of
# which touch dvc or git. Airflow's DockerOperator tasks launch one-shot
# containers from this image with the project directory bind-mounted at
# /opt/project, so the image doesn't need rebuilding every time a script
# under src/ changes -- only when a dependency in requirements.txt changes.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY training/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
