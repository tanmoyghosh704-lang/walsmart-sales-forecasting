# Build from repo root: docker build -f airflow/Dockerfile -t m5-airflow .
FROM apache/airflow:2.10.4-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY airflow/requirements.txt /requirements.txt

RUN python -m venv /opt/airflow/task_venv && \
    /opt/airflow/task_venv/bin/pip install --no-cache-dir -r /requirements.txt
