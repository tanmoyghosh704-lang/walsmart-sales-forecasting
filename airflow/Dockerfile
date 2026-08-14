# Extends the official Airflow image with this project's training deps
# (prophet/mlflow/pandas) so the DAG can actually run src/train.py.
#
# Build from repo root: docker build -f airflow/Dockerfile -t m5-airflow .
FROM apache/airflow:2.10.4-python3.10

# LightGBM's compiled extension needs the GNU OpenMP runtime at import
# time, not just at pip-install time -- pip installing the lightgbm
# wheel succeeds fine without it, but `import lightgbm` inside a task
# then fails with `OSError: libgomp.so.1: cannot open shared object
# file`. This is a system library, not a Python package, so it has to
# come from apt, not pip. Needs root; switches back to airflow after.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY airflow/requirements.txt /requirements.txt

# Installed into an isolated venv, NOT Airflow's own Python environment.
# First attempt used `pip install ... --constraint <airflow-constraints>`
# directly into Airflow's environment, matching the officially recommended
# pattern -- but it failed twice in a row (pandas, then cryptography)
# because mlflow/prophet's own dependency requirements collide with
# versions Airflow's constraints file pins exactly. Rather than pin
# packages one at a time to chase an ever-moving conflict, isolating our
# training deps in their own venv sidesteps the problem entirely: nothing
# we install here can ever break Airflow's own packages, because they
# don't share an environment. The DAG invokes this venv's python directly.
RUN python -m venv /opt/airflow/task_venv && \
    /opt/airflow/task_venv/bin/pip install --no-cache-dir -r /requirements.txt
