# Build from the repo root: docker build -f serving/Dockerfile -t m5-forecast-api .
# (needs repo-root context for the two small data files it copies, not
# just the serving/ folder)
#
# Prerequisite: an `mlflow server` running on the host, reachable from
# containers as host.docker.internal (Docker Desktop's DNS alias for the
# host machine). Must be started with that hostname in --allowed-hosts,
# port included, or MLflow's DNS-rebinding protection will 403 it:
#   mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 \
#     --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
#
# Then run this image against it:
#   docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 m5-forecast-api
#
# Model artifacts are NOT baked into this image. Earlier attempt did that
# (COPY mlflow.db + mlruns/ in) and it broke: MLflow's local file store
# bakes an absolute host filesystem path into each experiment's artifact
# location, which doesn't exist inside the container. Pointing at a real
# tracking server over HTTP (MLFLOW_TRACKING_URI) is the actual fix, not
# a workaround -- see results/LEARNING_LOG.md for the full story.

FROM python:3.10-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ serving/

# Static reference data /predict needs regardless of where MLflow runs:
# calendar.csv for the snap regressor, subset_long.csv for valid series
# ids + the forecast window.
COPY data/raw/calendar.csv data/raw/calendar.csv
COPY data/processed/subset_long.csv data/processed/subset_long.csv

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
