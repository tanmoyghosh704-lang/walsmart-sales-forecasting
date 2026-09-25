# Build from the repo root: docker build -f serving/Dockerfile -t m5-forecast-api .
#
# Requires an `mlflow server` reachable from the container at
# host.docker.internal (Docker Desktop's DNS alias for the host):
#   mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 \
#     --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
#
# Run:
#   docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 m5-forecast-api

FROM python:3.10-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ serving/

# Static reference data /predict needs: calendar.csv for the snap
# feature, subset_long.csv for valid series ids + the forecast window.
COPY data/raw/calendar.csv data/raw/calendar.csv
COPY data/processed/subset_long.csv data/processed/subset_long.csv

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
