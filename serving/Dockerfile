# Build from the repo root: docker build -f serving/Dockerfile -t m5-forecast-api .
#
# Talks to DagsHub's hosted MLflow tracking server + registry (see
# src/common.py's MLFLOW_TRACKING_URI_DEFAULT; MLFLOW_TRACKING_USERNAME/
# PASSWORD come from .env via docker-compose.yml) to resolve and load
# whichever model currently carries the "champion" registry alias -- it
# never bakes a model into the image.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ serving/
COPY src/common.py src/features.py src/
COPY data/raw/calendar.csv data/raw/calendar.csv
COPY data/raw/sell_prices.csv data/raw/sell_prices.csv
COPY data/processed/subset_long.csv data/processed/subset_long.csv
COPY data/processed/ml_features.csv data/processed/ml_features.csv
COPY data/processed/encoders.json data/processed/encoders.json

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
