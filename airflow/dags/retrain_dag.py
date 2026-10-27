import os
from datetime import datetime, timedelta

import docker
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_DIR = "/opt/airflow/project"  # repo bind-mounted here (see docker-compose.yaml)

# Only the retrain/compare/promote tasks need the heavy ML/MLflow image --
# everything else (ingestion, dvc repro, git) runs directly in this
# container via BashOperator, since it already has git/dvc/pandas and the
# whole repo mounted. HOST_PROJECT_DIR must be the repo path as seen by
# the Docker *daemon* (not by this container) -- see docker-compose.yaml.
TRAINING_IMAGE = os.environ.get("TRAINING_IMAGE", "m5-training:latest")
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "mlops-network")
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", "")

# MLflow is DagsHub-hosted (see src/common.py's MLFLOW_TRACKING_URI_DEFAULT)
# -- not a compose service, so there's nothing to point at by service name.
# These three come from the airflow container's own environment (populated
# via docker-compose.yaml's `env_file: .env`) and get forwarded into every
# training container this DAG launches.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "https://dagshub.com/tanmoyghosh704-lang/walmart-mlops-forecasting.mlflow"
)
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

PROJECT_MOUNT = Mount(source=HOST_PROJECT_DIR, target="/opt/project", type="bind")


def bash_task(task_id: str, command: str) -> BashOperator:
    return BashOperator(task_id=task_id, bash_command=f"cd {PROJECT_DIR} && {command}")


def training_task(task_id: str, command: list, timeout: timedelta) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=TRAINING_IMAGE,
        command=command,
        working_dir="/opt/project",
        mounts=[PROJECT_MOUNT],
        network_mode=DOCKER_NETWORK,
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_TRACKING_USERNAME": MLFLOW_TRACKING_USERNAME,
            "MLFLOW_TRACKING_PASSWORD": MLFLOW_TRACKING_PASSWORD,
        },
        mount_tmp_dir=False,
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        execution_timeout=timeout,
    )


def reload_serving_app():
    """Force the running serving-app container to pick up the newly
    promoted champion immediately, instead of waiting for its own
    MODEL_CACHE_TTL_SECONDS cache to expire. A DockerOperator doesn't fit
    this well -- it always creates a new one-shot container, but what's
    needed here is "restart this one already-running compose service" --
    so this talks to the Docker daemon directly instead.
    """
    client = docker.from_env()
    containers = client.containers.list(filters={"label": "com.docker.compose.service=serving-app"})
    if not containers:
        print("No running serving-app container found -- it will pick up the new "
              "champion on its own within MODEL_CACHE_TTL_SECONDS.")
        return
    for container in containers:
        container.restart(timeout=30)
        print(f"Restarted {container.name} to pick up the newly promoted champion.")


with DAG(
    dag_id="m5_full_retrain",
    description="ingest new data -> dvc repro (validate/prepare/feature_engineering) -> "
                "commit dvc.lock -> retrain all models -> compare -> promote champion -> "
                "reload serving",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["m5", "retraining", "multi-model"],
) as dag:
    # --- Data: runs directly in this container (git/dvc/pandas baked in) ---
    ingest_new_data = bash_task("ingest_new_data", "python src/ingest_new_data.py")
    dvc_validate = bash_task("dvc_validate", "dvc repro -s validate")
    dvc_prepare = bash_task("dvc_prepare", "dvc repro -s prepare")
    dvc_feature_engineering = bash_task("dvc_feature_engineering", "dvc repro -s feature_engineering")
    commit_dvc_lineage = bash_task("commit_dvc_lineage", "python src/commit_artifacts.py")

    # --- Training + MLflow registry: one-shot m5-training containers ---
    retrain_baseline = training_task("retrain_baseline", ["python", "src/baseline.py"], timedelta(minutes=5))
    retrain_knn = training_task("retrain_knn", ["python", "src/train_knn.py"], timedelta(minutes=15))
    retrain_arima = training_task("retrain_arima", ["python", "src/train_arima.py"], timedelta(minutes=40))
    retrain_ml_models = training_task(
        "retrain_ml_models", ["python", "src/train_ml_models.py"], timedelta(minutes=10)
    )
    compare_models = training_task("compare_models", ["python", "src/compare_models.py"], timedelta(minutes=5))
    promote_champion = training_task(
        "promote_champion", ["python", "src/promote_champion.py"], timedelta(minutes=5)
    )

    # --- Deployment: restart the existing serving-app service ---
    reload_serving = PythonOperator(task_id="reload_serving", python_callable=reload_serving_app)

    ingest_new_data >> dvc_validate >> dvc_prepare
    dvc_prepare >> [retrain_baseline, retrain_knn, retrain_arima, dvc_feature_engineering]
    dvc_feature_engineering >> retrain_ml_models
    # Commits dvc.lock as soon as the data pipeline is done, in parallel
    # with training rather than blocking it -- the git history records
    # exactly what data each training run used.
    dvc_feature_engineering >> commit_dvc_lineage

    [retrain_baseline, retrain_knn, retrain_arima, retrain_ml_models] >> compare_models
    compare_models >> promote_champion >> reload_serving
