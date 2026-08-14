"""
Compare the freshly-trained global models against the currently
registered champion and conditionally re-point the `champion` alias.
Only promotes if the new candidate actually beats the current champion
on aggregate MAPE -- a retrain that regresses shouldn't silently
overwrite a better model.

Deliberately restricted to the 4 *global* models (Linear Regression,
Random Forest, XGBoost, LightGBM). Each has exactly one registered
model name (sales_<name>), so a single alias can meaningfully point at
"the current best version of that name." Prophet and ARIMA are
per-series (100 separately-registered models each, e.g. 100 different
"prophet_<series_id>" names) -- there is no single registered name an
alias could point to that would represent "the champion Prophet model."
If one of those wins on aggregate MAPE, this script reports it plainly
rather than silently ignoring it or doing something architecturally
incoherent, but it does not attempt to promote it.
"""

import os
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = "results"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
ALIAS = "champion"

PROMOTABLE_MODELS = {
    "linear_regression": "sales_linear_regression",
    "random_forest": "sales_random_forest",
    "xgboost": "sales_xgboost",
    "lightgbm": "sales_lightgbm",
}


def get_current_champion_mape(client: MlflowClient):
    """Returns (registered_name, aggregate_mape) for whichever model
    currently holds the champion alias, or (None, None) if no champion
    has ever been promoted yet."""
    for name in PROMOTABLE_MODELS.values():
        try:
            mv = client.get_model_version_by_alias(name, ALIAS)
        except Exception:
            continue
        run = client.get_run(mv.run_id)
        return name, run.data.metrics.get("aggregate_mape")
    return None, None


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    comparison = pd.read_csv(f"{RESULTS_DIR}/full_model_comparison.csv")

    overall_best = comparison.sort_values("aggregate_mape").iloc[0]
    if overall_best["model"] not in PROMOTABLE_MODELS:
        print(f"NOTE: '{overall_best['model']}' has the best overall aggregate MAPE "
              f"({overall_best['aggregate_mape']:.2f}%), but it's a per-series model, "
              f"not a single registered model this script can promote via one alias. "
              f"Considering only the promotable global models: "
              f"{list(PROMOTABLE_MODELS.keys())}.")

    promotable = comparison[comparison["model"].isin(PROMOTABLE_MODELS)]
    best = promotable.sort_values("aggregate_mape").iloc[0]
    best_key = best["model"]
    best_name = PROMOTABLE_MODELS[best_key]
    best_mape = float(best["aggregate_mape"])

    current_name, current_mape = get_current_champion_mape(client)

    if current_mape is not None and current_mape <= best_mape:
        print(f"Current champion ({current_name}, {current_mape:.2f}% aggregate MAPE) "
              f"still beats or matches the best new candidate "
              f"({best_name}, {best_mape:.2f}%). No promotion -- a retrain that "
              f"regresses should not overwrite a better model.")
        return

    versions = client.search_model_versions(f"name='{best_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(best_name, ALIAS, latest.version)
    client.update_model_version(
        name=best_name, version=latest.version,
        description=f"Champion: {best_mape:.2f}% aggregate MAPE "
                     f"(full comparison in results/full_model_comparison.csv).",
    )

    if current_mape is None:
        print(f"No prior champion found. Promoted {best_name} v{latest.version} "
              f"({best_mape:.2f}% aggregate MAPE) as the first champion.")
    else:
        print(f"New champion: {best_name} v{latest.version} ({best_mape:.2f}%) "
              f"beats previous champion {current_name} ({current_mape:.2f}%).")


if __name__ == "__main__":
    main()
