import os
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from common import CHAMPION_ALIAS as ALIAS
from common import MLFLOW_TRACKING_URI_DEFAULT, PROMOTABLE_MODELS

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = "results"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT)


def get_current_champion(client: MlflowClient):
    """Return the (name, version, aggregate_mape) of whichever promotable
    model currently carries the champion alias.

    At most one model should ever carry it -- promote() below removes the
    alias from every other promotable model whenever it promotes a new one,
    so a stale alias left on an old model family can't be mistaken for the
    live champion.
    """
    found = []
    for name in PROMOTABLE_MODELS.values():
        try:
            mv = client.get_model_version_by_alias(name, ALIAS)
        except Exception:
            continue
        run = client.get_run(mv.run_id)
        found.append((name, mv.version, run.data.metrics.get("aggregate_mape")))
    if len(found) > 1:
        print(f"WARNING: {len(found)} models carry the '{ALIAS}' alias simultaneously "
              f"({[n for n, _, _ in found]}); this should not happen. Using the first.")
    if not found:
        return None, None, None
    return found[0]


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

    current_name, current_version, current_mape = get_current_champion(client)

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

    # Only one model should ever carry the champion alias. If the new
    # champion is a different model family than the previous one, the old
    # family's alias is now stale -- drop it so a future lookup can't find
    # two (or more) models both claiming to be champion.
    for name in PROMOTABLE_MODELS.values():
        if name == best_name:
            continue
        try:
            client.delete_registered_model_alias(name, ALIAS)
        except Exception:
            pass

    if current_mape is None:
        print(f"No prior champion found. Promoted {best_name} v{latest.version} "
              f"({best_mape:.2f}% aggregate MAPE) as the first champion.")
    else:
        print(f"New champion: {best_name} v{latest.version} ({best_mape:.2f}%) "
              f"beats previous champion {current_name} ({current_mape:.2f}%).")


if __name__ == "__main__":
    main()
