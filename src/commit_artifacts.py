"""Last DAG step after promote_champion: commit the dvc.lock update that
this retrain produced (and the latest model comparison table) back to
GitHub, so the git history records which data pipeline run produced
which registered/promoted MLflow model version. Commits locally even
without a token; only the push is skipped.
"""
import os
import subprocess
import sys
from datetime import datetime, timezone

GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "m5-retrain-bot")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "m5-retrain-bot@users.noreply.github.com")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "master")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# dvc.lock is the authoritative "what data produced this" record. The
# comparison table is force-added as a human-readable summary even though
# results/*.csv is normally gitignored (regenerated, not meant to be
# tracked) -- this one file is the exception, by design, so retrain
# history is visible in `git log` without needing MLflow open.
TRACKED_PATHS = ["dvc.lock", "data/processed/.gitignore"]
FORCE_ADD_PATHS = ["results/full_model_comparison.csv"]


def run(*args, check=True):
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def main():
    run("git", "config", "user.name", GIT_USER_NAME)
    run("git", "config", "user.email", GIT_USER_EMAIL)

    existing = [p for p in TRACKED_PATHS if os.path.exists(p)]
    existing_forced = [p for p in FORCE_ADD_PATHS if os.path.exists(p)]
    if existing:
        run("git", "add", *existing)
    if existing_forced:
        run("git", "add", "-f", *existing_forced)

    status = run("git", "status", "--porcelain", "--", *existing, *existing_forced, check=False)
    if not status.stdout.strip():
        print("No dvc.lock/results changes from this retrain -- nothing to commit.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run("git", "commit", "-m", f"Automated retrain {timestamp}: update dvc.lock + model comparison")
    print(run("git", "log", "-1", "--stat").stdout)

    if not GITHUB_TOKEN:
        print("GITHUB_TOKEN not set -- committed locally only. "
              "Set GITHUB_TOKEN in .env to enable auto-push.")
        return

    remote_url = run("git", "remote", "get-url", "origin").stdout.strip()
    if not remote_url.startswith("https://github.com/"):
        print(f"origin remote '{remote_url}' isn't an HTTPS GitHub URL -- skipping push.")
        return

    authed_url = remote_url.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@")
    run("git", "push", authed_url, f"HEAD:{GIT_BRANCH}")
    print(f"Pushed dvc.lock update to origin/{GIT_BRANCH}.")


if __name__ == "__main__":
    main()
