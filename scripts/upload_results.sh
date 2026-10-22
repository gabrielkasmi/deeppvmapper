#!/bin/bash
set -euo pipefail

# Upload experiment artifacts to the Hugging Face Hub (refs #11).
#
#   default:          experiments/results  (small JSON summaries + preview PNGs)
#   UPLOAD_RUNS=1:    also upload runs/    (checkpoints + full metrics)
#
# Required environment (see .env.example):
#   HUGGINGFACE_HUB_TOKEN   HF write token
#   HF_REPO_ID              target repo, e.g. <username>/deeppvmapper-backbones
# Optional:
#   HF_PRIVATE=0            create the repo as public (default: private —
#                           keep weights private until dataset licensing is
#                           confirmed, especially for the google config)
#
# Never commit .env, tokens, or checkpoints to git — runs/ and *.pth are
# gitignored; this script is the shareable path for large artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_HUB_TOKEN is not set — see .env.example" >&2
    exit 1
fi
if [ -z "${HF_REPO_ID:-}" ]; then
    echo "ERROR: HF_REPO_ID is not set — e.g. HF_REPO_ID=<username>/deeppvmapper-backbones" >&2
    exit 1
fi

python - <<'PY'
import os

from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
private = os.environ.get("HF_PRIVATE", "1") == "1"

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

paths = ["experiments/results"]
if os.environ.get("UPLOAD_RUNS") == "1":
    paths.append("runs")

for path in paths:
    print(f"Uploading {path} -> {repo_id}/{path}")
    api.upload_folder(repo_id=repo_id, repo_type="model",
                      folder_path=path, path_in_repo=path)

print(f"Done: https://huggingface.co/{repo_id}")
PY
