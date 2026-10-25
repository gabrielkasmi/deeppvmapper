#!/bin/bash
set -euo pipefail

# Quick experiments — small subset, few epochs, one candidate backbone at a
# time (refs #11). All models share seed/subset logic, so runs are comparable.
#
# Usage:
#   bash scripts/run_quick.sh <model> [extra train.py args]
#   bash scripts/run_quick.sh all          # segformer + deeplab + unet
#
# Models (keys of configs/model.yaml):
#   segformer | segformer-b1 | deeplab | unet | unet-efficientnet
#
# Env:
#   SUBSET   fraction of each split to train on (default 0.05)
#   PYTHON   python executable (default python)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
SUBSET="${SUBSET:-0.05}"

MODEL="${1:-segformer}"
shift || true

RUN_NAME=""
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

run_one() {
    local model="$1"
    local run="${RUN_NAME:-${model}-mini}"

    if [ -f "runs/${run}/metrics.json" ]; then
        echo "[quick] runs/${run} already trained — skipping (delete to retrain)"
    else
        echo "[quick] training ${model} -> runs/${run}"
        "$PYTHON" train.py --config configs/train.yaml --model "$model" \
            --subset "$SUBSET" --run-name "$run" ${PASS[@]+"${PASS[@]}"}
    fi

    "$PYTHON" eval.py --checkpoint "runs/${run}/checkpoints/best.pth" \
        --out "experiments/results/${run}.json" \
        --previews-dir "experiments/results/previews/${run}"

    echo "[quick] ${run}: experiments/results/${run}.json + previews/${run}/"
}

if [ "$MODEL" = "all" ]; then
    for m in segformer deeplab unet; do
        run_one "$m"
    done
else
    run_one "$MODEL"
fi
