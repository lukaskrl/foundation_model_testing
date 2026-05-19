#!/usr/bin/env bash
# 10-epoch sanity sweep across working foundation-model adapters.
# Runs models sequentially on GPU 1, logs each to runs/<model>_e10/.
# Emits one tagged line per model start/end so a Monitor can pick it up.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/store/home/skrljl/projects/foundation_models/env/bin/python"
EPOCHS="${EPOCHS:-10}"
GPU="${GPU:-1}"

MODELS=(voco_b voco_h vista3d dino3d biomedparse ctclip)

mkdir -p runs
STATUS_LOG="runs/benchmark_sanity.status.log"
: > "$STATUS_LOG"

for m in "${MODELS[@]}"; do
    out="runs/${m}_e10"
    mkdir -p "$out"
    echo "[BENCH] START model=$m output=$out epochs=$EPOCHS gpu=$GPU at $(date -Iseconds)" | tee -a "$STATUS_LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m scripts.train \
        --config "configs/models/${m}.yaml" \
        --output "$out" \
        --epochs "$EPOCHS" \
        > "$out/train.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[BENCH] DONE  model=$m rc=0 at $(date -Iseconds)" | tee -a "$STATUS_LOG"
    else
        echo "[BENCH] FAIL  model=$m rc=$rc at $(date -Iseconds) (see $out/train.log)" | tee -a "$STATUS_LOG"
    fi
done

echo "[BENCH] ALL_DONE at $(date -Iseconds)" | tee -a "$STATUS_LOG"
