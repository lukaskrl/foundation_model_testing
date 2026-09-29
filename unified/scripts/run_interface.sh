#!/usr/bin/env bash
# Sequential, resume-resilient launcher for a hand-written list of configs.
#
# The Arm I1/I2 configs live in configs/interface/ and deliberately do NOT appear in
# configs/lowshot/MANIFEST.csv (they are not part of the Arm N/S/B/W matrix), so
# run_lowshot_matrix.sh cannot select them. This is the same retry/.done/fast-crash
# logic, driven by an explicit list instead of the manifest.
#
# Usage:
#   GPU=1 EPOCHS=500 bash scripts/run_interface.sh configs/interface/a.yaml configs/interface/b.yaml
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$REPO/.." && pwd)/env/bin/python}"
cd "$REPO" || exit 1

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCHS="${EPOCHS:-500}"
MAX_TRIES="${MAX_TRIES:-20}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"
FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"
OUTROOT="${OUTROOT:-$REPO/runs/interface}"; mkdir -p "$OUTROOT"
QLOG="$OUTROOT/queue_gpu${CUDA_VISIBLE_DEVICES}.log"
qlog() { echo "[$(date +%F\ %T)] $*" | tee -a "$QLOG"; }

[ "$#" -gt 0 ] || { echo "usage: GPU=n bash $0 <config.yaml> [more.yaml ...]"; exit 2; }

qlog "=== interface launcher: gpu=$CUDA_VISIBLE_DEVICES epochs=$EPOCHS -> $# run(s) ==="
for cfg in "$@"; do
    run="$(basename "$cfg" .yaml)"
    OUT="$OUTROOT/$run"
    if [ -f "$OUT/.done" ]; then qlog "SKIP $run (.done)"; continue; fi
    mkdir -p "$OUT"
    qlog ">>> RUN $run  cfg=$cfg  epochs=$EPOCHS"
    fast=0
    for ((i=1;i<=MAX_TRIES;i++)); do
        [ -f "$OUT/.done" ] && break
        start=$SECONDS
        "$PY" -m scripts.train --config "$cfg" --output "$OUT" --epochs "$EPOCHS" --resume 2>&1 | tee -a "$OUT/console.log"
        rc=${PIPESTATUS[0]}; dur=$((SECONDS-start))
        qlog "    $run attempt $i rc=$rc dur=${dur}s"
        if [ "$rc" -eq 0 ]; then touch "$OUT/.done"; qlog "    $run DONE"; break; fi
        if [ "$dur" -lt "$FAST_CRASH_SECS" ]; then
            fast=$((fast+1)); qlog "    fast crash $fast/$FAST_CRASH_CAP"
            [ "$fast" -ge "$FAST_CRASH_CAP" ] && { qlog "    ABORT $run (repeated fast crash — real bug)"; break; }
        else fast=0; fi
    done
done
qlog "=== interface launcher finished ==="
