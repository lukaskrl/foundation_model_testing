#!/usr/bin/env bash
# Queue for the H2 pilot (multi-encoder head pretraining vs single-encoder heads).
# Reads a plan file of lines:   run_name | config.yaml | epochs | extra train.py args
# Same retry / .done / fast-crash logic as run_interface.sh. A run whose
# --init-head-from checkpoint does not exist yet (head M still training) is
# SKIPPED, not failed — rerun the queue later and it picks up where it left off.
#
# Usage: GPU=1 PLAN=configs/h2/plan_gate.txt bash scripts/run_h2.sh
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$REPO/.." && pwd)/env/bin/python}"
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PLAN="${PLAN:?set PLAN=configs/h2/plan_*.txt}"
MAX_TRIES="${MAX_TRIES:-20}"; FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"; FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"
OUTROOT="${OUTROOT:-$REPO/runs/h2}"; mkdir -p "$OUTROOT"
QLOG="$OUTROOT/queue_gpu${CUDA_VISIBLE_DEVICES}.log"
qlog() { echo "[$(date +%F\ %T)] $*" | tee -a "$QLOG"; }

qlog "=== h2 launcher: gpu=$CUDA_VISIBLE_DEVICES plan=$PLAN ==="
while IFS='|' read -r run cfg epochs extra; do
    run="$(echo "$run" | xargs)"; cfg="$(echo "$cfg" | xargs)"; epochs="$(echo "$epochs" | xargs)"
    [ -z "$run" ] && continue; [[ "$run" == \#* ]] && continue
    OUT="$OUTROOT/$run"
    if [ -f "$OUT/.done" ]; then qlog "SKIP $run (.done)"; continue; fi
    head_ckpt="$(echo "$extra" | grep -oE -- '--init-head-from +[^ ]+' | awk '{print $2}')"
    # best.pt is rewritten at every val improvement, so its existence does NOT mean
    # the head finished training. Require the source run to have logged completion.
    if [ -n "$head_ckpt" ]; then
        if [ ! -f "$head_ckpt" ] || ! grep -q "total time" "$(dirname "$head_ckpt")/run.log" 2>/dev/null; then
            qlog "WAIT $run (head run $(dirname "$head_ckpt") not finished yet)"; continue
        fi
    fi
    mkdir -p "$OUT"; qlog ">>> RUN $run cfg=$cfg epochs=$epochs extra=$extra"
    fast=0
    for ((i=1;i<=MAX_TRIES;i++)); do
        [ -f "$OUT/.done" ] && break
        start=$SECONDS
        # shellcheck disable=SC2086
        "$PY" -m scripts.train --config "$cfg" --output "$OUT" --epochs "$epochs" $extra --resume 2>&1 | tee -a "$OUT/console.log"
        rc=${PIPESTATUS[0]}; dur=$((SECONDS-start))
        qlog "    $run attempt $i rc=$rc dur=${dur}s"
        if [ "$rc" -eq 0 ]; then touch "$OUT/.done"; qlog "    $run DONE"; break; fi
        if [ "$dur" -lt "$FAST_CRASH_SECS" ]; then
            fast=$((fast+1)); qlog "    fast crash $fast/$FAST_CRASH_CAP"
            [ "$fast" -ge "$FAST_CRASH_CAP" ] && { qlog "    ABORT $run"; break; }
        else fast=0; fi
    done
done < "$PLAN"
qlog "=== h2 launcher finished ==="
