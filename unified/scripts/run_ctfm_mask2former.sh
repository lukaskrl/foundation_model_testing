#!/usr/bin/env bash
# Resume-resilient single-run launcher: CT-FM backbone + Mask2Former head, GPU 1.
#
# The shared node's OOM-killer transiently SIGKILLs training. This loop relaunches
# `train --resume` (which auto-detects the latest checkpoint and continues with
# full optimizer/scheduler/epoch state) until the run completes (writes .done) or
# it crashes fast 3× in a row (a real bug, not transient OOM after progress).
#
# Re-running this exact command after ANY kill resumes from the last checkpoint —
# no work lost. Best run detached from a login shell (survives tool-call reaping):
#     setsid nohup bash scripts/run_ctfm_mask2former.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=1 (train.py uses cuda:0 → physical GPU 1). GPU 0 free.

REPO="/store/home/skrljl/projects/foundation_models/unified"
PY="/store/home/skrljl/projects/foundation_models/env/bin/python"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false

CFG="configs/models/ctfm_mask2former.yaml"
OUT="$REPO/runs/ctfm_mask2former"
MAX_TRIES="${MAX_TRIES:-20}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"   # a crash faster than this counts as "no progress"
FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"       # stop after this many consecutive fast crashes

mkdir -p "$OUT"
LOOP_LOG="$OUT/loop.log"
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOOP_LOG"; }

log "=== launcher start: cfg=$CFG out=$OUT gpu=$CUDA_VISIBLE_DEVICES wandb=$WANDB_MODE ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOOP_LOG"

fast_crashes=0
for ((i=1; i<=MAX_TRIES; i++)); do
    if [ -f "$OUT/.done" ]; then log "already .done — nothing to do"; break; fi
    log ">>> attempt $i/$MAX_TRIES: train --resume"
    start=$SECONDS
    "$PY" -m scripts.train --config "$CFG" --output "$OUT" --resume 2>&1 | tee -a "$OUT/console.log"
    rc=${PIPESTATUS[0]}
    dur=$(( SECONDS - start ))
    log "<<< attempt $i ended rc=$rc after ${dur}s"

    if [ "$rc" -eq 0 ]; then
        touch "$OUT/.done"
        log "=== TRAINING COMPLETE (rc=0) — wrote $OUT/.done ==="
        break
    fi

    if [ "$dur" -lt "$FAST_CRASH_SECS" ]; then
        fast_crashes=$(( fast_crashes + 1 ))
        log "fast crash (${dur}s < ${FAST_CRASH_SECS}s), consecutive=$fast_crashes/$FAST_CRASH_CAP"
        if [ "$fast_crashes" -ge "$FAST_CRASH_CAP" ]; then
            log "=== ABORT: $FAST_CRASH_CAP consecutive fast crashes — likely a real bug, not OOM. See console.log ==="
            exit 2
        fi
    else
        fast_crashes=0   # made real progress before dying → treat next relaunch as transient-OOM recovery
        log "crash after real runtime — assuming transient OOM, will resume"
    fi
done

[ -f "$OUT/.done" ] || log "=== launcher exhausted $MAX_TRIES tries without .done ==="
