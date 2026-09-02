#!/usr/bin/env bash
# Resume-resilient launcher for the BEST-EFFORT FOV run:
# CT-FM backbone + U-Net head (base default) + recipe-v2, at 128^3 patch / 128^3
# eval ROI (config ctfm_unet_128.yaml). GPU 1.
#
# bs=2 x num_samples(2) = 4 patches/forward, grad_accum 2 -> optimizer batch 8.
# VRAM probe: 35.4 GB peak / 41.3 reserved (fits 49 GB w/ expandable_segments).
#
# NO --epochs cap by default: uses base.yaml's 500 + early_stop_patience so the
# run trains to convergence. Override with e.g.  EPOCHS=300 bash scripts/run_..sh
# to cap the schedule (also shortens the cosine anneal to that horizon).
#
# Detach from a login shell so it survives tool-call reaping:
#     setsid nohup bash scripts/run_ctfm_unet_128.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=1 (train.py uses cuda:0 -> physical GPU 1). Override
# with GPU=0 env if GPU 0 is the free card:  GPU=0 bash scripts/run_ctfm_unet_128.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env/bin/python}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
# 128^3 activations fragment more; expandable segments claws the reserved back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="configs/models/ctfm_unet_128.yaml"
OUT="$REPO/runs/ctfm_unet_128"
MAX_TRIES="${MAX_TRIES:-20}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"
FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"
EPOCHS_ARG=""
[ -n "${EPOCHS:-}" ] && EPOCHS_ARG="--epochs ${EPOCHS}"

mkdir -p "$OUT"
LOOP_LOG="$OUT/loop.log"
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOOP_LOG"; }

log "=== launcher start: cfg=$CFG out=$OUT gpu=$CUDA_VISIBLE_DEVICES wandb=$WANDB_MODE epochs_arg='${EPOCHS_ARG:-<base 500>}' ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOOP_LOG"

fast_crashes=0
for ((i=1; i<=MAX_TRIES; i++)); do
    if [ -f "$OUT/.done" ]; then log "already .done — nothing to do"; break; fi
    log ">>> attempt $i/$MAX_TRIES: train --resume"
    start=$SECONDS
    "$PY" -m scripts.train --config "$CFG" --output "$OUT" $EPOCHS_ARG --resume 2>&1 | tee -a "$OUT/console.log"
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
        fast_crashes=0
        log "crash after real runtime — assuming transient OOM, will resume"
    fi
done

[ -f "$OUT/.done" ] || log "=== launcher exhausted $MAX_TRIES tries without .done ==="
