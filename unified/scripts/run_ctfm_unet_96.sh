#!/usr/bin/env bash
# Resume-resilient launcher for the 96^3 FOV CONTROL run (partner to the 128^3
# run): CT-FM backbone + U-Net head (base default) + recipe-v2, at the base
# 96^3 patch / 96^3 eval ROI (config ctfm_unet_96.yaml). GPU 0.
#
# Batch matched to the 128^3 run: bs=2 x num_samples(2) = 4 patches/forward,
# grad_accum 2 -> optimizer batch 8. 96^3 bs=2 ~18-20 GB, fits with wide margin.
#
# NO --epochs cap by default: base.yaml 500 + early_stop_patience. Override with
# EPOCHS=300 bash scripts/run_ctfm_unet_96.sh to cap the schedule.
#
# Detach from a login shell so it survives tool-call reaping:
#     setsid nohup bash scripts/run_ctfm_unet_96.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=0 (train.py uses cuda:0 -> physical GPU 0). Override
# with GPU=1 env:  GPU=1 bash scripts/run_ctfm_unet_96.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env/bin/python}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="configs/models/ctfm_unet_96.yaml"
OUT="$REPO/runs/ctfm_unet_96"
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
