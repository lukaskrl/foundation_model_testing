#!/usr/bin/env bash
# Resume-resilient launcher for the BEST-HEAD run (step 3, "Option B"):
# CT-FM backbone + two-tier Mask2Former head — full-res final masks
# (mask_feat_stride=1) with coarse attention tier (attn_feat_stride=2),
# dec_layers=9, bs=4. GPU 1.
#
# NO --epochs cap: uses base.yaml's 500 + early_stop_patience so the run trains
# to convergence (both prior M2F runs were still improving when capped). Mirrors
# run_ctfm_mask2former_s1.sh otherwise: relaunches `train --resume` until the run
# completes (.done) or crashes fast 3x in a row.
#
# Detach from a login shell so it survives tool-call reaping:
#     setsid nohup bash scripts/run_ctfm_mask2former_s2.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=1 (train.py uses cuda:0 -> physical GPU 1). Override
# with GPU=0 env if GPU 0 is the free card:  GPU=0 bash scripts/run_..._s2.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env/bin/python}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
# Full-res final readout can fragment; expandable segments claws it back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="configs/models/ctfm_mask2former_s2.yaml"
OUT="$REPO/runs/ctfm_mask2former_s2"
MAX_TRIES="${MAX_TRIES:-20}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"
FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"

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
        fast_crashes=0
        log "crash after real runtime — assuming transient OOM, will resume"
    fi
done

[ -f "$OUT/.done" ] || log "=== launcher exhausted $MAX_TRIES tries without .done ==="
