#!/usr/bin/env bash
# Resume-resilient launcher for the RESOLUTION ABLATION (step 2):
# CT-FM backbone + Mask2Former head at mask_feat_stride=1 (full-res masks),
# dec_layers=3, bs=2 x grad_accum 2. GPU 1.
#
# Mirrors run_ctfm_mask2former.sh (the stride-2 baseline). Relaunches
# `train --resume` until the run completes (.done) or crashes fast 3x in a row.
# Horizon capped at 150 epochs (--epochs 150) — enough to compare the val
# trajectory against the stride-2 run at its 25/50/.../150 checkpoints; this is
# a confirmation, not a full 500-epoch run.
#
# Detach from a login shell so it survives tool-call reaping:
#     setsid nohup bash scripts/run_ctfm_mask2former_s1.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=1 (train.py uses cuda:0 -> physical GPU 1).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env/bin/python}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
# Full-res masks reserve ~41 GB with ~4 GB fragmentation; expandable segments
# claws that back so the run keeps headroom under the 49 GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="configs/models/ctfm_mask2former_s1.yaml"
OUT="$REPO/runs/ctfm_mask2former_s1"
EPOCHS="${EPOCHS:-150}"
MAX_TRIES="${MAX_TRIES:-20}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-180}"
FAST_CRASH_CAP="${FAST_CRASH_CAP:-3}"

mkdir -p "$OUT"
LOOP_LOG="$OUT/loop.log"
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOOP_LOG"; }

log "=== launcher start: cfg=$CFG out=$OUT gpu=$CUDA_VISIBLE_DEVICES epochs=$EPOCHS wandb=$WANDB_MODE ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOOP_LOG"

fast_crashes=0
for ((i=1; i<=MAX_TRIES; i++)); do
    if [ -f "$OUT/.done" ]; then log "already .done — nothing to do"; break; fi
    log ">>> attempt $i/$MAX_TRIES: train --resume --epochs $EPOCHS"
    start=$SECONDS
    "$PY" -m scripts.train --config "$CFG" --output "$OUT" --epochs "$EPOCHS" --resume 2>&1 | tee -a "$OUT/console.log"
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
