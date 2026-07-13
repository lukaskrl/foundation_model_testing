#!/usr/bin/env bash
# Resume-resilient launcher for the CT-FM + BOTH new levers run:
#   * laterality-safe L-R mirror (data.augment.flip_prob 0.5)
#   * sigmoid soft-tissue intensity window (model.preprocessing.intensity.mode)
# on top of recipe-v2 + U-Net head (config ctfm_sigmoid_mirror.yaml). Compares
# against the ctfm_unet_96 baseline (0.8673); batch matched (bs2 x nsamp2, accum2
# -> optimizer batch 8). Confounds both levers on purpose (single combined run).
#
# NO --epochs cap by default: base.yaml 500 + early_stop_patience. Override with
# EPOCHS=300 bash scripts/run_ctfm_sigmoid_mirror.sh to cap.
#
# Detach so it survives tool-call reaping:
#     GPU=1 setsid nohup bash scripts/run_ctfm_sigmoid_mirror.sh >/dev/null 2>&1 &
#
# Pins CUDA_VISIBLE_DEVICES=1 by default; override with GPU=0.

REPO="/store/home/skrljl/projects/foundation_models/unified"
PY="/store/home/skrljl/projects/foundation_models/env/bin/python"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="configs/models/ctfm_sigmoid_mirror.yaml"
OUT="$REPO/runs/ctfm_sigmoid_mirror"
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
