#!/usr/bin/env bash
# CT-CLIP pyramid-mode A/B on GPU 1: upsample vs multiscale, matched recipe.
#
# Why both arms: runs/ctclip_run (val_mean_dice 0.8534) was trained BEFORE
# `pyramid_mode: upsample` existed in configs/models/ctclip.yaml (added
# 2026-06-29, commit 13da93d), so it used the code default `spm` — confirmed by
# its trainable-param count 35,176,064 (spm adapter 611,792) vs upsample /
# multiscale 35,074,160 (509,888). It also ran at eff_batch=3, while the
# low-shot matrix uses eff_batch=4. So there is no matched `upsample` number to
# compare against and we train it here.
#
# The two arms differ ONLY in pyramid_mode + canvases; both adapters have
# 509,888 params, so the comparison isolates feature provenance:
#   upsample   = all 5 contract levels are resampled copies of ONE 480 pass
#   multiscale = levels drawn from 3 passes at canvases 120/200/320
#
# Crash-resilient (the node's OOM-killer transiently reaps jobs): a run with
# .done is skipped, otherwise --resume picks up the newest checkpoint. Re-run
# this exact command after any kill to continue.
set -u

REPO="/store/home/skrljl/projects/foundation_models/unified"
PY="/store/home/skrljl/projects/foundation_models/env/bin/python"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES=1          # GPU 0 belongs to other users
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"

EPOCHS="${EPOCHS:-500}"
BATCH_DIR="${1:-$REPO/runs/ctclip_pyramid_ab}"
mkdir -p "$BATCH_DIR"
STATUS="$BATCH_DIR/STATUS.log"
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$STATUS"; }

# arm_name : config
ARMS=(
  "upsample:configs/lowshot/ls_ctclip_ft_pt_f100.yaml"
  "multiscale:configs/lowshot/ls_ctclip_ms_ft_pt_f100.yaml"
)

log "=== CT-CLIP pyramid A/B ==="
log "batch dir : $BATCH_DIR"
log "epochs    : $EPOCHS   device: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv 2>&1 | tee -a "$STATUS"

for arm in "${ARMS[@]}"; do
    name="${arm%%:*}"
    cfg="${arm#*:}"
    out="$BATCH_DIR/$name"
    mkdir -p "$out"

    if [ -f "$out/.done" ]; then
        log "SKIP  $name (already .done)"
        continue
    fi

    resume=()
    if compgen -G "$out/epoch_*.pt" > /dev/null || [ -f "$out/best.pt" ]; then
        resume=(--resume)
        log "RESUME $name from newest checkpoint in $out"
    fi

    log "START $name  cfg=$cfg  out=$out"
    echo "$out/train.log" > "$REPO/runs/ACTIVE_LOG"
    "$PY" -m scripts.train \
        --config "$cfg" \
        --output "$out" \
        --epochs "$EPOCHS" \
        "${resume[@]}" \
        >> "$out/train.log" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        touch "$out/.done"
        log "DONE  $name rc=0"
    else
        log "FAIL  $name rc=$rc (see $out/train.log) — continuing to next arm"
    fi
done

log "=== ALL_DONE ==="
