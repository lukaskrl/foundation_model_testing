#!/usr/bin/env bash
# Head-to-head arm A: CT-FM's OWN pipeline (project-lighter) on TotalSegmentator.
#
# Faithful to CT-FM/evaluation/totalseg.yaml: SegResNetDS decoder, 96x160x160
# patches, SPL orientation, 300 epochs, validation on crops up to 192x240x240,
# MONAI DiceMetric (include_background=False, ignore_empty=True).
# Local deviations, all in evaluation/overrides/local.yaml: 1 GPU instead of 4
# (strategy auto, no sync_batchnorm), our data and cache paths.
#
# The two earlier attempts on the polymtl box died from an external SIGTERM
# (DataLoader worker killed), so this wrapper auto-resumes from last.ckpt.
#
#   GPU=0 bash scripts/run_ctfm_original.sh
set -u
REPO=/home/lukas/projects/foundation_model_testing/CT-FM
LIGHTER=$REPO/.venv/bin/lighter
CKPT=/home/lukas/projects/foundation_model_testing/weights/CT-FM/ct_fm_pretrained_segresenc.ckpt
GROUP=${GROUP:-headtohead_v1}
SAVE=$REPO/evaluation/runs/totalseg/checkpoints/ct_fm_${GROUP}
LOG=$REPO/evaluation/runs/totalseg/ct_fm_${GROUP}.loop.log
MAX_TRIES=${MAX_TRIES:-20}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p "$(dirname "$LOG")"
cd "$REPO" || exit 1
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOG"; }

log "=== arm A: CT-FM original pipeline | gpu=$CUDA_VISIBLE_DEVICES group=$GROUP ==="
for i in $(seq 1 "$MAX_TRIES"); do
    RESUME=()
    [ -f "$SAVE/last.ckpt" ] && RESUME=(--args#fit#ckpt_path="$SAVE/last.ckpt") && log "resuming from last.ckpt"
    log "try $i/$MAX_TRIES starting"
    "$LIGHTER" fit \
        --config=./evaluation/totalseg.yaml,./evaluation/baselines/segresnetds_ctfm.yaml,./evaluation/overrides/local.yaml \
        --vars#name=ct_fm --vars#project=totalseg --vars#wandb_group="$GROUP" \
        --trainer#callbacks#0#until_epoch=0 \
        --trainer#callbacks#1#save_last=true \
        --system#model#trunk#ckpt_path="$CKPT" \
        "${RESUME[@]+"${RESUME[@]}"}" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then log "finished cleanly after $i tr(y|ies)"; exit 0; fi
    log "try $i exited rc=$rc; retrying in 60s"
    sleep 60
done
log "gave up after $MAX_TRIES tries"
exit 1
