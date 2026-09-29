#!/usr/bin/env bash
# Head-to-head arm A: CT-FM's OWN pipeline (project-lighter) on TotalSegmentator.
#
# Faithful to CT-FM/evaluation/totalseg.yaml: SegResNetDS decoder, 96x160x160
# patches, SPL orientation, 300 epochs, validation on crops up to 192x240x240,
# MONAI DiceMetric (include_background=False, ignore_empty=True).
# Local deviations, all in configs/ctfm_original/local.yaml: 1 GPU instead of 4
# (4x grad accumulation keeps effective batch 8, strategy auto, no
# sync_batchnorm). Data, cache and checkpoint paths are passed below.
#
# One-time setup (CT-FM venv, lighter patch, dataset view): see
# scripts/setup_ctfm_original.py.
#
# The two earlier attempts on the polymtl box died from an external SIGTERM
# (DataLoader worker killed), so this wrapper auto-resumes from last.ckpt.
#
#   GPU=0 bash scripts/run_ctfm_original.sh
#
# Paths follow unified/utils/paths.py: FM_ROOT (default: parent of unified/),
# WEIGHTS_ROOT (default: $FM_ROOT/weights), DATA_ROOT (default: ~/data).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FM_ROOT=${FM_ROOT:-$(cd "$HERE/../.." && pwd)}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-$FM_ROOT/weights}
DATA_ROOT=${DATA_ROOT:-$HOME/data}
REPO=$FM_ROOT/CT-FM
LIGHTER=$REPO/.venv/bin/lighter
OVERRIDES=$(cd "$HERE/.." && pwd)/configs/ctfm_original/local.yaml
CKPT=$WEIGHTS_ROOT/CT-FM/ct_fm_pretrained_segresenc.ckpt
DATASET=$DATA_ROOT/TotalSegmentatorDataset_ctfm
NAME=ct_fm
GROUP=${GROUP:-headtohead_v1}
CACHE=$DATA_ROOT/cache/ctfm_lighter/${NAME}_${GROUP}   # per-run, so a second run cannot corrupt it
SAVE=$REPO/evaluation/runs/totalseg/checkpoints/${NAME}_${GROUP}
LOG=$REPO/evaluation/runs/totalseg/${NAME}_${GROUP}.loop.log
MAX_TRIES=${MAX_TRIES:-20}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WANDB_MODE=${WANDB_MODE:-online}

die() { echo "run_ctfm_original: $*" >&2; exit 1; }
[ -x "$LIGHTER" ] || die "no $LIGHTER -- create the CT-FM venv (scripts/setup_ctfm_original.py)"
[ -f "$CKPT" ] || die "no $CKPT -- bash scripts/download_weights.sh ctfm"
[ -f "$DATASET/meta.csv" ] || die "no $DATASET/meta.csv -- run scripts/setup_ctfm_original.py"
SYSTEM_PY=$("$REPO/.venv/bin/python" -c "import lighter.system as s; print(s.__file__)") || die "lighter not importable"
grep -q on_train_epoch_start "$SYSTEM_PY" \
    || die "lighter is unpatched (crashes every validation epoch) -- run scripts/setup_ctfm_original.py"

mkdir -p "$(dirname "$LOG")"
cd "$REPO" || exit 1
log() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOG"; }

log "=== arm A: CT-FM original pipeline | gpu=$CUDA_VISIBLE_DEVICES group=$GROUP ==="
for i in $(seq 1 "$MAX_TRIES"); do
    RESUME=()
    [ -f "$SAVE/last.ckpt" ] && RESUME=(--args#fit#ckpt_path="$SAVE/last.ckpt") && log "resuming from last.ckpt"
    log "try $i/$MAX_TRIES starting"
    "$LIGHTER" fit \
        --config=./evaluation/totalseg.yaml,./evaluation/baselines/segresnetds_ctfm.yaml,"$OVERRIDES" \
        --vars#name=$NAME --vars#project=totalseg --vars#wandb_group="$GROUP" \
        --vars#dataset_dir="$DATASET" --vars#cache_dir="$CACHE" --vars#save_dir="$SAVE" \
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
