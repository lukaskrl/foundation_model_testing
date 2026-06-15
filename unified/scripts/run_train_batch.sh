#!/usr/bin/env bash
# Back-to-back fine-tuning of foundation models on GPU 1.
#
#   Phase 0 (preflight): eval_smoke each model to confirm weights load + the eval
#                        path runs. Informational. Toggle with DO_PREFLIGHT=0/1.
#   Phase 1 (train):     train each model sequentially. CRASH-RESILIENT:
#                          - a model with runs/<m>_run/.done is SKIPPED,
#                          - otherwise train --resume auto-detects the latest
#                            checkpoint (epoch_*.pt / best.pt) and continues; with
#                            no checkpoint it just starts fresh,
#                          - on success we touch runs/<m>_run/.done.
#                        On a crash we record FAIL and CONTINUE to the next model.
#
# Designed to run as a persistent background job (NOT inside a tmux spawned from a
# tool call — those get reaped). Re-running this exact command after any kill
# resumes every model from its last checkpoint.
#
# Pins to the second GPU via CUDA_VISIBLE_DEVICES=1 (train.py uses cuda:0). GPU 0
# is left free.
#
# Usage: run_train_batch.sh [BATCH_DIR]
#   BATCH_DIR defaults to runs/batch_<timestamp>.

REPO="/store/home/skrljl/projects/foundation_models/unified"
PY="/store/home/skrljl/projects/foundation_models/env/bin/python"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"

# 5 freshly batch-tuned configs first, then the untuned / extra-dependency ones.
# ctfm, stunet_huge, stunet_small are excluded.
MODELS=(suprem_unet suprem_segresnet suprem_swinunetr vista3d ctclip \
        voco_b voco_h dino3d sam_med3d biomedparse)

BATCH_DIR="${1:-$REPO/runs/batch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BATCH_DIR"
STATUS="$BATCH_DIR/STATUS.log"
PREFLIGHT_LOG="$BATCH_DIR/PREFLIGHT.log"
ORCH_LOG="$BATCH_DIR/orchestrator.log"
ACTIVE_PTR="$REPO/runs/ACTIVE_LOG"          # tmux viewer follows this
echo "$BATCH_DIR" > "$REPO/runs/LATEST_BATCH"

log() { echo "[$(date +%F\ %T)] $*" | tee -a "$ORCH_LOG"; }

log "batch dir : $BATCH_DIR"
log "device    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES   WANDB_MODE=$WANDB_MODE"
log "python    : $PY"
log "models    : ${MODELS[*]}"
log "preflight : $DO_PREFLIGHT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv 2>&1 | tee -a "$ORCH_LOG"

# ---------------------------------------------------------------- Phase 0
if [ "$DO_PREFLIGHT" = "1" ]; then
    log "=== PREFLIGHT: eval_smoke per model (15-min cap each) ==="
    : > "$PREFLIGHT_LOG"
    for m in "${MODELS[@]}"; do
        cfg="configs/models/$m.yaml"
        log "preflight $m"
        out="$(timeout 900 "$PY" -m scripts.eval_smoke --config "$cfg" \
                2>"$BATCH_DIR/preflight_${m}.err")"
        rc=$?
        line="$(echo "$out" | grep '^RESULT' | tail -1)"
        if [ "$rc" -eq 124 ]; then
            echo "$m  TIMEOUT (>900s)" | tee -a "$PREFLIGHT_LOG"
        elif [ -n "$line" ]; then
            echo "$m  rc=$rc  ${line#RESULT }" | tee -a "$PREFLIGHT_LOG"
        else
            echo "$m  rc=$rc  no-RESULT (see preflight_${m}.err)" | tee -a "$PREFLIGHT_LOG"
        fi
    done
    log "=== PREFLIGHT done (informational; all models will still be trained) ==="
else
    log "=== PREFLIGHT skipped (DO_PREFLIGHT=0) ==="
fi

# ---------------------------------------------------------------- Phase 1
[ -f "$STATUS" ] || echo "model | start | end | exit | result | best_val_dice" > "$STATUS"
for m in "${MODELS[@]}"; do
    cfg="configs/models/$m.yaml"
    out_dir="$REPO/runs/${m}_run"
    if [ -f "$out_dir/.done" ]; then
        log "=== SKIP $m (already completed: $out_dir/.done) ==="
        echo "$m | - | - | 0 | DONE(skip) | -" >> "$STATUS"
        continue
    fi
    mkdir -p "$out_dir"
    echo "$out_dir/run.log" > "$ACTIVE_PTR"
    start="$(date +%T)"
    log ">>> TRAIN START  $m  ->  $out_dir  (--resume auto-detect)"
    # --resume: continue from the latest checkpoint if one exists, else fresh.
    # Output flows to the tmux pane for live viewing AND is captured to
    # console.log (stdout+stderr combined). run.log only holds the clean INFO
    # logging, so without this any crash traceback — which Python prints to
    # stderr — is lost. That gap is exactly why earlier step-0 failures were
    # undiagnosable. PIPESTATUS[0] keeps the python rc, not tee's.
    "$PY" -m scripts.train --config "$cfg" --output "$out_dir" --resume \
        2>&1 | tee -a "$out_dir/console.log"
    rc=${PIPESTATUS[0]}
    end="$(date +%T)"
    bd="$(grep -oE 'val_mean_dice=[0-9.]+' "$out_dir/run.log" 2>/dev/null \
          | grep -oE '[0-9.]+$' | sort -g | tail -1)"
    [ -z "$bd" ] && bd="n/a"
    if [ "$rc" -eq 0 ]; then res="OK"; touch "$out_dir/.done"; else res="FAIL"; fi
    echo "$m | $start | $end | $rc | $res | $bd" | tee -a "$STATUS"
    log "<<< TRAIN END    $m  rc=$rc  result=$res  best_val_dice=$bd"
done

echo "(batch idle)" > "$ACTIVE_PTR"
log "=== BATCH COMPLETE ==="
cat "$STATUS" | tee -a "$ORCH_LOG"
touch "$BATCH_DIR/DONE"
