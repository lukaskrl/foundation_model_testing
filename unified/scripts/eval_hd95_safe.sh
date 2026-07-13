#!/usr/bin/env bash
# RAM-safe final HD95 (+dice) evaluation over every model's best.pt.
#
# scripts/evaluate.py builds Evaluator with the FULL eval.metrics (dice+hd95),
# i.e. the dense-one-hot HD95 path. With the per-case malloc_trim fix in
# evaluator.py + MALLOC_ARENA_MAX=1 the peak should stay at the per-case working
# set, but this is a SHARED node, so a host-RAM watchdog kills the running eval
# and ABORTS the loop if available RAM nears exhaustion -- never let it OOM the
# box. Each model runs one at a time (sequential).
#
# Usage: eval_hd95_safe.sh [SPLIT] [GPU] [MIN_AVAIL_MB]
set -u
REPO="/store/home/skrljl/projects/foundation_models/unified"
PY="$REPO/../env/bin/python"
cd "$REPO" || exit 1
SPLIT="${1:-val}"
GPU="${2:-1}"
MIN_AVAIL_MB="${3:-40000}"     # abort if host avail drops below this
MODELS=(suprem_unet suprem_segresnet suprem_swinunetr vista3d ctclip \
        voco_b voco_h dino3d sam_med3d biomedparse)
SUMMARY="$REPO/runs/HD95_EVAL_SUMMARY.log"
: > "$SUMMARY"
log(){ echo "[$(date +%F\ %T)] $*" | tee -a "$SUMMARY"; }

log "HD95 eval start: split=$SPLIT gpu=$GPU min_avail=${MIN_AVAIL_MB}MB malloc_arena_max=1"
printf '%-20s %-10s %-10s %-12s %s\n' MODEL MEAN_DICE MEAN_HD95 PEAK_USEDMB STATUS | tee -a "$SUMMARY"

for m in "${MODELS[@]}"; do
  run="$REPO/runs/${m}_run"
  ckpt="$run/best.pt"
  out="$run/eval_${SPLIT}_metrics.json"
  elog="$run/eval_${SPLIT}.log"
  if [ ! -f "$ckpt" ]; then
    printf '%-20s %-10s %-10s %-12s %s\n' "$m" - - - "NO_CKPT" | tee -a "$SUMMARY"; continue
  fi
  MALLOC_ARENA_MAX=1 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scripts.evaluate \
      --config "configs/models/$m.yaml" --checkpoint "$ckpt" \
      --split "$SPLIT" --out "$out" > "$elog" 2>&1 &
  pid=$!
  maxused=0; aborted=0
  while kill -0 "$pid" 2>/dev/null; do
    read used avail < <(free -m | awk '/^Mem:/{print $3, $7}')
    [ "$used" -gt "$maxused" ] && maxused=$used
    if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
      log "WATCHDOG: avail=${avail}MB < ${MIN_AVAIL_MB}MB during $m -- killing eval + aborting loop"
      kill -TERM "$pid" 2>/dev/null; sleep 5
      pkill -TERM -f "scripts.evaluate --config configs/models/$m" 2>/dev/null; sleep 3
      pkill -KILL -f "scripts.evaluate --config configs/models/$m" 2>/dev/null
      aborted=1; break
    fi
    sleep 6
  done
  wait "$pid" 2>/dev/null; rc=$?
  if [ "$aborted" = 1 ]; then
    printf '%-20s %-10s %-10s %-12s %s\n' "$m" - - "$maxused" "ABORTED_RAM" | tee -a "$SUMMARY"
    log "ABORTED at $m to protect the node. Remaining models NOT evaluated. Investigate before retrying."
    break
  fi
  md=$("$PY" -c "import json;d=json.load(open('$out'));print(f\"{d.get('mean_dice',float('nan')):.4f}\")" 2>/dev/null || echo "?")
  mh=$("$PY" -c "import json;d=json.load(open('$out'));print(f\"{d.get('mean_hd95',float('nan')):.4f}\")" 2>/dev/null || echo "?")
  st="OK"; [ "$rc" -ne 0 ] && st="FAIL_rc$rc"
  printf '%-20s %-10s %-10s %-12s %s\n' "$m" "$md" "$mh" "$maxused" "$st" | tee -a "$SUMMARY"
done

log "HD95 eval loop finished."
echo "(eval-done)" >> "$SUMMARY"
