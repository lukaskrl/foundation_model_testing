#!/usr/bin/env bash
# Re-evaluate the 3 ViT models whose best.pt was trained with the `spm` adapter
# but whose configs/models/<m>.yaml was switched to `pyramid_mode: upsample` by
# commit 13da93d (Jun 29 "vit adapter setup") for an ablation -- so the strict
# load in scripts/evaluate.py fails against the current config. We eval each
# against a temp config that restores pyramid_mode: spm (matching the saved
# checkpoint), written to /tmp/<m>_eval_spm.yaml. RAM-safe like eval_hd95_safe.sh
# (sequential, MALLOC_ARENA_MAX=1, host-RAM watchdog).
#
# Usage: eval_hd95_vit_spm.sh [SPLIT] [GPU] [MIN_AVAIL_MB]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/../env/bin/python"
cd "$REPO" || exit 1
SPLIT="${1:-val}"; GPU="${2:-1}"; MIN_AVAIL_MB="${3:-40000}"
MODELS=(ctclip dino3d sam_med3d)
SUMMARY="$REPO/runs/HD95_EVAL_SUMMARY.log"   # append to the same table
log(){ echo "[$(date +%F\ %T)] $*" | tee -a "$SUMMARY"; }

log "ViT spm re-eval start: split=$SPLIT gpu=$GPU min_avail=${MIN_AVAIL_MB}MB"
for m in "${MODELS[@]}"; do
  cfg="/tmp/${m}_eval_spm.yaml"
  run="$REPO/runs/${m}_run"; ckpt="$run/best.pt"
  out="$run/eval_${SPLIT}_metrics.json"; elog="$run/eval_${SPLIT}.log"
  if [ ! -f "$cfg" ]; then
    printf '%-20s %-10s %-10s %-12s %s\n' "$m" - - - "NO_SPM_CFG" | tee -a "$SUMMARY"; continue
  fi
  MALLOC_ARENA_MAX=1 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scripts.evaluate \
      --config "$cfg" --checkpoint "$ckpt" --split "$SPLIT" --out "$out" > "$elog" 2>&1 &
  pid=$!; maxused=0; aborted=0
  while kill -0 "$pid" 2>/dev/null; do
    read used avail < <(free -m | awk '/^Mem:/{print $3, $7}')
    [ "$used" -gt "$maxused" ] && maxused=$used
    if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
      log "WATCHDOG: avail=${avail}MB < ${MIN_AVAIL_MB}MB during $m -- killing + aborting"
      kill -TERM "$pid" 2>/dev/null; sleep 5
      pkill -KILL -f "scripts.evaluate --config /tmp/${m}_eval_spm.yaml" 2>/dev/null
      aborted=1; break
    fi
    sleep 6
  done
  wait "$pid" 2>/dev/null; rc=$?
  if [ "$aborted" = 1 ]; then
    printf '%-20s %-10s %-10s %-12s %s\n' "$m(spm)" - - "$maxused" "ABORTED_RAM" | tee -a "$SUMMARY"; break
  fi
  md=$("$PY" -c "import json;d=json.load(open('$out'));print(f\"{d.get('mean_dice',float('nan')):.4f}\")" 2>/dev/null || echo "?")
  mh=$("$PY" -c "import json;d=json.load(open('$out'));print(f\"{d.get('mean_hd95',float('nan')):.4f}\")" 2>/dev/null || echo "?")
  st="OK"; [ "$rc" -ne 0 ] && st="FAIL_rc$rc"
  printf '%-20s %-10s %-10s %-12s %s\n' "$m(spm)" "$md" "$mh" "$maxused" "$st" | tee -a "$SUMMARY"
done
log "ViT spm re-eval finished."
echo "(vit-eval-done)" >> "$SUMMARY"
