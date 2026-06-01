#!/usr/bin/env bash
# Sequentially trains every non-stunet model to 200 epochs on the GPU
# selected by CUDA_VISIBLE_DEVICES. Each model writes to runs/<name>_e200/.
# Continues to the next model even if one fails, so a single backbone
# bug doesn't kill the whole sweep.
set -u

REPO="/home/skrljl/projects/foundation_models/unified"
PY="/store/home/skrljl/projects/foundation_models/env/bin/python"
MODELS=(voco_b voco_h vista3d dino3d biomedparse ctclip)
SWEEP_LOG="$REPO/runs/sweep_e200.log"

cd "$REPO"
mkdir -p runs
: > "$SWEEP_LOG"

# Limit glibc malloc arenas to reduce RSS bloat in long-running persistent
# dataloader workers (prior runs OOMed at ~170 GB RSS after ~5 epochs).
export MALLOC_ARENA_MAX=2

# Reduce CUDA fragmentation under shared-GPU pressure — voco_h previously
# OOMed at backward pass when an external process held 16 GB on GPU 1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Don't preserve sweep log across restarts since we resume past done-markers.
: > "$SWEEP_LOG.tmp" && mv "$SWEEP_LOG.tmp" "$SWEEP_LOG" || true

echo "[sweep] $(date -Iseconds) start, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$SWEEP_LOG"
echo "[sweep] models: ${MODELS[*]}" | tee -a "$SWEEP_LOG"

for m in "${MODELS[@]}"; do
  outdir="runs/${m}_e200"
  cfg="configs/models/${m}.yaml"
  mkdir -p "$outdir"

  if [[ -f "$outdir/done" ]]; then
    echo "[sweep] $(date -Iseconds) SKIP $m — done marker present" | tee -a "$SWEEP_LOG"
    continue
  fi

  echo "[sweep] $(date -Iseconds) START $m → $outdir" | tee -a "$SWEEP_LOG"
  t0=$(date +%s)
  "$PY" -u -m scripts.train --config "$cfg" --output "$outdir" \
       >> "$outdir/stdout.log" 2>&1
  rc=$?
  t1=$(date +%s)
  dt=$((t1 - t0))

  if [[ $rc -eq 0 ]]; then
    touch "$outdir/done"
    echo "[sweep] $(date -Iseconds) OK   $m rc=0 dt=${dt}s" | tee -a "$SWEEP_LOG"
  else
    echo "FAIL rc=$rc dt=${dt}s" > "$outdir/failed"
    echo "[sweep] $(date -Iseconds) FAIL $m rc=$rc dt=${dt}s — continuing" | tee -a "$SWEEP_LOG"
  fi
done

echo "[sweep] $(date -Iseconds) sweep finished" | tee -a "$SWEEP_LOG"
