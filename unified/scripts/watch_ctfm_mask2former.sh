#!/usr/bin/env bash
# Crash-forensics daemon for the ctfm_mask2former run. Detached & persistent.
#
# Every time a train.py invocation dies, appends a diagnostic snapshot to
# runs/ctfm_mask2former/crashes.log that distinguishes the failure mode:
#   - kernel OOM-killer  -> SIGKILL, NO python traceback in console.log, dmesg
#                           "Killed process ... (python)" (host-RAM exhaustion).
#   - CUDA OOM           -> "CUDA out of memory" in console.log.
#   - code exception     -> a python Traceback in console.log.
# Also records whether the launcher auto-resumed (transient) or the run died for
# good. Exits on terminal state: .done (ok) / launcher gone / ABORT / time cap.
#
# Launch detached so it survives this shell:
#   setsid nohup bash scripts/watch_ctfm_mask2former.sh >/dev/null 2>&1 &

REPO="/store/home/skrljl/projects/foundation_models/unified"
OUT="$REPO/runs/ctfm_mask2former"
CRASH="$OUT/crashes.log"
POLL="${POLL:-30}"
MAX_HOURS="${MAX_HOURS:-48}"

TRAIN_PAT="scripts.train --config configs/models/ctfm_mask2former.yaml"
LAUNCH_PAT="run_ctfm_mask2former.sh"

mkdir -p "$OUT"
log() { echo "[$(date +%F\ %T)] $*" >> "$CRASH"; }
train_pids() { pgrep -f "$TRAIN_PAT" 2>/dev/null; }
launcher_up() { pgrep -f "$LAUNCH_PAT" >/dev/null 2>&1; }

snapshot() {   # $1 = reason tag
  {
    echo "======== CRASH SNAPSHOT ($1)  $(date +%F\ %T) ========"
    echo "--- last epoch/step reached ---"
    grep -oE "epoch +[0-9]+/[0-9]+ \[train\]:[^,]*loss=[0-9.]+" "$OUT/console.log" 2>/dev/null | tail -1
    echo "--- console.log tail (CUDA-OOM? python traceback? or nothing=silent SIGKILL) ---"
    tail -45 "$OUT/console.log" 2>/dev/null
    echo "--- kernel OOM-killer (dmesg) ---"
    { dmesg -T 2>/dev/null || dmesg 2>/dev/null; } | grep -iE "killed process|out of memory|oom-kill" | tail -8 \
      || echo "(dmesg unreadable — likely dmesg_restrict; a SIGKILL with NO traceback above => kernel OOM)"
    echo "--- GPU 1 ---"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader -i 1 2>/dev/null
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head
    echo "--- host RAM (GB) ---"
    free -g 2>/dev/null | head -2
    echo "--- loop.log tail (launcher rc / relaunch decision) ---"
    tail -6 "$OUT/loop.log" 2>/dev/null
    echo "=================================================================="
    echo
  } >> "$CRASH"
}

log "watcher start (poll=${POLL}s, cap=${MAX_HOURS}h)"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
had_train=0

while :; do
  [ -f "$OUT/.done" ] && { log "TERMINAL: .done — run completed OK"; break; }
  if grep -q "ABORT" "$OUT/loop.log" 2>/dev/null; then
    snapshot "launcher-ABORT"; log "TERMINAL: launcher ABORT (3 fast crashes)"; break
  fi

  if [ -n "$(train_pids)" ]; then
    had_train=1
  elif [ "$had_train" = "1" ]; then
    # train was up and just vanished -> a death happened. Let the launcher react.
    sleep 20
    if [ -f "$OUT/.done" ]; then log "TERMINAL: .done after exit — completed OK"; break; fi
    if [ -n "$(train_pids)" ]; then
      snapshot "train-died-then-RELAUNCHED (transient, launcher recovered)"; had_train=1
    elif launcher_up; then
      snapshot "train-down-launcher-alive (mid-relaunch/backoff)"; had_train=0
    else
      snapshot "train-down-launcher-GONE (run is dead)"
      log "TERMINAL: launcher gone, no train, no .done — dead"; break
    fi
  else
    launcher_up || { [ -f "$OUT/.done" ] || { log "TERMINAL: launcher not running, never saw train"; break; }; }
  fi

  [ "$(date +%s)" -ge "$deadline" ] && { log "TERMINAL: ${MAX_HOURS}h cap"; break; }
  sleep "$POLL"
done
log "watcher exit"
