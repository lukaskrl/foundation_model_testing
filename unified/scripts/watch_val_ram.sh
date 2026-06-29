#!/usr/bin/env bash
# Watch host RAM through the NEXT validation of a training run, to confirm the
# HD95 dice-only fix keeps the validation memory spike bounded.
#
# Exits (re-invoking the caller) when ONE of:
#   - a `epoch N val_mean_dice=` line with N >= TARGET_EP appears  -> VAL_OK
#   - no `scripts.train` process is alive                          -> NO_PROC
#   - CAP_MIN minutes elapse                                       -> CAP
#
# Usage: watch_val_ram.sh <RUN_DIR> <TARGET_EP> [CAP_MIN]
RUN="${1:?run dir}"; TARGET_EP="${2:?target epoch}"; CAP_MIN="${3:-90}"
LOG="$RUN/val_ram_watch.log"
: > "$LOG"
start=$(date +%s); maxused=0; maxused_t=""
echo "watch start $(date +%T)  run=$RUN  target_ep>=$TARGET_EP  cap=${CAP_MIN}m" >> "$LOG"
while :; do
  now=$(date +%s); el=$(( (now - start) / 60 ))
  read used avail < <(free -m | awk '/^Mem:/{print $3, $7}')
  if [ "$used" -gt "$maxused" ]; then maxused=$used; maxused_t=$(date +%T); fi
  g1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  ep=$(grep -oE 'epoch [0-9]+ (step [0-9]+/[0-9]+|done)|\[val\]' "$RUN/run.log" 2>/dev/null | tail -1)
  printf '%s el=%2dm used=%6dMB avail=%6dMB peak_used=%6dMB gpu1=%sMB | %s\n' \
    "$(date +%T)" "$el" "$used" "$avail" "$maxused" "$g1" "$ep" >> "$LOG"
  # success: a validation at or beyond TARGET_EP has logged its mean dice
  reached=$(grep -oE 'epoch [0-9]+ val_mean_dice=' "$RUN/run.log" 2>/dev/null \
            | grep -oE 'epoch [0-9]+' | awk -v t="$TARGET_EP" '$2>=t{print $2}' | tail -1)
  if [ -n "$reached" ]; then
    echo "VAL_OK  validation ep${reached} completed at $(date +%T)" >> "$LOG"
    echo "        peak host used during watch = ${maxused}MB at ${maxused_t}" >> "$LOG"
    break
  fi
  if ! pgrep -f "scripts.train" >/dev/null 2>&1; then
    echo "NO_PROC  no scripts.train alive at $(date +%T); peak host used=${maxused}MB" >> "$LOG"
    break
  fi
  if [ "$el" -ge "$CAP_MIN" ]; then
    echo "CAP  reached ${CAP_MIN}m at $(date +%T); peak host used=${maxused}MB" >> "$LOG"
    break
  fi
  sleep 10
done
echo "--- last 4 run.log lines ---" >> "$LOG"
tail -n 4 "$RUN/run.log" >> "$LOG" 2>&1
echo "(watch done)" >> "$LOG"
