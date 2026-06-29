#!/usr/bin/env bash
# Autonomous multi-pass supervisor for the back-to-back trainer.
#
# Standing preference (see memory training-node-shared-oom): keep re-running
# run_train_batch.sh so OOM / VRAM-leak casualties (which leave NO .done) get
# resumed from their latest checkpoint on a FRESH orchestrator invocation (which
# also clears any leaked VRAM), repeating until every model is .done OR a full
# pass makes zero forward progress (no new .done AND no checkpoint advance) --
# the latter signalling a genuine, non-OOM error that needs inspection rather
# than blind retrying.
#
# Run fully detached (setsid/nohup) so it survives the session ending.
REPO="/store/home/skrljl/projects/foundation_models/unified"
cd "$REPO" || exit 1
LOG="$REPO/runs/SUPERVISOR.log"
MODELS=(suprem_unet suprem_segresnet suprem_swinunetr vista3d ctclip \
        voco_b voco_h dino3d sam_med3d biomedparse)
say(){ echo "[$(date +%F\ %T)] $*" | tee -a "$LOG"; }

progress_sig(){   # per-model: .done flag + highest checkpoint epoch
  local s="" m d e
  for m in "${MODELS[@]}"; do
    d=0; [ -f "runs/${m}_run/.done" ] && d=1
    e=$(ls runs/${m}_run/epoch_*.pt 2>/dev/null \
        | sed -E 's/.*epoch_0*([0-9]+)\.pt/\1/' | sort -n | tail -1); e=${e:-0}
    s+="${m}:${d}:${e} "
  done
  echo "$s"
}
done_count(){ local c=0 m; for m in "${MODELS[@]}"; do [ -f "runs/${m}_run/.done" ] && c=$((c+1)); done; echo "$c"; }

say "supervisor start (pid $$); models=${MODELS[*]}"

# Wait out any orchestrator already in flight (e.g. the pass the agent launched).
while pgrep -f "run_train_batch.sh" >/dev/null 2>&1; do
  say "waiting for in-flight orchestrator to finish (.done=$(done_count)/10)..."
  sleep 180
done

pass=1
while true; do
  dc=$(done_count)
  say "==== PASS $pass START  (.done=${dc}/10) ===="
  if [ "$dc" -ge 10 ]; then say "ALL 10 MODELS DONE -- supervisor exiting."; break; fi
  before=$(progress_sig); say "progress before: $before"

  DO_PREFLIGHT=0 WANDB_MODE=online bash scripts/run_train_batch.sh >>"$LOG" 2>&1

  after=$(progress_sig); dc2=$(done_count)
  say "==== PASS $pass END    (.done=${dc2}/10) ===="
  say "progress after : $after"
  if [ "$dc2" -ge 10 ]; then say "ALL 10 MODELS DONE -- supervisor exiting."; break; fi
  if [ "$before" = "$after" ]; then
    say "STALL: full pass made NO forward progress (no new .done, no checkpoint advance)."
    say "A genuine (non-OOM) error likely remains -- stopping for inspection. .done=${dc2}/10."
    break
  fi
  say "progress made; cooling down 60s to let VRAM settle before next pass."
  sleep 60
  pass=$((pass+1))
done
say "supervisor done."
