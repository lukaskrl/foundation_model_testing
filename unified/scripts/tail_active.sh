#!/usr/bin/env bash
# Live viewer for the back-to-back training batch. Follows whichever model's
# run.log is currently active (the orchestrator writes the active path to
# runs/ACTIVE_LOG) and switches automatically when the batch advances to the
# next model. Read-only; safe to start/stop/reattach at any time.
#
# Usage: tail_active.sh [PTR_FILE]   (default runs/ACTIVE_LOG)
REPO="/store/home/skrljl/projects/foundation_models/unified"
PTR="${1:-$REPO/runs/ACTIVE_LOG}"

echo "viewer: following active training log via $PTR"
echo "(Ctrl-b d to detach this tmux view; training keeps running)"
while true; do
    log="$(cat "$PTR" 2>/dev/null)"
    if [ -n "$log" ] && [ -f "$log" ]; then
        echo
        echo "================ tailing $log ================"
        tail -n 30 -F "$log" &
        tpid=$!
        # Follow until the orchestrator points ACTIVE_LOG somewhere else.
        while [ "$(cat "$PTR" 2>/dev/null)" = "$log" ]; do sleep 5; done
        kill "$tpid" 2>/dev/null
    else
        sleep 3
    fi
done
