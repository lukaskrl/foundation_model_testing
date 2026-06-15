#!/usr/bin/env bash
# Monitor a training batch and EXIT (so the parent agent is re-invoked) when:
#   - the batch finishes (a DONE marker appears), or
#   - a NEW training failure is recorded beyond the given baseline count.
# Read-only: it only polls files, never touches the GPU.
#
# Usage: watch_train_batch.sh <BATCH_DIR> [BASELINE_FAIL_COUNT]
BATCH_DIR="$1"
BASELINE="${2:-0}"
STATUS="$BATCH_DIR/STATUS.log"

if [ -z "$BATCH_DIR" ]; then
    echo "usage: watch_train_batch.sh <BATCH_DIR> [baseline_fails]"; exit 2
fi

while true; do
    if [ -f "$BATCH_DIR/DONE" ]; then
        echo "=== BATCH COMPLETE ==="
        [ -f "$STATUS" ] && cat "$STATUS"
        exit 0
    fi
    n="$(grep -c '| FAIL |' "$STATUS" 2>/dev/null)"
    n="${n:-0}"
    if [ "$n" -gt "$BASELINE" ]; then
        echo "=== NEW TRAINING FAILURE (fails=$n, baseline=$BASELINE) ==="
        [ -f "$STATUS" ] && cat "$STATUS"
        # Show the tail of the most recently modified run.log for context.
        last_log="$(ls -t "$BATCH_DIR"/../*_run/run.log 2>/dev/null | head -1)"
        if [ -n "$last_log" ]; then
            echo "--- tail $last_log ---"
            tail -n 25 "$last_log"
        fi
        exit 1
    fi
    sleep 120
done
