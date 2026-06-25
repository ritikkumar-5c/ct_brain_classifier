#!/usr/bin/env bash
# Watchdog supervisor for ct_brain_classifier training.
# Restarts training on crash (bounded retries + backoff); exits when training
# finishes cleanly (exit 0, e.g. early stopping / max epochs). Detached &
# survives the launching shell when started with setsid+nohup.
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class
TRAIN_LOG=$OUT.log
WLOG=$OUT.watchdog.log
PIDFILE=$OUT.watchdog.pid
MAX_RETRIES=5
BACKOFF=30

mkdir -p "$OUT"
cd "$CD" || exit 1

# single-instance guard
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[watchdog] already running (pid $(cat "$PIDFILE")); abort." >> "$WLOG"
  exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log() { echo "[watchdog $(date '+%F %T')] $*" >> "$WLOG"; }

log "start (pid=$$), out_dir=$OUT, max_retries=$MAX_RETRIES"
attempt=0
while :; do
  attempt=$((attempt + 1))
  log "launch attempt $attempt -> $TRAIN_LOG"
  "$PY" train_main.py \
      --out_dir "$OUT" \
      --xai_enabled true --log_histograms true --use_amp true \
      >> "$TRAIN_LOG" 2>&1
  code=$?
  log "training exited code=$code"
  if [ "$code" -eq 0 ]; then
    log "clean exit — training complete. stopping watchdog."
    break
  fi
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    log "max retries ($MAX_RETRIES) reached — giving up."
    break
  fi
  log "crash; restarting in ${BACKOFF}s (attempt $((attempt + 1))/$MAX_RETRIES)"
  sleep "$BACKOFF"
done
log "watchdog finished."
