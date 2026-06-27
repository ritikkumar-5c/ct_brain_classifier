#!/usr/bin/env bash
# Watchdog supervisor for ct_brain_classifier training.
# Restarts training on crash (bounded retries + backoff); exits when training
# finishes cleanly (exit 0, e.g. early stopping / max epochs). Detached &
# survives the launching shell when started with setsid+nohup.
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class_clinical
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

# Return the MAIN (non-worker) train_main pid for this out_dir, if any.
# DataLoader workers are forked children of main and share its argv, so the
# main process is the matching pid whose parent is NOT itself a match.
find_train_pid() {
  local pids p ppid
  pids=$(ps -eo pid,args | awk -v od="$OUT" '/ct_brain\/bin\/python train_main/ && !/awk/ && $0 ~ od {print $1}')
  for p in $pids; do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    if ! echo "$pids" | tr ' ' '\n' | grep -qw "$ppid"; then echo "$p"; return 0; fi
  done
  return 1
}

log "start (pid=$$), out_dir=$OUT, max_retries=$MAX_RETRIES"

# Adopt phase: if a training is already running (e.g. the watchdog was killed
# but train_main survived as an orphan), DO NOT launch a duplicate on the same
# out_dir. Wait for it to exit, then decide: clean finish (test eval logged)
# => done; otherwise treat as a crash and fall through to the resume loop.
existing=$(find_train_pid || true)
if [ -n "${existing:-}" ]; then
  log "adopting already-running training pid=$existing; waiting (no duplicate launch)"
  while kill -0 "$existing" 2>/dev/null; do sleep 30; done
  log "adopted training pid=$existing exited"
  if tail -n 60 "$TRAIN_LOG" | grep -q "TEST (best ckpt):"; then
    log "clean completion marker found — training complete. stopping watchdog."
    exit 0
  fi
  log "no completion marker — treating as crash; will resume from last.pt"
fi

attempt=0
while :; do
  attempt=$((attempt + 1))
  log "launch attempt $attempt -> $TRAIN_LOG"
  "$PY" train_main.py \
      --out_dir "$OUT" \
      --loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95 \
      --resume "$OUT/last.pt" \
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
