#!/usr/bin/env bash
# =====================================================================
# config_v2 — FOLLOW-UP run for ct_brain_classifier (MaxViT-MIL, 3-class)
# =====================================================================
# Launch ONLY if v1 (runs/maxvit384_3class_clinical) does not improve past
# its ~0.66 val balanced_acc plateau. Writes to a SEPARATE out_dir so v1 is
# preserved and the two are directly comparable.
#
# v1 diagnosis (epochs 0-13):
#   - LR schedule is healthy (warmup->3e-4->cosine). No bug.
#   - AUC climbs monotonically 0.79->0.85: model genuinely learning.
#   - BUT val balanced_acc plateaus ~0.66 from ep8 while train climbs to 0.71
#     => mild early overfitting.
#   - recall_normal oscillates violently (0.04..0.66): the cost_sensitive loss
#     penalizes calling pathology "normal" so hard that the NORMAL class is
#     sacrificed, capping balanced_acc and destabilizing per-class recall.
#
# config_v2 changes (flag -> reason), all else identical to v1:
#   cost_miss_abnormal   5.0 -> 3.0   ease asymmetry crushing normal recall (#1 lever)
#   cost_miss_near_normal 3.0 -> 2.0  same, for the near_normal->normal cost
#   cost_ce_lambda       0.3 -> 0.4   stronger CE blend => damp per-class oscillation
#   lr                   3e-4 -> 2e-4 lower peak => fewer early collapse episodes
#   warmup_epochs        2   -> 3     smoother ramp into the lower peak
#   weight_decay         1e-4 -> 5e-4 close the train/val gap (mild overfit)
#   dropout              0.04 -> 0.1  regularize the gated-attention MIL head
#   window_jitter        0.05 -> 0.1  slightly stronger CT-correct intensity aug
#   early_stop_patience  10  -> 12    give the smoother schedule room to converge
# Unchanged: loss=cost_sensitive, monitor=balanced_acc, target_sensitivity=0.95,
#   epochs=50, batch_size=16, grad_checkpoint=true, use_amp=true, image_size=384.
#
# Launch (detached, survives the shell):
#   setsid nohup bash run_watchdog_v2.sh </dev/null >/dev/null 2>&1 &
# Kill the v2 training (NOT `pkill -f train_main`, which hits this shell):
#   ps -eo pid,args | awk '/ct_brain\/bin\/python train_main/ && /clinical_v2/ && !/awk/{print $1}' | xargs -r kill
# =====================================================================
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class_clinical_v2
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

# Adopt phase: if a v2 training is already running, wait for it instead of
# launching a duplicate on the same out_dir.
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
      --cost_miss_abnormal 3.0 --cost_miss_near_normal 2.0 --cost_ce_lambda 0.4 \
      --lr 2e-4 --warmup_epochs 3 --weight_decay 5e-4 \
      --dropout 0.1 --window_jitter 0.1 --early_stop_patience 12 \
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
