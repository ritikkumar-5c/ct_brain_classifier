#!/usr/bin/env bash
# =====================================================================
# config_v3 — anti-overfit WARM-START of ct_brain_classifier (MaxViT-MIL, 3-class)
# =====================================================================
# Warm-starts from v2's best checkpoint (epoch 4, val balanced_acc=0.6469),
# weights only, with a fresh optimizer/scheduler so the NEW hyperparameters
# take effect. Writes to a SEPARATE out_dir so v2 is preserved/comparable.
#
# v2 diagnosis (epochs 0-6):
#   - Overfit onset at ep3-4: val_loss min @ep3 (0.870) then rises every epoch
#     (0.907->0.913->0.925) while train_loss keeps falling (0.90->0.79).
#   - train-val balanced_acc gap blows out to +0.08 by ep6.
#   - LR stuck near peak (1.96e-4 @ep6): cosine T_max=epochs=50 barely decays,
#     so it trains at full LR straight through the overfit zone.
#   - Per-class recall still oscillates (eased cost matrix helped, didn't fix).
#   - AUC capped ~0.82-0.84 (== v1): label-boundary ceiling, not optimization.
#
# config_v3 changes (flag -> reason), warm-started from v2 best ep4 weights:
#   drop_path             0.0 -> 0.15  STOCHASTIC DEPTH — the missing MaxViT
#                                      regularizer (50 DropPath layers, 0.015->0.15)
#   train_slices_per_study 96 -> 48    MIL slice-dropout: random 48/study each epoch
#                                      = strong bag-level regularizer + ~2x faster
#                                      (eval/val still uses max_slices_per_study=96)
#   lr                    2e-4 -> 1.5e-4  lower peak
#   epochs                50  -> 18    so cosine ACTUALLY decays to ~0 over the run
#   warmup_epochs         3   -> 1     warm weights don't need a long ramp
#   weight_decay          5e-4 -> 1e-3 stronger L2 to close the train/val gap
#   label_smoothing       0.05 -> 0.1  softer targets on the noisy normal/near boundary
#   early_stop_patience   12  -> 5     v2 best came at ep4; don't burn compute overfitting
# Unchanged from v2: loss=cost_sensitive, cost 3.0/2.0, cost_ce_lambda 0.4,
#   monitor=balanced_acc, target_sensitivity=0.95, dropout=0.1, window_jitter=0.1,
#   batch_size=16, grad_checkpoint=true, use_amp=true, image_size=384.
#
# Launch (detached, survives the shell):
#   setsid nohup bash run_watchdog_v3.sh </dev/null >/dev/null 2>&1 &
# Kill the v3 training (NOT `pkill -f train_main`, which hits this shell):
#   ps -eo pid,args | awk '/ct_brain\/bin\/python train_main/ && /clinical_v3/ && !/awk/{print $1}' | xargs -r kill
# =====================================================================
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class_clinical_v3
WARMSTART=$CD/runs/_warmstart_v2best_ep4_weights.pt   # weights-only -> fresh optim/sched, start_epoch=0
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

# Adopt phase: if a v3 training is already running, wait for it instead of
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
  # First launch -> warm-start (weights only). Crash-restart -> full resume from last.pt.
  if [ -f "$OUT/last.pt" ]; then
    RESUME="$OUT/last.pt"; RMODE="full-resume(last.pt)"
  else
    RESUME="$WARMSTART"; RMODE="warm-start(v2 best ep4 weights)"
  fi
  log "launch attempt $attempt [$RMODE] -> $TRAIN_LOG"
  "$PY" train_main.py \
      --out_dir "$OUT" \
      --loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95 \
      --cost_miss_abnormal 3.0 --cost_miss_near_normal 2.0 --cost_ce_lambda 0.4 \
      --lr 1.5e-4 --epochs 18 --warmup_epochs 1 --weight_decay 1e-3 \
      --dropout 0.1 --drop_path 0.15 --train_slices_per_study 48 \
      --label_smoothing 0.1 --window_jitter 0.1 --early_stop_patience 5 \
      --resume "$RESUME" \
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
