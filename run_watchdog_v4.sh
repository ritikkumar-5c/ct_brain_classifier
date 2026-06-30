#!/usr/bin/env bash
# =====================================================================
# config_v4 — FRESH (no warm-start) ct_brain_classifier on the NEW large dataset
# =====================================================================
# Fresh fine-tune from ImageNet-pretrained MaxViT (NO --resume on first launch),
# keeping the proven v3 recipe and changing only what fresh-start + 4x data +
# 6% normal imbalance require. Writes to a separate out_dir.
#
# DATA (new, disk_vdc): 53,847 train / 8,975 val series (3-class), normal only
#   ~6% of train (3,276 / 28,551 / 22,020). Separate held-out TEST set (test_csv="").
#
# v4 changes vs v3 (flag -> reason):
#   init                warm-start -> FRESH (omit --resume; pretrained=True default)
#   train/val_csv       old splits -> NEW splits (disk_vdc); test_csv "" (separate test)
#   lr                  1.5e-4 -> 2.5e-4   fresh fine-tune needs a higher peak
#   warmup_epochs       1 -> 2             random head + fresh optimizer need a ramp
#   epochs              18 -> 12           4x steps/epoch -> converges in fewer epochs
#   early_stop_patience 5 -> 4             ~6 h/epoch; don't burn compute past plateau
#   balanced_sampler    (n/a) -> true      WeightedRandomSampler: normal exposure
#                                          6.1% -> 15.3% (oversample x2.5), partial balance
#   sampler_alpha       (n/a) -> 0.5       sqrt (mild) — avoids replicating the 3,276
#                                          normal studies x5.4 (overfit risk at alpha=1)
#   use_class_weights   true -> FALSE      MUST be off WITH the sampler: stacking inverse-
#                                          freq loss weights (normal 5.48x) on top of the
#                                          sampler = 58% grad emphasis on normal -> model
#                                          OVER-predicts normal (misses pathology). Sampler
#                                          gives balance; cost_sensitive gives clinical
#                                          asymmetry. (To use BOTH instead, set this true
#                                          AND drop sampler_alpha toward ~0.2.)
#   slice_chunk         96 -> 48           CAP GPU MEMORY: backbone forward is chunked to
#                                          48 slices (train: recomputed via grad_ckpt;
#                                          val: under no_grad) so peak activation is
#                                          bounded by 48, NOT by K=48(train)/96(val).
#                                          ~halves the v3 (chunk=96) activation peak; BN
#                                          stats over 48 slices (capacity unaffected).
# Unchanged from v3 (the part that earned bal_acc 0.688): loss=cost_sensitive 3.0/2.0,
#   cost_ce_lambda 0.4, use_class_weights, monitor=balanced_acc, target_sensitivity 0.95,
#   drop_path 0.15, dropout 0.1, label_smoothing 0.1, weight_decay 1e-3, window_jitter 0.1,
#   train_slices_per_study 48 (eval 96), batch_size 16, grad_checkpoint, use_amp.
#
# Launch (detached, survives the shell):
#   setsid nohup bash run_watchdog_v4.sh </dev/null >/dev/null 2>&1 &
# Kill the v4 training (NOT `pkill -f train_main`, which hits this shell):
#   ps -eo pid,args | awk '/ct_brain\/bin\/python train_main/ && /clinical_v4/ && !/awk/{print $1}' | xargs -r kill
# =====================================================================
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class_clinical_v4
NEW=/root/ritikkumar/disk_vdc/train_data/csvs/splits
TRAIN_LOG=$OUT.log
WLOG=$OUT.watchdog.log
PIDFILE=$OUT.watchdog.pid
MAX_RETRIES=5
BACKOFF=30

# reduce allocator fragmentation -> lower reserved (not just allocated) GPU memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# Adopt phase: if a v4 training is already running, wait for it instead of
# launching a duplicate on the same out_dir.
existing=$(find_train_pid || true)
if [ -n "${existing:-}" ]; then
  log "adopting already-running training pid=$existing; waiting (no duplicate launch)"
  while kill -0 "$existing" 2>/dev/null; do sleep 30; done
  log "adopted training pid=$existing exited"
  if tail -n 60 "$TRAIN_LOG" | grep -q "Best val F1:"; then
    log "clean completion marker found — training complete. stopping watchdog."
    exit 0
  fi
  log "no completion marker — treating as crash; will resume from last.pt"
fi

attempt=0
while :; do
  attempt=$((attempt + 1))
  # First launch -> FRESH (no resume). Crash-restart -> full resume from last.pt.
  RESUME_ARGS=()
  RMODE="fresh(pretrained, no resume)"
  if [ -f "$OUT/last.pt" ]; then
    RESUME_ARGS=(--resume "$OUT/last.pt"); RMODE="full-resume(last.pt)"
  fi
  log "launch attempt $attempt [$RMODE] -> $TRAIN_LOG"
  "$PY" train_main.py \
      --out_dir "$OUT" \
      --train_csv "$NEW/train.csv" --val_csv "$NEW/val.csv" --test_csv "" \
      --loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95 \
      --cost_miss_abnormal 3.0 --cost_miss_near_normal 2.0 --cost_ce_lambda 0.4 \
      --use_class_weights false --balanced_sampler true --sampler_alpha 0.5 \
      --batch_size 16 --num_workers 4 \
      --lr 2.5e-4 --epochs 12 --warmup_epochs 2 --weight_decay 1e-3 \
      --dropout 0.1 --drop_path 0.15 --train_slices_per_study 48 --slice_chunk 48 \
      --label_smoothing 0.1 --window_jitter 0.1 --early_stop_patience 4 \
      "${RESUME_ARGS[@]}" \
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
