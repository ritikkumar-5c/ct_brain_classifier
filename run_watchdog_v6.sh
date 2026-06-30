#!/usr/bin/env bash
# =====================================================================
# config_v6 — v5 recipe at 512px native resolution (resolution A/B)
# =====================================================================
# HYPOTHESIS: the rule-out tail fails on sparse 1-2 slice, few-mm findings
# (thin SDH, small hyperdense bleed, early infarct). Native CT brain is 512x512;
# v1-v5 resize to 384 and throw away ~44% of in-plane pixels. v6 changes ONLY the
# input resolution (384 -> 512, + the matching 512 MaxViT checkpoint) on top of the
# EXACT v5 recipe, so any delta in the certifiable auto-clear tail is attributable
# to resolution alone. Judge v6 on auto-clear volume at <=2% certified miss
# (p_ruleout -> split-conformal), NOT balanced_acc.
#
# DATA: identical to v5 — old splits, patient-grouped 70/15/15, ~20% normal:
#   /root/ritikkumar/train_data/csvs/splits/{train,val,test}.csv
#
# v6 changes vs v5 (flag -> reason):
#   timm_name   maxvit_tiny_tf_384.in1k -> maxvit_tiny_tf_512.in1k   MaxViT-tf has FIXED
#                                         window/grid partitioning, so 512 input REQUIRES the
#                                         512-pretrained checkpoint (can't feed 512 into the 384 model)
#   image_size  384 -> 512               use native CT resolution; the slice pipeline resizes to this
#   batch_size  16 -> 8                  CO-RESIDENCY: v6 shares the A100 with the running v5.
#                                         Halving the batch halves per-step activation AND the val
#                                         /dev/shm footprint (guards the prior exit-137 box-kill)
#   (slice_chunk 48, num_workers 4 KEPT at v5 values: bs8 already halves v6's per-step activation
#    AND its val /dev/shm batch vs v5, so chunk 48 still fits beside v5 in 80GB (~22-26GB) and
#    4 workers stay within shm. Capacity UNAFFECTED — MIL/top-k pool over all 48 train / 96 eval slices.)
# Everything else identical to v5: multitask_ruleout true, ruleout_weight 0.5, ruleout_topk 8,
#   ruleout_pauc_lambda 1.0, pos_frac 0.5, neg_frac 1.0, margin 1.0, bce_pos_weight 1.0,
#   loss=cost_sensitive 3.0/2.0, cost_ce_lambda 0.4, target_sensitivity 0.95, monitor=ruleout_pauc,
#   FRESH ImageNet init, lr 2.5e-4, epochs 20, warmup 2, wd 1e-3, dropout 0.1, drop_path 0.15,
#   label_smoothing 0.1, window_jitter 0.1, train_slices_per_study 48 (eval 96), grad_checkpoint, amp.
#
# NOTE on throughput: v6 at 512 is ~1.78x the per-slice FLOPs of v5, AND both jobs time-slice ONE
#   A100, so expect each to run at roughly half speed while co-resident. v6 epoch ~3-5h under contention.
#
# Outputs (at the end): best.pt/last.pt + series_probs_{val,test}.csv with a p_ruleout column.
#
# Launch (detached, survives the shell):
#   setsid nohup bash run_watchdog_v6.sh </dev/null >/dev/null 2>&1 &
# Kill the v6 training (NOT `pkill -f train_main`, which hits this shell):
#   ps -eo pid,args | awk '/ct_brain\/bin\/python train_main/ && /clinical_v6/ && !/awk/{print $1}' | xargs -r kill
# =====================================================================
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit512_3class_clinical_v6
OLD=/root/ritikkumar/train_data/csvs/splits
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

# Adopt phase: if a v6 training is already running, wait for it instead of
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
      --train_csv "$OLD/train.csv" --val_csv "$OLD/val.csv" --test_csv "$OLD/test.csv" \
      --timm_name maxvit_tiny_tf_512.in1k --image_size 512 \
      --loss cost_sensitive --monitor ruleout_pauc --target_sensitivity 0.95 \
      --cost_miss_abnormal 3.0 --cost_miss_near_normal 2.0 --cost_ce_lambda 0.4 \
      --multitask_ruleout true --ruleout_weight 0.5 --ruleout_topk 8 \
      --ruleout_pauc_lambda 1.0 --ruleout_pos_frac 0.5 --ruleout_neg_frac 1.0 \
      --ruleout_margin 1.0 --ruleout_bce_pos_weight 1.0 \
      --batch_size 8 --num_workers 4 \
      --lr 2.5e-4 --epochs 20 --warmup_epochs 2 --weight_decay 1e-3 \
      --dropout 0.1 --drop_path 0.15 --train_slices_per_study 48 --slice_chunk 48 \
      --label_smoothing 0.1 --window_jitter 0.1 --early_stop_patience 5 \
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
