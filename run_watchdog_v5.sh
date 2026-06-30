#!/usr/bin/env bash
# =====================================================================
# config_v5 — rule-out MULTI-TASK head + top-k pooling, on the OLD (v3) data
# =====================================================================
# Goal: grow the high-confidence-normal TAIL for AUTO-REPORTING normal studies
# (maximize specificity at very high sensitivity / low false-clear rate). v3 hit a
# ~0.85 3-class AUC wall and its auto-clear tail was tiny + miscalibrated, so v5
# adds a SECOND head purpose-built for the tail and trains it with a partial-AUC
# objective, while KEEPING the proven v3 3-class head for triage (near/abnormal).
#
# DATA: the SAME old splits v3 trained on (patient-grouped 70/15/15, ~20% normal):
#   /root/ritikkumar/train_data/csvs/splits/{train,val,test}.csv
#   train 13,114 / val 2,623 / test 2,623 series. Held-out TEST evaluated at the end.
#
# v5 changes vs v3 (flag -> reason):
#   multitask_ruleout   false -> TRUE     add binary normal-vs-not-normal head (top-k pool)
#   ruleout_topk        (n/a) -> 8        mean of the 8 most-pathological slices = bag score;
#                                         top-k pooling does NOT dilute a 1-2 slice finding
#                                         (the v3 attention-MEAN head did) -> subtle pathology
#                                         can't masquerade as a confident normal
#   ruleout_weight      (n/a) -> 0.5      weight of the rule-out loss vs the 3-class loss
#   ruleout_pauc_lambda (n/a) -> 1.0      partial-AUC rank term: pushes the HARDEST positives
#                                         (low-scoring pathology) above the HARDEST negatives
#                                         (high-scoring normals = would-be false clears) ->
#                                         directly raises specificity at the target sensitivity
#   ruleout_pos_frac    (n/a) -> 0.5      focus the pAUC term on the harder half of pathology
#   monitor             balanced_acc -> ruleout_pauc   SELECT the checkpoint that best grows
#                                         the tail (mean specificity over the high-sens region),
#                                         not the symmetric balanced_acc
#   init                warm-start -> FRESH (ImageNet)  the new head must co-adapt from start;
#                                         avoids a strict-load mismatch on v3 ckpts
#   lr                  1.5e-4 -> 2.5e-4  fresh fine-tune needs a higher peak (matches v4)
#   warmup_epochs       1 -> 2            random heads + fresh optimizer need a ramp
#   epochs              18 -> 20          fresh start; small dataset (~1.3 h/epoch at K48)
#   early_stop_patience 5 -> 5            (kept) stop once tail metric plateaus
#   slice_chunk         96 -> 48          CAP GPU MEM (~24 GB): backbone forward chunked to 48
#                                         (train recomputed via grad_ckpt, val under no_grad);
#                                         capacity unaffected (MIL/top-k pool over ALL slices)
# Unchanged from v3 (the part that earned bal_acc 0.688): loss=cost_sensitive 3.0/2.0,
#   cost_ce_lambda 0.4, target_sensitivity 0.95, drop_path 0.15, dropout 0.1,
#   label_smoothing 0.1, weight_decay 1e-3, window_jitter 0.1, train_slices_per_study 48
#   (eval 96), batch_size 16, num_workers 4, grad_checkpoint, use_amp, length_bucketing.
#
# Outputs (at the end, written to $OUT): best.pt/last.pt + series_probs_{val,test}.csv
#   with a `p_ruleout` column -> feed straight into the split-conformal / auto-clear analysis.
#
# Launch (detached, survives the shell):
#   setsid nohup bash run_watchdog_v5.sh </dev/null >/dev/null 2>&1 &
# Kill the v5 training (NOT `pkill -f train_main`, which hits this shell):
#   ps -eo pid,args | awk '/ct_brain\/bin\/python train_main/ && /clinical_v5/ && !/awk/{print $1}' | xargs -r kill
# =====================================================================
set -u

CD=/root/ritikkumar/ct_brain_classifier
PY=/root/ritikkumar/ct_brain/bin/python
OUT=$CD/runs/maxvit384_3class_clinical_v5
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

# Adopt phase: if a v5 training is already running, wait for it instead of
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
      --loss cost_sensitive --monitor ruleout_pauc --target_sensitivity 0.95 \
      --cost_miss_abnormal 3.0 --cost_miss_near_normal 2.0 --cost_ce_lambda 0.4 \
      --multitask_ruleout true --ruleout_weight 0.5 --ruleout_topk 8 \
      --ruleout_pauc_lambda 1.0 --ruleout_pos_frac 0.5 --ruleout_neg_frac 1.0 \
      --ruleout_margin 1.0 --ruleout_bce_pos_weight 1.0 \
      --batch_size 16 --num_workers 4 \
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
