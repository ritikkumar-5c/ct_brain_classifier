# CT Brain Study Classifier

**MaxViT + Multiple-Instance Learning + Grad-CAM++** — a 3-class
(`normal` / `near_normal` / `abnormal`) classifier for CT brain **series sets**:
many DICOM slices per `studyID/seriesID`, **one label per set**.

Each slice is rendered into a **3-channel image, one clinical window per channel** —
brain (W80/L40), subdural (W200/L80), bone (W2800/L600). The architecture and
training recipe follow Qari & Thafar, *"Brain Stroke Detection and Classification
Using CT Imaging with Transformer Models and Explainable AI"*
([arXiv:2507.09630](https://arxiv.org/abs/2507.09630)), adapted from their
slice-level / PNG setup to a **study-level, multi-slice, DICOM** setup.

---

## At a glance

| | |
|---|---|
| **Task** | 3-class study classification — `normal` / `near_normal` / `abnormal` |
| **Input** | A `studyID/seriesID` folder of `*.dcm` slices → one label |
| **Backbone** | MaxViT-tiny @ 384px (`maxvit_tiny_tf_384.in1k`), `timm` |
| **Aggregation** | Gated-attention MIL pooling over slices (one decision per study) |
| **Channels** | 3 HU windows — brain / subdural / bone |
| **Loss / monitor** | `cost_sensitive` (run) · checkpoint on `balanced_acc` |
| **Decision** | 0.95 not-normal-sensitivity operating point on test |
| **Explainability** | Slice attention (*which slices*) + Grad-CAM++ (*where in them*) |

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Data layout
Series folders of `*.dcm` slices, **nested** as `<class>/<studyID>/<seriesID>/`:
```
train_data/
  normal/      00197F33/ 00198653/ *.dcm     # one study can hold multiple series
                        9A5727A1/ *.dcm
  near_normal/ ...
  abnormal/    ...
```
> The legacy flat layout `<studyID>_<seriesID>/*.dcm` is also supported — the
> study id is recovered from either layout for patient-grouped splitting.

### 3. Build & split the CSVs
Index every series folder into one labeled CSV, then split it patient-grouped +
class-stratified into train/val/test:
```bash
# (a) index every series folder (any depth) -> path,slice_size,label
python data/build_classifier_csv.py \
  --class normal=/root/ritikkumar/train_data/normal \
  --class near_normal=/root/ritikkumar/train_data/near_normal \
  --class abnormal=/root/ritikkumar/train_data/abnormal \
  --out /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv

# (b) 70/15/15 study-grouped split (no patient leaks across splits)
python data/split_classifier_csv.py \
  --csv /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv \
  --out-dir /root/ritikkumar/train_data/csvs/splits
```
This writes `splits/{train,val,test}.csv` — the `train_csv` / `val_csv` /
`test_csv` defaults in `config.py`.

### 4. Train
`config.py` already encodes the full recipe — split CSVs, MaxViT-384 @ 384×384,
≤96 slices/study, batch 16, `slice_chunk=96`, `grad_checkpoint=True`,
length bucketing, `num_workers=4`, `label_smoothing=0.05`, `window_jitter=0.05`
(ColorJitter off), checkpoint selection on **`balanced_acc`**, and a
0.95-sensitivity test operating point. The clinical run only overrides the loss
and the output directory:

```bash
python train_main.py \
  --out_dir runs/maxvit384_3class_clinical \
  --loss cost_sensitive \
  --monitor balanced_acc \
  --target_sensitivity 0.95 \
  --xai_enabled true --log_histograms true --use_amp true
```

Run it detached (≈2.3 h/epoch on the full set, up to 50 epochs with early stopping):
```bash
nohup python train_main.py \
  --out_dir runs/maxvit384_3class_clinical \
  --loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95 \
  --xai_enabled true --log_histograms true --use_amp true \
  > runs/maxvit384_3class_clinical.log 2>&1 &
```

**Crash-safe supervision.** `run_watchdog.sh` runs the same command with
`--resume runs/maxvit384_3class_clinical/last.pt` and auto-restarts on crash
(bounded retries + backoff). Resume restores the **full** state — model,
optimizer, scheduler, AMP scaler, epoch, best score, early-stop counter and RNG —
so a restart continues bit-for-bit:
```bash
setsid nohup bash run_watchdog.sh >/dev/null 2>&1 &
```

> **Alternatives.** Drop `--loss cost_sensitive` to use the default `weighted_ce`.
> Any `config.py` field is a CLI flag — swap `--backbone vit|tnt|convnext` for the
> paper's comparison models, or override any recipe knob directly.

### 5. Explain a single study
```bash
python infer.py --ckpt runs/maxvit384_3class_clinical/best.pt \
  --study_dir /root/ritikkumar/train_data/normal/00197F33/00198653 \
  --out_dir explanations/00198653 --topk 3
```
Prints the study-level prediction and saves Grad-CAM++ overlay PNGs for the most
attended slices.

> **Memory note.** Full fine-tune at batch 16 × 96 slices × 384px **requires
> `grad_checkpoint=True`** (default) — without it the ~1536 slice activations per
> step OOM even on an 80 GB A100. With it, peak ≈ **47.7 GB** (~20–30% slower).
> `slice_chunk` alone does *not* bound training memory in full-fine-tune mode. To
> trade the slowdown for memory differently, use a smaller real batch with
> `--batch_size 4 --grad_accum_steps 4`.

---

## How it works

This is a **Multiple-Instance Learning** problem: one label over a bag of slices.

1. **Per-slice encoding** — every slice is encoded by MaxViT-tiny @ 384px
   (`models/maxvit_mil.py`).
2. **Gated-attention pooling** (Ilse et al., 2018) aggregates slices into one
   study embedding → a single 3-class decision. The attention weights reveal
   *which slices* drove the decision.
3. **Grad-CAM++** (`xai/gradcampp.py`) then shows *where* within those slices,
   on the deep conv layer `stages.3.blocks.1.conv.conv2_kxk`.

**Two adaptations from the paper:**

- **DICOM input + multi-window channels** — raw DICOM pixels are converted via
  RescaleSlope/Intercept → Hounsfield Units, then three clinical windows (brain,
  subdural, bone) are stacked as the 3 channels, instead of replicating a single
  grayscale window.
- **Study-level label over many slices** — the MIL head above, instead of the
  paper's per-PNG slice-level classification.

<details>
<summary><b>What matches the paper</b> (component-by-component)</summary>

| Paper component | Here |
|---|---|
| **MaxViT** backbone (CNN MBConv + block + grid attention), from `timm` | `models/build.py` → `maxvit_tiny_tf_384.in1k` (384px). ViT/TNT/ConvNeXt selectable for comparison. |
| Preprocess: resize, grayscale→3ch, ImageNet norm (Sec 4.4) | `data/transforms.py` (resize 384×384; 3 channels = 3 HU windows) |
| Classical augmentation: crop, h-flip, rotation, color jitter (Sec 4.5.1) | `data/transforms.py` |
| Weighted loss for class imbalance (Sec 4.3) | `engine/losses.py` + `class_weights()` |
| Adam, LR∈{1e-3,3e-4,1e-5}, batch∈{16,32,64}, epochs∈{25,40,50,100}, dropout∈{0.03–0.05} (Table 3) | `config.py` defaults |
| Grouped, class-stratified split | `data/split_classifier_csv.py` (study-level, no patient leakage) |
| Metrics: accuracy, precision, recall, F1, AUC, confusion matrix (Sec 5.1/6.1) | `engine/metrics.py` (macro P/R/F1; macro-OVR AUC for 3-class) |
| **Grad-CAM++** on `stages.3.blocks.1.conv.conv2_kxk` (Sec 6.3) | `xai/gradcampp.py` (this exact layer is the default, verified to exist) |

</details>

---

## Dataset & splits

In-house CT-brain DICOM collection, indexed at the **series** level (one row =
one `studyID/seriesID` folder, one label per series). A study may contribute
several series, and every series of a study carries the same class.

**Full set** — `csvs/ct_brain_3class_slice_count.csv`:

| Class | Series | Studies | Slices | Series share |
|---|--:|--:|--:|--:|
| `normal` | 3,819 | 2,349 | 420,487 | 20.8% |
| `near_normal` | 6,596 | 3,909 | 793,371 | 35.9% |
| `abnormal` | 7,945 | 4,824 | 1,029,263 | 43.3% |
| **Total** | **18,360** | **11,082** | **2,243,121** | **100%** |

**Splits** — `csvs/splits/{train,val,test}.csv`, a **70/15/15 study-grouped,
class-stratified** split (`StratifiedGroupKFold`, `seed=42`). Splitting is by
study, so **all series of a study stay in one split — no patient leakage**
(verified disjoint: train∩val = train∩test = val∩test = ∅):

| Split | Series | Studies | Slices | normal | near_normal | abnormal |
|---|--:|--:|--:|--:|--:|--:|
| train | 13,114 | 7,916 | 1,597,442 | 2,727 | 4,712 | 5,675 |
| val | 2,623 | 1,583 | 327,753 | 546 | 942 | 1,135 |
| test | 2,623 | 1,583 | 317,926 | 546 | 942 | 1,135 |

The class imbalance (≈1 : 1.7 : 2.1) is handled by inverse-frequency class
weights and the cost-sensitive loss.

---

## Metrics, loss & decision logic

Built for *"don't miss `near_normal` / `abnormal`"* — three independent levers:

| Lever | Flag | Current run | What it does |
|---|---|---|---|
| **Loss** | `--loss` | `cost_sensitive` | Penalizes under-calling pathology to `normal` (cost: `abnormal→normal=5`, `near_normal→normal=3`). Also `weighted_ce` (default), `focal`. |
| **Selection** | `--monitor` | `balanced_acc` | Picks the best-discriminating checkpoint (mean per-class recall); not gameable by over-calling. |
| **Operating point** | `--target_sensitivity` | `0.95` | Tunes the decision threshold on val to a sensitivity floor, reported on test as `op_test_*`. |

Full detail (formulas, cost matrix, monitor options, per-epoch progress reads) is in
[`docs/metrics_losses_and_selection.md`](docs/metrics_losses_and_selection.md) and the
`docs/training_progress_analysis_epoch*.md` reports.

---

## TensorBoard
```bash
tensorboard --logdir runs
```
Logged: loss, accuracy, macro P/R/F1, balanced accuracy, per-class recall,
`not_normal_sensitivity`, `normal_specificity`, macro-OVR AUC, lr; plus confusion
matrix, Grad-CAM++ overlays, per-study attention, and the test operating-point
sens/spec.

---

## Notes / knobs

- **Fine-tune vs. freeze** — `freeze_backbone=True` reproduces the paper's
  "freeze all but classifier" setting; default `False` (full fine-tune) usually
  scores higher.
- **Variable bag size** — `slice_chunk>0` encodes slices in chunks so activation
  memory stays bounded regardless of bag size K. To feed *every* slice of every
  study: `all_slices=True, batch_size=1, slice_chunk>0, grad_checkpoint=True`.
  `length_bucketing` groups similar-length studies so the backbone wastes less
  compute on padding.
- **Class imbalance & clinical cost** — `use_class_weights=True` (default) applies
  inverse-frequency weights; `--loss cost_sensitive` adds the asymmetric
  "don't miss pathology" cost. Watch `recall_abnormal` / `not_normal_sensitivity`,
  and remember `normal` ↔ `near_normal` is the genuinely hard, label-noisy boundary.
- **Windows** — configurable via `cfg.windows` (the three `(center, width)` pairs
  stacked into the 3 channels).
