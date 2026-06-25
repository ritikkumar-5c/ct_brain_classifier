# CT Brain Study Classifier — MaxViT + MIL + Grad-CAM++

3-class (**normal / near_normal / abnormal**) classifier for CT brain **series
sets** (many DICOM slices per `studyID/seriesID`, **one label per set**). Each
slice is rendered into a **3-channel image, one clinical window per channel**:
brain (W80/L40), subdural (W200/L80) and bone (W2800/L600). Architecture and
training recipe follow Qari & Thafar, *"Brain Stroke Detection and
Classification Using CT Imaging with Transformer Models and Explainable AI"*
(arXiv:2507.09630), adapted from their slice-level / PNG setup to a
**study-level, multi-slice, DICOM** setup.

## What matches the paper, and the two adaptations

| Paper component | Here |
|---|---|
| **MaxViT** backbone (CNN MBConv + block + grid attention), from `timm` | `models/build.py` → `maxvit_tiny_tf_384.in1k` (384px). ViT/TNT/ConvNeXt selectable for comparison. |
| Preprocess: resize, grayscale→3ch, ImageNet norm (Sec 4.4) | `data/transforms.py` (resize **384×384**; 3 channels = 3 HU windows, see below) |
| Classical augmentation: crop, h-flip, rotation, color jitter (Sec 4.5.1) | `data/transforms.py` |
| Weighted loss for class imbalance (Sec 4.3) | `engine/losses.py` + `class_weights()` |
| Adam, LR∈{1e-3,3e-4,1e-5}, batch∈{16,32,64}, epochs∈{25,40,50,100}, dropout∈{0.03–0.05} (Table 3) | `config.py` defaults |
| Grouped, class-stratified split | `data/split_classifier_csv.py` (study-level, no patient leakage) |
| Metrics: accuracy, precision, recall, F1, AUC, confusion matrix (Sec 5.1/6.1) | `engine/metrics.py` (macro P/R/F1; macro-OVR AUC for 3-class) |
| **Grad-CAM++** on deep conv layer `stages.3.blocks.1.conv.conv2_kxk` (Sec 6.3) | `xai/gradcampp.py` (this exact layer is the default and was verified to exist) |

**Adaptation 1 — DICOM input + multi-window channels.** The paper used
pre-exported PNGs; here raw DICOM pixels are converted via
RescaleSlope/Intercept → Hounsfield Units, then three clinical windows are
applied — brain (W80/L40), subdural (W200/L80), bone (W2800/L600) — one per
channel (`data/transforms.py`), instead of replicating a single grayscale window.

**Adaptation 2 — one label per study over many slices.** This is a
Multiple-Instance Learning problem. `models/maxvit_mil.py` encodes every slice
with MaxViT, then a **gated-attention pooling head** (Ilse et al., 2018)
aggregates slices into a single study embedding → one 3-class decision. The
attention weights reveal *which slices* drove the decision; Grad-CAM++ then
shows *where* within those slices.

## Install
```bash
pip install -r requirements.txt
```

## Data layout
Series folders of `*.dcm` slices, **nested** as `<class>/<studyID>/<seriesID>/`:
```
train_data/
  normal/      00197F33/ 00198653/ *.dcm        # one study can hold multiple series
                        9A5727A1/ *.dcm
  near_normal/ ...
  abnormal/    ...
```
(The legacy flat layout `<studyID>_<seriesID>/*.dcm` is still supported — the
study id is recovered from either layout for patient-grouped splitting.)

## Prepare the CSVs (build → split)
Build one `path,slice_size,label` CSV across the 3 classes, then split it
patient-grouped + class-stratified into train/val/test:
```bash
# 1) index every series folder (any depth) into a labeled CSV
python data/build_classifier_csv.py \
  --class normal=/root/ritikkumar/train_data/normal \
  --class near_normal=/root/ritikkumar/train_data/near_normal \
  --class abnormal=/root/ritikkumar/train_data/abnormal \
  --out /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv

# 2) 70/15/15 study-grouped split (no patient leaks across splits)
python data/split_classifier_csv.py \
  --csv /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv \
  --out-dir /root/ritikkumar/train_data/csvs/splits
```
This writes `splits/{train,val,test}.csv`, which are the defaults in `config.py`.

## Train
`config.py` defaults already encode the full recipe — split CSVs, MaxViT-384 @
384×384, ≤96 slices/study, batch 16, `slice_chunk=96`, `grad_checkpoint=True`,
length bucketing, `weighted_ce` + `label_smoothing=0.05`, `window_jitter=0.05`
(ColorJitter off). Final run command:
```bash
python train_main.py \
  --out_dir runs/maxvit384_3class \
  --xai_enabled true --log_histograms true --use_amp true
```
Run it detached (≈30 min/epoch, up to 50 epochs with early stopping):
```bash
nohup python train_main.py \
  --out_dir runs/maxvit384_3class \
  --xai_enabled true --log_histograms true --use_amp true \
  > runs/maxvit384_3class.log 2>&1 &
```
Any `config.py` field is a CLI flag, e.g. swap `--backbone vit|tnt|convnext` for
the paper's comparison models, or override the recipe knobs directly.

> **Memory note.** Full fine-tune at batch 16 × 96 slices × 384px **requires
> `grad_checkpoint=True`** (default) — without it, the ~1536 slice activations
> per step OOM even on an 80GB A100. With it, peak ≈ 47.7 GB (~20–30% slower).
> `slice_chunk` alone does *not* bound training memory in full-fine-tune mode.
> To trade the slowdown for memory differently, use a smaller real batch with
> `--grad_accum_steps` (e.g. `--batch_size 4 --grad_accum_steps 4`).

## TensorBoard
```bash
tensorboard --logdir runs
```
Logged: scalars (train/val loss, accuracy, macro precision/recall/F1, AUC, lr),
images (confusion matrix, **Grad-CAM++ overlays** on top-attended slices,
per-study **attention** bar charts), weight/grad histograms, and an HPARAMS entry
with best val-F1 vs the full config. (The ROC curve is logged for the binary case
only; for 3-class, rely on macro-OVR AUC + the confusion matrix.)

## Explain a single study
```bash
python infer.py --ckpt runs/maxvit_mil/best.pt \
  --study_dir /root/ritikkumar/train_data/normal/00197F33/00198653 \
  --out_dir explanations/00198653 --topk 3
```
Prints the study-level prediction and saves Grad-CAM++ overlay PNGs for the most
attended slices.

## Notes / knobs
- `freeze_backbone=True` reproduces the paper's "freeze all but classifier"
  setting; default `False` (full fine-tune) usually scores higher.
- **Memory / variable bag size.** `slice_chunk>0` encodes slices in chunks so
  activation memory stays bounded regardless of bag size K. To feed *every*
  slice of every study: `all_slices=True, batch_size=1, slice_chunk>0,
  grad_checkpoint=True`. `length_bucketing` groups similar-length studies into a
  batch so the backbone wastes less compute on padding.
- **Class imbalance.** `use_class_weights=True` (default) applies inverse-
  frequency weights via `weighted_ce`; `loss=focal` is also available. Watch the
  per-class recall of the minority `normal` class, and remember `normal` ↔
  `near_normal` is the genuinely hard, label-noisy boundary.
- Slice windows are configurable via `cfg.windows` (the three (center, width)
  pairs stacked into the 3 channels).
