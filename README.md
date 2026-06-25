# CT Brain Study Classifier — MaxViT + MIL + Grad-CAM++

3-class (**normal / near-normal / abnormal**) classifier for CT brain **series
sets** (many DICOM slices per `studyID_seriesID`, **one label per set**). Each
slice is rendered into a **3-channel image, one clinical window per channel**:
brain (W80/L40), subdural (W200/L80) and bone (W2800/L600). Architecture and training recipe
follow Qari & Thafar, *"Brain Stroke Detection and Classification Using CT
Imaging with Transformer Models and Explainable AI"* (arXiv:2507.09630), adapted
from their slice-level / 3-class / PNG setup to your study-level / binary / DICOM
setup.

## What matches the paper, and the two adaptations

| Paper component | Here |
|---|---|
| **MaxViT** backbone (CNN MBConv + block + grid attention), from `timm` | `models/build.py` → `maxvit_tiny_tf_224.in1k`. ViT/TNT/ConvNeXt selectable for comparison. |
| Preprocess: resize 224×224, grayscale→3ch, ImageNet norm (Sec 4.4) | `data/transforms.py` |
| Classical augmentation: crop, h-flip, rotation, color jitter (Sec 4.5.1) | `data/transforms.py` |
| Weighted loss for class imbalance (Sec 4.3) | `engine/losses.py` + `class_weights()` |
| Adam, LR∈{1e-3,3e-4,1e-5}, batch∈{16,32,64}, epochs∈{25,40,50,100}, dropout∈{0.03–0.05} (Table 3) | `config.py` defaults |
| Stratified 80/20 split (Sec 5.2) | `stratified_split()` (at **study** level) |
| Metrics: accuracy, precision, recall, F1, AUC, confusion matrix (Sec 5.1/6.1) | `engine/metrics.py` |
| **Grad-CAM++** on deep conv layer `stages.3.blocks.1.conv.conv2_kxk` (Sec 6.3) | `xai/gradcampp.py` (this exact layer is the default and was verified to exist) |

**Adaptation 1 — DICOM input.** The paper used pre-exported PNGs; you have DICOM.
`data/transforms.py` applies RescaleSlope/Intercept → Hounsfield Units → a brain
window (center 40, width 80, configurable) before the 3-channel replication.

**Adaptation 2 — one label per study over many slices.** This is a
Multiple-Instance Learning problem. `models/maxvit_mil.py` encodes every slice
with MaxViT, then a **gated-attention pooling head** (Ilse et al., 2018)
aggregates slices into a single study embedding → one normal/abnormal decision.
The attention weights reveal *which slices* drove the decision; Grad-CAM++ then
shows *where* within those slices. Together they form the interpretability layer
the paper emphasizes.

## Install
```bash
pip install -r requirements.txt
```

## Data layout
```
data_root/                         # e.g. /root/ritikkumar/prepared_brain
  4434203_1B29EAB6/ *.dcm          # one folder per studyID_seriesID set
  4434203_3D906D31/ *.dcm
labels.csv          # columns: study_id,label   (label: normal|near-normal|abnormal or 0|1|2)
```

Generate **random placeholder labels** (to test the training setup before real
labels exist):
```bash
python make_random_labels.py --data_root /root/ritikkumar/prepared_brain --out labels_random.csv
```

## Train
```bash
python train_main.py \
  --data_root /data/ct_studies --labels_csv /data/labels.csv \
  --backbone maxvit --epochs 50 --lr 3e-4 --batch_size 8 \
  --out_dir runs/maxvit_mil
```
Any `config.py` field is a CLI flag. Swap `--backbone vit|tnt|convnext` for the
paper's comparison models.

## TensorBoard
```bash
tensorboard --logdir runs
```
Logged: scalars (train/val loss, accuracy, precision, recall, F1, AUC, lr),
images (confusion matrix, ROC, **Grad-CAM++ overlays** on top-attended slices,
per-study **attention** bar charts), weight/grad histograms, and an HPARAMS entry
with best val-F1 vs the full config.

## Explain a single study
```bash
python infer.py --ckpt runs/maxvit_mil/best.pt \
  --study_dir /data/ct_studies/study_0007 --out_dir explanations/study_0007 --topk 3
```
Prints the study-level prediction and saves Grad-CAM++ overlay PNGs for the most
attended slices.

## Notes / knobs
- `freeze_backbone=True` reproduces the paper's "freeze all but classifier"
  setting; default `False` (full fine-tune) usually scores higher on small sets.
- The paper's best result added **cGAN** synthetic minority-class images on top of
  classical augmentation for ~0.1% gain over classical alone (MaxViT 98.0 vs
  97.9). Classical augmentation is built in; a cGAN generator can be dropped in by
  writing synthetic studies to `data_root` and adding their rows to `labels.csv`.
- For a held-out **test** set, create a third split or a separate `labels.csv` and
  run `infer.py` per study; `engine/metrics.py` gives the same metrics offline.
- Slice windowing: if your abnormalities are subdural/bone, expose a second window
  (e.g. center 50/width 130) and stack windows into the 3 channels instead of
  replicating — a common CT trick that often beats single-window replication.
```
