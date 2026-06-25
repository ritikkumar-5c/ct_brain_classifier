"""
config.py
Central configuration. Defaults follow the hyperparameter grid in
Qari & Thafar, "Brain Stroke Detection and Classification Using CT Imaging
with Transformer Models and Explainable AI" (Table 3), adapted to a
*3-class, study-level* task (normal / near_normal / abnormal) on DICOM input.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass
class Config:
    # ----- data -----
    # Two ways to supply data:
    #  (A) pre-split path CSVs (preferred): set train_csv/val_csv/test_csv. Each CSV
    #      has columns  path,label  (+ optional slice_size); `path` = a series folder
    #      of *.dcm slices. Built by download_utils/build_classifier_csv.py and split
    #      by download_utils/split_classifier_csv.py. No data_root needed.
    #  (B) legacy: data_root (one folder per study_id) + labels_csv (study_id,label);
    #      split in-memory at train time via grouped_split().
    train_csv: str = "/root/ritikkumar/train_data/csvs/splits/train.csv"   # columns: path,label[,slice_size]
    val_csv: str = "/root/ritikkumar/train_data/csvs/splits/val.csv"
    test_csv: str = "/root/ritikkumar/train_data/csvs/splits/test.csv"     # "" to skip the held-out test set
    data_root: str = "/path/to/dataset"      # (legacy) root with one folder per study
    labels_csv: str = "/path/to/labels.csv"  # (legacy) columns: study_id,label  (label in {normal,near_normal,abnormal} or {0,1,2})
    image_size: int = 384                     # maxvit_tiny_tf_384 native input
    # ----- HU windowing -> 3 channels -----
    # Each slice is rendered into a 3-channel image, one clinical window per
    # channel (center=level L, width W), instead of replicating one grayscale
    # window. Order = (channel 0, channel 1, channel 2).
    #   Brain     W80   L40
    #   Subdural  W200  L80
    #   Bone      W2800 L600
    windows: Tuple[Tuple[float, float], ...] = (
        (40.0, 80.0),      # brain    (center, width)
        (80.0, 200.0),     # subdural
        (600.0, 2800.0),   # bone
    )
    # legacy single-window fields (kept for backward-compat / fallback only)
    window_center: float = 40.0
    window_width: float = 80.0
    # MIL bag construction
    max_slices_per_study: int = 96            # cap a study's bag size (eval)
    train_slices_per_study: int = 96          # random-sample this many slices per study while training
    all_slices: bool = False                  # use EVERY slice per set (ignores the two caps above)
    # ImageNet normalization (3-channel replicated grayscale, paper 4.4)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ----- augmentation (train only; set a knob to 0 / False to disable it) -----
    # HU-window jitter: per-slice random scaling of each window's (center, width)
    # by +/- this fraction BEFORE windowing -> CT-specific intensity augmentation.
    # e.g. 0.1 => center/width each multiplied by U(0.9, 1.1). 0.0 = off.
    window_jitter: float = 0.05               # CT-correct intensity aug (HU window jitter); preferred over ColorJitter
    aug_hflip: bool = True                    # random horizontal flip (p=0.5)
    aug_rotation_deg: float = 15.0            # +/- degrees (0 = off)
    aug_crop_scale_min: float = 0.85          # RandomResizedCrop area lower bound (1.0 = off)
    aug_brightness: float = 0.0               # ColorJitter brightness (0 = off; window_jitter used instead)
    aug_contrast: float = 0.0                 # ColorJitter contrast (0 = off; window_jitter used instead)

    # ----- model -----
    # backbone: maxvit (primary, paper) | vit | tnt | convnext (comparison)
    backbone: str = "maxvit"
    timm_name: str = "maxvit_tiny_tf_384.in1k"  # 384px MaxViT-tiny; resolved per-backbone in models/build.py
    pretrained: bool = True
    freeze_backbone: bool = False             # paper froze all but classifier; False = fine-tune fully (usually better)
    mil_attn_dim: int = 256                   # gated-attention hidden dim
    dropout: float = 0.04                     # paper Table 3: {0.03, 0.04, 0.05}
    num_classes: int = 3                       # normal / near_normal / abnormal
    class_names: Tuple[str, ...] = ("normal", "near_normal", "abnormal")

    # ----- training -----
    optimizer: str = "adam"                   # paper Table 3: Adam selected
    lr: float = 3e-4                          # paper grid: {1e-3, 3e-4, 1e-5}
    weight_decay: float = 1e-4
    batch_size: int = 16                      # NOTE: studies-per-batch (each study = a bag of slices)
    epochs: int = 50                          # paper grid: {25, 40, 50, 100}
    warmup_epochs: int = 2
    use_class_weights: bool = True            # paper used a weighted class function for imbalance
    loss: str = "weighted_ce"                 # weighted_ce | focal
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05             # softens the noisy normal/near_normal boundary
    grad_clip: float = 1.0
    early_stop_patience: int = 10             # epochs w/o val-F1 improvement
    num_workers: int = 4
    seed: int = 42
    # length-bucketed batching: group studies of similar bag size into the same batch
    # so mil_collate pads less -> the backbone wastes less compute on padded slices.
    # train batches stay randomized (shuffle within pools of bucket_pool_factor*batch_size).
    length_bucketing: bool = True
    bucket_pool_factor: int = 20              # larger = tighter length grouping, less randomness
    # ----- memory / variable-slice handling -----
    # To feed ALL slices of every set: all_slices=True, batch_size=1, slice_chunk>0,
    # grad_checkpoint=True. These bound activation memory regardless of bag size K.
    slice_chunk: int = 96                     # encode slices in chunks of N (0 = all at once); bounds backbone activation memory
    grad_checkpoint: bool = True              # REQUIRED here: full fine-tune at B16xK96x384 OOMs without it (47.7 GB w/ it). ~20-30% slower
    use_amp: bool = True                      # mixed-precision (autocast+GradScaler) on CUDA; auto-off on CPU
    grad_accum_steps: int = 1                 # accumulate grads over N batches -> larger effective batch
    # patient(study)-grouped split: every series of a study stays in ONE split (no leakage).
    # fractions of the FULL set -> default ~70/15/15 train/val/test.
    val_fraction: float = 0.15
    test_fraction: float = 0.15               # set 0 to skip the held-out test set

    # ----- xai -----
    # MaxViT deep conv layer the paper found most clinically focused (Sec 6.3)
    gradcam_layer: str = "stages.3.blocks.1.conv.conv2_kxk"
    xai_enabled: bool = True                   # set False for quick smoke tests
    xai_every_n_epochs: int = 5               # log Grad-CAM++ overlays to TensorBoard this often
    xai_num_studies: int = 4                  # studies to visualize each time

    # ----- io -----
    out_dir: str = "./runs/maxvit_mil"
    log_histograms: bool = True

    def to_dict(self):
        return asdict(self)


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
