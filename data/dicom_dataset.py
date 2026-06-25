"""
data/dicom_dataset.py
Series-level Multiple-Instance Learning dataset for CT brain studies.

Each labelable unit is a `<studyID>_<seriesID>` folder of DICOM slices carrying
ONE 3-class label. A dataset item = one set = a "bag" of slice images + its label.

Expected layout (flexible -- adapt `index_studies` if yours differs):

    data_root/                         # e.g. /root/ritikkumar/prepared_brain
        4434203_1B29EAB6/  *.dcm ...   # one folder per studyID_seriesID set
        4434203_3D906D31/  *.dcm ...
    labels.csv  ->  study_id,label
        label in {normal, near-normal, abnormal}  or  {0, 1, 2}

Slice -> 3 channels: HU windowed into brain (W80/L40), subdural (W200/L80) and
bone (W2800/L600) -- see data/transforms.py.

Bag construction (StudyMILDataset): all slices if cfg.all_slices, else a random
subset (train) / evenly-spaced subset (eval) capped at cfg.max_slices_per_study.
Collation (mil_collate) pads each batch of bags to the max bag size and returns a
boolean mask so the MIL attention head ignores padding.

Splitting: use grouped_split() -- patient(study)-grouped + class-stratified
train/val/test so no series of a study leaks across splits.
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit

try:
    import pydicom
except ImportError:  # allow import without pydicom for unit-testing collate
    pydicom = None

from .transforms import build_transforms, dicom_to_multiwindow

# Default label aliases; numeric labels pass through as-is. The active mapping is
# built per-run from cfg.class_names via build_label_map() so the same code serves
# the 2-class (normal/abnormal) and the legacy 3-class task.
_LABEL_ALIASES = {
    "near_normal": "near-normal", "nearnormal": "near-normal", "near normal": "near-normal",
}


def build_label_map(class_names):
    """Map class name / index -> int label index, driven by cfg.class_names.

    e.g. ("normal","abnormal") -> {"normal":0,"abnormal":1,"0":0,"1":1,0:0,1:1}.
    Aliases (near_normal -> near-normal) are honored if the canonical name is used.
    """
    m = {}
    for i, name in enumerate(class_names):
        m[name.strip().lower()] = i
        m[str(i)] = i
        m[i] = i
    for alias, canon in _LABEL_ALIASES.items():
        if canon in m:
            m[alias] = m[canon]
    return m


def _to_label(value, label_map):
    key = str(value).strip().lower() if isinstance(value, str) else int(value)
    if key not in label_map:
        raise KeyError(f"label {value!r} not in class_names mapping {sorted(set(label_map))}")
    return label_map[key]


def _find_slices(sdir):
    """All DICOM slice paths under a series folder (falls back to any file)."""
    slices = sorted(glob.glob(os.path.join(sdir, "**", "*.dcm"), recursive=True))
    if not slices:
        slices = sorted(glob.glob(os.path.join(sdir, "**", "*"), recursive=True))
        slices = [p for p in slices if os.path.isfile(p)]
    return slices


def index_studies(data_root: str, labels_csv: str, class_names=("normal", "abnormal")):
    """(legacy) Index studies from data_root + a study_id,label CSV.

    Return list of (study_id, [slice_paths...], label_int).
    """
    label_map = build_label_map(class_names)
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"study_id", "label"}.issubset(df.columns), "labels.csv needs study_id,label"
    items = []
    for _, row in df.iterrows():
        sid = str(row["study_id"])
        label = _to_label(row["label"], label_map)
        sdir = os.path.join(data_root, sid)
        slices = _find_slices(sdir)
        if slices:
            items.append((sid, slices, label))
        else:
            print(f"[warn] no slices found for study {sid} at {sdir}")
    return items


def index_studies_from_csv(csv_path: str, class_names=("normal", "abnormal")):
    """Index studies from a path,label CSV (preferred; one row per series folder).

    Columns: `path` (absolute series-folder path) and `label`; `slice_size` ignored
    if present. study_id is the folder basename. Returns (study_id, [slice_paths], label).
    """
    label_map = build_label_map(class_names)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"path", "label"}.issubset(df.columns), f"{csv_path} needs path,label columns"
    items = []
    for _, row in df.iterrows():
        sdir = str(row["path"]).rstrip("/")
        # study_id must let _study_of() recover the patient/study so splits stay
        # patient-disjoint. Flat layout '<studyID>_<seriesID>' already encodes it
        # in the basename; nested layout '<studyID>/<seriesID>' carries it in the
        # parent dir, so fold it in as '<studyID>_<seriesID>'.
        base = os.path.basename(sdir)
        sid = base if "_" in base else f"{os.path.basename(os.path.dirname(sdir))}_{base}"
        label = _to_label(row["label"], label_map)
        slices = _find_slices(sdir)
        if slices:
            items.append((sid, slices, label))
        else:
            print(f"[warn] no slices found at {sdir}")
    return items


def _instance_number(path):
    """Sort key by DICOM InstanceNumber so slices stay in anatomical order."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return float(getattr(ds, "InstanceNumber", 0) or 0)
    except Exception:
        return 0.0


class StudyMILDataset(Dataset):
    def __init__(self, items, cfg, train: bool):
        self.items = items
        self.cfg = cfg
        self.train = train
        self.tf = build_transforms(cfg, train=train)

    def __len__(self):
        return len(self.items)

    def _load_slice(self, path):
        """Load one slice -> 3xHxW float, or None if the DICOM is unreadable/corrupt.

        Data may still be downloading, so truncated/partially-written .dcm files are
        expected. Returning None lets __getitem__ skip them instead of crashing training.
        """
        try:
            ds = pydicom.dcmread(path)
            jitter = self.cfg.window_jitter if self.train else 0.0  # HU-window aug: train only
            img = dicom_to_multiwindow(ds, self.cfg.windows, jitter=jitter)  # HxWx3 uint8
            return self.tf(img)                                      # 3xHxW float
        except Exception as e:
            print(f"[warn] skip unreadable slice {path}: {type(e).__name__}")
            return None

    def __getitem__(self, idx):
        sid, slices, label = self.items[idx]
        # order slices anatomically, then sample for the bag
        slices = sorted(slices, key=_instance_number)
        if getattr(self.cfg, "all_slices", False):
            # use every slice in anatomical order (variable K per set)
            chosen = list(range(len(slices)))
        elif self.train:
            k = min(self.cfg.train_slices_per_study, len(slices))
            chosen = sorted(random.sample(range(len(slices)), k))
        else:
            # evenly sample up to the cap at eval for determinism
            k = min(self.cfg.max_slices_per_study, len(slices))
            chosen = np.linspace(0, len(slices) - 1, k).round().astype(int).tolist()
        # load defensively: drop slices that fail to decode (corrupt / mid-download)
        tiles = [t for t in (self._load_slice(slices[i]) for i in chosen) if t is not None]
        if not tiles:
            # every chosen slice failed -> scan the rest of the study for any good slice
            chosen_set = set(chosen)
            for i in range(len(slices)):
                if i in chosen_set:
                    continue
                t = self._load_slice(slices[i])
                if t is not None:
                    tiles.append(t)
                    break
        if not tiles:
            # whole study unreadable (rare) -> zero placeholder so collation/training continues
            print(f"[warn] no readable slices for study {sid}; using zero placeholder")
            tiles = [torch.zeros(3, self.cfg.image_size, self.cfg.image_size)]
        bag = torch.stack(tiles)                                  # K x 3 x H x W
        return {"bag": bag, "label": torch.tensor(label, dtype=torch.long), "study_id": sid}


def effective_lengths(items, cfg, train: bool):
    """Effective bag size (#slices actually loaded) per item, for length bucketing.

    Mirrors StudyMILDataset.__getitem__: all slices if cfg.all_slices, else capped at
    train_slices_per_study (train) / max_slices_per_study (eval).
    """
    if getattr(cfg, "all_slices", False):
        cap = None
    else:
        cap = cfg.train_slices_per_study if train else cfg.max_slices_per_study
    return [len(sl) if cap is None else min(cap, len(sl)) for _, sl, _ in items]


class LengthBucketedBatchSampler(Sampler):
    """Batch studies of similar bag size together to minimize padding waste.

    mil_collate pads every bag in a batch up to the batch's largest bag, and the
    backbone encodes the padded slices too -- so a batch mixing a 13-slice and a
    128-slice study wastes compute on ~115 padded slices per small study. Grouping
    similar-length studies into the same batch removes most of that waste.

    train (shuffle=True): shuffle all indices, cut into pools of pool_factor*batch_size,
      sort each pool by length, split into batches, then shuffle the batch order. Keeps
      each batch length-homogeneous while still randomizing composition every epoch.
    eval (shuffle=False): deterministic ascending-by-length batches.

    Call set_epoch(e) each epoch so the shuffle differs (and stays resumable).
    """
    def __init__(self, lengths, batch_size, shuffle, seed=42, pool_factor=20, drop_last=False):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool_factor = pool_factor
        self.drop_last = drop_last
        self.epoch = 0
        n = len(self.lengths)
        self._num_batches = n // batch_size if drop_last else (n + batch_size - 1) // batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        idx = list(range(n))
        if self.shuffle:
            g = random.Random(self.seed + self.epoch)
            g.shuffle(idx)
            pool = self.batch_size * self.pool_factor
            batches = []
            for i in range(0, n, pool):
                chunk = sorted(idx[i:i + pool], key=lambda j: self.lengths[j])
                for b in range(0, len(chunk), self.batch_size):
                    bt = chunk[b:b + self.batch_size]
                    if self.drop_last and len(bt) < self.batch_size:
                        continue
                    batches.append(bt)
            g.shuffle(batches)
        else:
            idx.sort(key=lambda j: self.lengths[j])
            batches = [idx[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
        yield from batches

    def __len__(self):
        return self._num_batches


def mil_collate(batch):
    """Pad variable-length bags to max K in batch; build attention mask."""
    maxK = max(b["bag"].shape[0] for b in batch)
    B = len(batch)
    C, H, W = batch[0]["bag"].shape[1:]
    bags = torch.zeros(B, maxK, C, H, W)
    mask = torch.zeros(B, maxK, dtype=torch.bool)      # True = real slice
    labels = torch.empty(B, dtype=torch.long)
    sids = []
    for i, b in enumerate(batch):
        k = b["bag"].shape[0]
        bags[i, :k] = b["bag"]
        mask[i, :k] = True
        labels[i] = b["label"]
        sids.append(b["study_id"])
    return {"bag": bags, "mask": mask, "label": labels, "study_id": sids}


def stratified_split(items, val_fraction, seed):
    """Series-level stratified train/val split (LEAKS across series of one study).

    Kept for backward-compat only. Prefer grouped_split() for real data.
    """
    labels = [it[2] for it in items]
    tr, va = train_test_split(
        items, test_size=val_fraction, random_state=seed, stratify=labels
    )
    return tr, va


def _study_of(sid: str) -> str:
    """'<studyID>_<seriesID>' -> '<studyID>'.  All series of one study share this key."""
    return sid.rsplit("_", 1)[0]


def grouped_split(items, val_fraction, test_fraction, seed):
    """Patient(study)-grouped, class-stratified train / val / test split.

    Every series of the same study lands in exactly ONE split, so no patient
    leaks across train/val/test. Splits are also stratified by class as far as
    the grouping allows. Returns (train_items, val_items, test_items).

    val_fraction / test_fraction are fractions of the FULL set (e.g. 0.15/0.15
    -> ~70/15/15). They are approximate because group K-folding can't hit an
    exact ratio while keeping groups intact.
    """
    y = np.array([it[2] for it in items])
    groups = np.array([_study_of(it[0]) for it in items])
    idx = np.arange(len(items))

    def carve(sub, frac):
        """Split positions `sub` -> (rest, picked) where picked ~= frac, group-disjoint."""
        if frac <= 0 or len(sub) == 0:
            return sub, np.array([], dtype=int)
        n_splits = max(2, round(1.0 / frac))
        n_splits = min(n_splits, len(np.unique(groups[sub])))  # can't exceed #groups
        if n_splits < 2:
            return sub, np.array([], dtype=int)
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            rest, pick = next(splitter.split(sub, y[sub], groups[sub]))
        except Exception:
            gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
            rest, pick = next(gss.split(sub, y[sub], groups[sub]))
        return sub[rest], sub[pick]

    trainval, test = carve(idx, test_fraction)
    val_rel = val_fraction / max(1e-9, 1.0 - test_fraction)   # val as fraction of remaining
    train, val = carve(trainval, val_rel)
    take = lambda ix: [items[i] for i in ix]
    return take(train), take(val), take(test)


def class_weights(items, num_classes=2):
    """Inverse-frequency class weights for the weighted loss (paper imbalance handling)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, _, y in items:
        counts[y] += 1
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
