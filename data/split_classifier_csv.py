"""
split_classifier_csv.py
Split a path,slice_size,label CSV into train/val/test CSVs for ct_brain_classifier.

Patient(study)-grouped + class-stratified: every series of one study stays in
exactly ONE split, so no patient leaks across train/val/test. Output CSVs keep
the same columns as the input.

The study key is derived from the series path via --study-key:
    parent  (default) nested layout '.../<studyID>/<seriesID>'  -> parent dir name
    prefix            flat layout   '.../<studyID>_<seriesID>'   -> part before last '_'

Examples
--------
# default 70/15/15 on the nested 3-class CSV:
python3 split_classifier_csv.py \
    --csv /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv \
    --out-dir /root/ritikkumar/train_data/csvs/splits

# 80/10/10, custom seed, no test set:
python3 split_classifier_csv.py --csv labels.csv --out-dir splits \
    --val-fraction 0.1 --test-fraction 0.0 --seed 123

# old flat layout (<studyID>_<seriesID>):
python3 split_classifier_csv.py --csv labels.csv --out-dir splits --study-key prefix
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


def study_key(path, mode="parent"):
    """Map a series path -> study id (groups all series of one study).

    parent: '.../<studyID>/<seriesID>' -> parent dir name (nested layout).
    prefix: '.../<studyID>_<seriesID>' -> part before last '_' (flat layout).
    """
    p = str(path).rstrip("/")
    if mode == "prefix":
        return os.path.basename(p).rsplit("_", 1)[0]
    return os.path.basename(os.path.dirname(p))


def carve(idx, y, groups, frac, seed):
    """Split positions `idx` -> (rest, picked) where picked ~= frac, group-disjoint."""
    if frac <= 0 or len(idx) == 0:
        return idx, np.array([], dtype=int)
    n_splits = max(2, round(1.0 / frac))
    n_splits = min(n_splits, len(np.unique(groups[idx])))
    if n_splits < 2:
        return idx, np.array([], dtype=int)
    try:
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest, pick = next(sp.split(idx, y[idx], groups[idx]))
    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        rest, pick = next(gss.split(idx, y[idx], groups[idx]))
    return idx[rest], idx[pick]


def dist(df):
    return df["label"].value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="input CSV (path,slice_size,label)")
    ap.add_argument("--out-dir", required=True, help="dir to write train/val/test CSVs")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15, help="0 to skip test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study-key", choices=["parent", "prefix"], default="parent",
                    help="how to derive study id from path: parent dir (nested) or "
                         "'<studyID>_<seriesID>' prefix (flat). default parent")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"path", "label"}.issubset(df.columns), "CSV needs at least path,label columns"

    y = df["label"].to_numpy()
    groups = df["path"].map(lambda p: study_key(p, args.study_key)).to_numpy()
    idx = np.arange(len(df))

    trainval, test = carve(idx, y, groups, args.test_fraction, args.seed)
    val_rel = args.val_fraction / max(1e-9, 1.0 - args.test_fraction)
    train, val = carve(trainval, y, groups, val_rel, args.seed)

    # sanity: splits must be patient-disjoint
    gs = [set(groups[s]) for s in (train, val, test)]
    assert not (gs[0] & gs[1]) and not (gs[0] & gs[2]) and not (gs[1] & gs[2]), \
        "patient leakage across splits!"

    os.makedirs(args.out_dir, exist_ok=True)
    for name, sel in [("train", train), ("val", val), ("test", test)]:
        sub = df.iloc[sel].drop(columns=[c for c in ("sid", "study") if c in df.columns])
        out = os.path.join(args.out_dir, f"{name}.csv")
        sub.to_csv(out, index=False)
        npat = len(set(groups[sel]))
        print(f"{name:<6}: {len(sub):>5} series / {npat:>5} patients  {dist(sub)}  -> {out}")
    print(f"total : {len(df):>5} series / {len(set(groups)):>5} patients")


if __name__ == "__main__":
    main()
