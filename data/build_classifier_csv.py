"""
build_classifier_csv.py
Build a path + slice_size (+ label) CSV for the ct_brain_classifier.

Each row = one DICOM series folder (the leaf dir that actually holds .dcm
files) with:
    path        absolute path to the series folder
    slice_size  number of .dcm slices in it
    label       e.g. normal | near_normal | abnormal

Series folders are discovered at ANY depth under a class dir, so both the
flat layout (<class>/<studyID>_<seriesID>/*.dcm) and the nested layout
(<class>/<studyID>/<seriesID>/*.dcm) work without changes.

Reusable on NEW data: point --class at any "<label>=<dir>" pair(s). Series are
sorted by path (deterministic), incomplete / scout series (< --min-slices) are
skipped, and at most --per-class series are taken per label (0 = take all).

Examples
--------
# current 3-class run (train_data, take everything available per class):
python3 build_classifier_csv.py \
    --class normal=/root/ritikkumar/train_data/normal \
    --class near_normal=/root/ritikkumar/train_data/near_normal \
    --class abnormal=/root/ritikkumar/train_data/abnormal \
    --out /root/ritikkumar/train_data/csvs/ct_brain_3class_slice_count.csv

# cap rows per class:
python3 build_classifier_csv.py \
    --class normal=/root/ritikkumar/train_data/normal \
    --class abnormal=/root/ritikkumar/train_data/abnormal \
    --per-class normal=1000 --per-class abnormal=1500 \
    --out /data/new/labels.csv
"""
import os
import glob
import csv
import argparse


def parse_pairs(values, what):
    """['a=b', ...] -> {'a': 'b', ...}."""
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} expects '<label>=<value>', got: {v!r}")
        k, val = v.split("=", 1)
        out[k.strip()] = val.strip()
    return out


def collect(root, label, target, min_slices, ext):
    """Return up to `target` (path, slice_size, label) rows from `root` (0 = all).

    A "series folder" is any directory (at any depth under `root`) that directly
    contains >= min_slices files matching `ext`. Results are sorted by path so
    the selection is deterministic.
    """
    if not os.path.isdir(root):
        raise SystemExit(f"[{label}] data dir not found: {root}")
    series = []
    for cur, _dirs, _files in os.walk(root):
        n = len(glob.glob(os.path.join(cur, ext)))
        if n >= min_slices:
            series.append((os.path.abspath(cur), n))
    series.sort(key=lambda x: x[0])
    rows = [(p, n, label) for p, n in series]
    if target:
        rows = rows[:target]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="classes", action="append", required=True,
                    metavar="LABEL=DIR", help="class folder, e.g. normal=/path/to/data (repeatable)")
    ap.add_argument("--per-class", action="append", default=[], metavar="LABEL=N",
                    help="cap rows for a label (repeatable); default = take all")
    ap.add_argument("--min-slices", type=int, default=10,
                    help="skip series with fewer slices (incomplete/scout). default 10")
    ap.add_argument("--ext", default="*.dcm", help="slice glob within a series folder. default *.dcm")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    class_dirs = parse_pairs(args.classes, "class")
    caps = {k: int(v) for k, v in parse_pairs(args.per_class, "per-class").items()}

    all_rows = []
    for label, root in class_dirs.items():
        target = caps.get(label, 0)
        rows = collect(root, label, target, args.min_slices, args.ext)
        all_rows.extend(rows)
        ss = [r[1] for r in rows]
        req = f"(requested {target})" if target else "(all available)"
        stats = f"slice_size min={min(ss)} max={max(ss)} mean={sum(ss)/len(ss):.1f}" if ss else "EMPTY"
        print(f"{label:<10}: {len(rows):>5} {req:<18} {stats}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "slice_size", "label"])
        w.writerows(all_rows)
    print(f"total     : {len(all_rows):>5} rows -> {args.out}")


if __name__ == "__main__":
    main()
