"""
prepare_cases.py
Precompute the test-set analysis payload consumed by webapp/serve.py.

For each dataset (held-out enriched split + June 21-27 production week) it:
  - loads the v3 per-series probabilities (series_probs_{val,test}.csv),
  - aggregates to per-study (patient) mean probabilities,
  - computes the 2-stage cascade thresholds (T1 abn-vs-rest, T2 near-vs-normal)
    on VAL for a few S1 operating points (applied to TEST -> no leakage), so the
    frontend can switch operating point live without recompute,
  - joins each study to its study_path, StudyInstanceUID (if available),
    radiologist classification + findings (June set only).

Writes webapp/cases.json. Cascade decision itself is computed in the browser
from the stored per-study probs + thresholds (see index.html).

Run with the ct_brain venv:
  /root/ritikkumar/ct_brain/bin/python webapp/prepare_cases.py
"""
import os, csv, re, json, argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score

csv.field_size_limit(10 ** 9)

REPO = "/root/ritikkumar/ct_brain_classifier"
RUN = os.path.join(REPO, "runs/maxvit384_3class_clinical_v3")
CLASS_NAMES = ("normal", "near_normal", "abnormal")

S1_TARGETS = [0.95, 0.98, 0.99]   # abnormal-sensitivity operating points (Stage 1)
S2_TARGET = 0.95                  # near_normal-sensitivity target (Stage 2)


# ---------------------------------------------------------------- cascade math
def load_probs(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            pid, lab, p0, p1, p2 = line.strip().split(",")
            rows.append((pid, int(lab), float(p0), float(p1), float(p2)))
    return rows


def patient_mean(rows):
    byp = defaultdict(list)
    for pid, lab, p0, p1, p2 in rows:
        byp[pid].append((lab, p0, p1, p2))
    pids, y, P = [], [], []
    for pid, rs in byp.items():
        pids.append(pid)
        y.append(rs[0][0])
        P.append([float(np.mean([r[i] for r in rs])) for i in (1, 2, 3)])
    return pids, np.array(y), np.array(P)


def thr_at(y, score, poslab, target):
    """Smallest-FPR threshold on the ROC that still reaches `target` recall."""
    pos = (y == poslab).astype(int)
    fpr, tpr, th = roc_curve(pos, score)
    ok = tpr >= target
    return float(th[int(np.where(ok)[0][np.argmin(fpr[ok])])] if ok.any()
                 else th[int(np.argmax(tpr))])


def cascade_thresholds(vy, vP):
    """Return {s1_target: {'T1':..,'T2':..}} computed on the validation set."""
    out = {}
    for s1 in S1_TARGETS:
        T1 = thr_at(vy, vP[:, 2], 2, s1)
        vpass = vP[:, 2] < T1
        vadj = vP[vpass, 1] / (vP[vpass, 1] + vP[vpass, 0] + 1e-9)
        m = vy[vpass] != 2
        T2 = thr_at(vy[vpass][m], vadj[m], 1, S2_TARGET)
        out[f"{s1:.2f}"] = {"T1": T1, "T2": T2}
    return out


# ---------------------------------------------------------------- joins
def study_paths_from_split(split_csv):
    """studyID (parent-dir basename) -> study directory path."""
    m = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            series_path = row["path"]
            study_dir = os.path.dirname(series_path)
            m[os.path.basename(study_dir)] = study_dir
    return m


def folder_to_iuid(download_log):
    """June download log: '<iuid> -> <date>/<HEX> (id <HEX>): ...' -> {HEX: iuid}."""
    pat = re.compile(r"^(\S+) -> \S+ \(id ([0-9A-Fa-f]+)\):")
    m = {}
    if not os.path.exists(download_log):
        return m
    for line in open(download_log):
        mt = pat.match(line.strip())
        if mt:
            m[mt.group(2)] = mt.group(1)
    return m


def iuid_meta(big_csv):
    """iuid -> {classification, findings, study_name} from the report CSV."""
    m = {}
    if not os.path.exists(big_csv):
        return m
    with open(big_csv, newline="") as f:
        for row in csv.DictReader(f):
            u = row.get("study_iuid")
            if u and u not in m:
                m[u] = {
                    "classification": (row.get("classification") or "").strip(),
                    "findings": (row.get("findings_list") or "").strip(),
                    "study_name": (row.get("study_name") or "").strip(),
                }
    return m


def folder_meta_from_report(report_csv):
    """Report CSV whose `study_path` ends in the folder-ID hex (the held-out
    join): {folder_id: {iuid, classification, findings, study_name}}."""
    m = {}
    if not report_csv or not os.path.exists(report_csv):
        return m
    with open(report_csv, newline="") as f:
        for row in csv.DictReader(f):
            fid = os.path.basename((row.get("study_path") or "").rstrip("/"))
            if fid and fid not in m:
                m[fid] = {
                    "iuid": (row.get("study_iuid") or "").strip(),
                    "classification": (row.get("classification") or "").strip(),
                    "findings": (row.get("findings_list") or "").strip(),
                    "study_name": (row.get("study_name") or "").strip(),
                }
    return m


# ---------------------------------------------------------------- build one dataset
def build_dataset(name, run_dir, split_csv, folder_meta=None):
    """folder_meta: {folder_id -> {iuid, classification, findings, study_name}}."""
    folder_meta = folder_meta or {}
    vpids, vy, vP = patient_mean(load_probs(os.path.join(run_dir, "series_probs_val.csv")))
    tpids, ty, tP = patient_mean(load_probs(os.path.join(run_dir, "series_probs_test.csv")))

    thresholds = cascade_thresholds(vy, vP)
    spaths = study_paths_from_split(split_csv)

    cases = []
    for pid, lab, p in zip(tpids, ty, tP):
        meta = folder_meta.get(pid, {})
        cases.append({
            "pid": pid,
            "study_path": spaths.get(pid, ""),
            "iuid": meta.get("iuid", ""),
            "actual": CLASS_NAMES[int(lab)],
            "actual_idx": int(lab),
            "radiologist_class": meta.get("classification", ""),
            "findings": meta.get("findings", ""),
            "study_name": meta.get("study_name", ""),
            "p_normal": round(float(p[0]), 5),
            "p_near": round(float(p[1]), 5),
            "p_abn": round(float(p[2]), 5),
        })

    # dataset-level AUCs for the header
    auc_abn = float(roc_auc_score((ty == 2).astype(int), tP[:, 2]))
    m = ty != 2
    padj = tP[m, 1] / (tP[m, 1] + tP[m, 0] + 1e-9)
    auc_near = float(roc_auc_score((ty[m] == 1).astype(int), padj)) if m.any() else None

    counts = {CLASS_NAMES[i]: int((ty == i).sum()) for i in range(3)}
    return {
        "name": name,
        "n_studies": len(cases),
        "counts": counts,
        "auc_abn_vs_rest": round(auc_abn, 4),
        "auc_near_vs_normal": round(auc_near, 4) if auc_near is not None else None,
        "s2_target": S2_TARGET,
        "thresholds": thresholds,
        "cases": cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "webapp/cases.json"))
    args = ap.parse_args()

    payload = {"class_names": list(CLASS_NAMES), "model": "v3 (maxvit384_3class_clinical_v3, best.pt)", "datasets": {}}

    # Held-out (old Jan-May data): folder-ID -> iuid/findings via the report CSV
    # whose study_path column ends in the folder-ID hex.
    heldout_meta = folder_meta_from_report(
        "/root/ritikkumar/disk_vdc/ct_brain_orig_csv/ct_brain_jan-may_2026_final.csv")
    payload["datasets"]["heldout"] = build_dataset(
        "Held-out enriched split",
        RUN,
        "/root/ritikkumar/train_data/csvs/splits/test.csv",
        folder_meta=heldout_meta,
    )

    # June 21-27: folder-ID -> iuid via the download log, then iuid -> findings
    # via the report CSV.
    f2u = folder_to_iuid("/root/ritikkumar/disk_vdc/test_data/_download.log")
    u2m = iuid_meta("/root/ritikkumar/disk_vdc/test_data/ct_brain_test_set_june_21_27.csv")
    june_meta = {fid: {"iuid": u, **u2m.get(u, {})} for fid, u in f2u.items()}
    payload["datasets"]["june"] = build_dataset(
        "June 21-27 production week",
        os.path.join(RUN, "eval_test_21_27"),
        "/root/ritikkumar/disk_vdc/test_data/csvs/test_june_21_27.csv",
        folder_meta=june_meta,
    )

    with open(args.out, "w") as f:
        json.dump(payload, f)
    for k, d in payload["datasets"].items():
        print(f"[{k}] {d['n_studies']} studies  counts={d['counts']}  "
              f"AUC(abn)={d['auc_abn_vs_rest']}  AUC(near)={d['auc_near_vs_normal']}  "
              f"thresholds={d['thresholds']}")
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
