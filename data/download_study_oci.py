"""
Download CT studies from the OCI `secure-dcm` bucket given StudyInstanceUID(s)
(bare, an api.5cnetwork.com download URL, or a CSV column), prune junk/duplicate
series at the STUDY level, then z-sort + anonymize the kept series.

Flow:
  StudyInstanceUID --(5C storage-path API)--> study_path (e.g. 2026/06/27/AE4EFAE0)
  list objects under study_path in OCI secure-dcm, grouped by series
  PRUNE: read ONE header per series, classify it, decide the keep-set:
       - drop junk (localizer/scout, dose, secondary, derived/reformatted,
         non-axial, too-few-slices)
       - drop bone-kernel reconstructions as duplicates of the soft-kernel series
       - NEVER empty a study: if only bone-kernel series exist, keep them
  download only the kept series: slices sorted by ImagePositionPatient z,
       anonymized, written as <series>_<N>.dcm

Usage (run with the OCI venv, which also has pydicom/dotenv):
    PY=/root/ritikkumar/oci_env/bin/python

    $PY download_study_oci.py <uid|url>                 # single study, prune on
    $PY download_study_oci.py --csv studies.csv         # every study_iuid in CSV
    $PY download_study_oci.py <uid|url> -o /root/ritikkumar/disk_vdc
    $PY download_study_oci.py <uid|url> --no-prune      # download every series
    $PY download_study_oci.py <uid|url> --keep-bone     # keep bone-kernel recons
    $PY download_study_oci.py <uid|url> --min-slices 5
"""

import os
import re
import csv
import sys
import json
import argparse
import threading
import urllib.request
import concurrent.futures as futures
from collections import defaultdict

import pydicom
from pydicom import dcmread
from pydicom.multival import MultiValue
from dotenv import load_dotenv
import oci

# ---------------------------------------------------------------- config
ENV_PATH          = "/root/ritikkumar/ct_brain_classifier/data/pwd.env"
DATA_DOWNLOAD_DIR = "/root/ritikkumar/disk_vdc"
BUCKET_NAME       = "secure-dcm"
WORKERS           = 32
UID_COLUMN        = "study_iuid"

# --- pruning policy (overridable via CLI) ---
PRUNE              = True
DROP_BONE_KERNEL   = True
MIN_SLICES         = 5
DEFAULT_MIN_SLICES = 5

REFORMAT_KEYWORDS = ("reformat", "mpr", "mip", "sagittal", "coronal",
                     "sag ", "cor ", " sag", " cor", "recon ", "rebuild",
                     "average", "summary", "screen")
SCOUT_KEYWORDS = ("scout", "topogram", "surview", "localiz", "scanogram")
DOSE_KEYWORDS  = ("dose", "report")

# Display-window thresholds (fallback only, when ConvolutionKernel is absent)
BONE_WW_MIN, BONE_WC_MIN = 1300, 300
SUBDURAL_WW_MIN, SUBDURAL_WC_MIN = 150, 60

# Reconstruction-kernel hints (primary signal for soft vs sharp/bone)
KERNEL_BONE_WORDS = ("BONE", "SHARP", "EDGE", "LUNG", "DETAIL")
KERNEL_SOFT_WORDS = ("SOFT", "STANDARD", "SMOOTH", "STND", "STD")
KERNEL_SHARP_NUM = 60
KERNEL_SOFT_NUM  = 45

load_dotenv(ENV_PATH)
STUDY_PATH_API       = os.getenv("STUDY_PATH_API")
STUDY_PATH_API_TOKEN = os.getenv("STUDY_PATH_API_TOKEN")

# OCI clients are not thread-safe -> one per worker thread.
_cfg = oci.config.from_file()
_namespace = oci.object_storage.ObjectStorageClient(_cfg).get_namespace().data
_local = threading.local()


def get_client():
    c = getattr(_local, "client", None)
    if c is None:
        c = oci.object_storage.ObjectStorageClient(_cfg)
        _local.client = c
    return c


# ---------------------------------------------------------------- uid -> path
def extract_study_uid(uid_or_url):
    """Accept a bare StudyInstanceUID or a .../dicom/download/<uid> URL."""
    s = str(uid_or_url).strip().rstrip("/")
    return s.rsplit("/", 1)[-1] if "/" in s else s


def resolve_study_path(study_uid):
    """Resolve a StudyInstanceUID to its bucket storage path via the 5C API."""
    if not STUDY_PATH_API or not STUDY_PATH_API_TOKEN:
        raise RuntimeError("STUDY_PATH_API / STUDY_PATH_API_TOKEN missing in env")
    url = STUDY_PATH_API.rstrip("/") + "/" + study_uid
    req = urllib.request.Request(url, headers={"Authorization": STUDY_PATH_API_TOKEN})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    study_path = data.get("path")
    if not study_path:
        raise ValueError(f"No 'path' in API response for {study_uid}: {data}")
    return study_path


# ---------------------------------------------------------------- listing
def list_study_objects(study_path):
    """Return all object names under study_path, grouped by series."""
    prefix = study_path.rstrip("/") + "/"
    series = defaultdict(list)  # series_name -> [object_name, ...]
    client = get_client()
    nxt = None
    while True:
        page = client.list_objects(_namespace, BUCKET_NAME, prefix=prefix,
                                    fields="name", start=nxt)
        for o in page.data.objects:
            rel = o.name[len(prefix):].split("/")
            if len(rel) < 2 or not rel[0]:
                continue  # object directly under study root (no series) -> skip
            series[rel[0]].append(o.name)
        nxt = page.data.next_start_with
        if not nxt:
            break
    return series


# ---------------------------------------------------------------- classification
def _min_val(v):
    """Min of a (possibly MultiValue) numeric element; dual-preset WindowCenter
    like [450, 40] -> the brain preset is the lower value."""
    try:
        if isinstance(v, MultiValue) or (hasattr(v, "__iter__") and not isinstance(v, str)):
            vals = [float(x) for x in v]
            return min(vals) if vals else None
        return float(v)
    except Exception:
        return None


def find_plane(iop):
    """Axial/Coronal/Sagittal from ImageOrientationPatient (cross product)."""
    if not iop:
        return None
    try:
        r = [round(float(x)) for x in iop]
        a, b = r[0:3], r[3:6]
        n = [abs(a[1] * b[2] - a[2] * b[1]),
             abs(a[2] * b[0] - a[0] * b[2]),
             abs(a[0] * b[1] - a[1] * b[0])]
        if n[0] == 1:
            return "Sagittal"
        if n[1] == 1:
            return "Coronal"
        if n[2] == 1:
            return "Axial"
    except Exception:
        return None
    return None


def window_kind(wc, ww):
    wc, ww = _min_val(wc), _min_val(ww)
    if wc is None or ww is None:
        return "Unknown"
    if ww >= BONE_WW_MIN or wc >= BONE_WC_MIN:
        return "Bone"
    if ww >= SUBDURAL_WW_MIN or wc >= SUBDURAL_WC_MIN:
        return "Subdural"
    return "Brain"


def kernel_is_bone(ds):
    """soft(False)/bone-sharp(True)/unknown(None) from ConvolutionKernel.
    Siemens kernels arrive char-split (['H','3','0','s']) so join them."""
    parts = ds.get("ConvolutionKernel", [])
    if isinstance(parts, str):
        parts = [parts]
    k = "".join(str(x) for x in parts).upper()
    if not k:
        return None
    if any(w in k for w in KERNEL_BONE_WORDS):
        return True
    if any(w in k for w in KERNEL_SOFT_WORDS):
        return False
    nums = [int(x) for x in re.findall(r"(\d{2})", k)]
    if nums:
        m = max(nums)
        if m >= KERNEL_SHARP_NUM:
            return True
        if m <= KERNEL_SOFT_NUM:
            return False
    return None


def classify_header(ds, n_slices):
    """Base classification of a single series.
    Returns dict: keep_candidate(bool), reason, plane, window, is_bone, + meta."""
    modality = str(ds.get("Modality", ""))
    image_type = [str(x).upper() for x in ds.get("ImageType", [])]
    desc = str(ds.get("SeriesDescription", "")).strip()
    desc_l = desc.lower()
    plane = find_plane(ds.get("ImageOrientationPatient"))
    window = window_kind(ds.get("WindowCenter"), ds.get("WindowWidth"))
    kb = kernel_is_bone(ds)
    is_bone = kb if kb is not None else (window == "Bone")

    out = {"plane": plane or "No Plane", "window": window, "is_bone": is_bone,
           "modality": modality, "image_type": "/".join(image_type), "series_desc": desc}

    def res(keep, reason):
        out.update(keep_candidate=keep, reason=reason)
        return out

    if modality not in ("CT",):
        return res(False, f"non-CT modality ({modality})")
    if "LOCALIZER" in image_type or any(k in desc_l for k in SCOUT_KEYWORDS):
        return res(False, "localizer/scout")
    if "DOSE_INFO" in image_type or any(k in desc_l for k in DOSE_KEYWORDS):
        return res(False, "dose report")
    if "DERIVED" in image_type or "SECONDARY" in image_type:
        return res(False, "derived/secondary")
    if "REFORMATTED" in image_type or any(k in desc_l for k in REFORMAT_KEYWORDS):
        return res(False, "reformatted (image_type/desc)")
    if plane in ("Sagittal", "Coronal"):
        return res(False, f"non-axial reconstruction ({plane})")
    if n_slices < MIN_SLICES:
        return res(False, f"too few slices ({n_slices} < {MIN_SLICES})")
    return res(True, "primary axial CT brain")


# ---------------------------------------------------------------- dicom helpers
def anonymize_dataset(ds):
    for tag in ("PatientName", "InstitutionName", "PatientID", "OtherPatientIDs",
                "InstitutionAddress", "ReferringPhysicianName", "StudyDescription",
                "PerformedProcedureStepDescription"):
        if hasattr(ds, tag):
            setattr(ds, tag, "anonymized")
    return ds


def fetch_header(object_name):
    """Download one object and read its header only (no pixels)."""
    resp = get_client().get_object(_namespace, BUCKET_NAME, object_name)
    return dcmread(pydicom.filebase.DicomBytesIO(resp.data.content),
                   stop_before_pixels=True, force=True)


def fetch_slice(object_name):
    """Download one object; return (z_position_or_None, object_name, ds)."""
    try:
        resp = get_client().get_object(_namespace, BUCKET_NAME, object_name)
        ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(resp.data.content))
        z = None
        if hasattr(ds, "ImagePositionPatient"):
            try:
                z = float(ds.ImagePositionPatient[2])
            except (TypeError, ValueError, IndexError):
                z = None
        return z, object_name, ds
    except Exception as e:
        print(f"  ! failed {object_name}: {e}")
        return None, object_name, None


# ---------------------------------------------------------------- inspect + download
def inspect_series(series, object_names):
    """Read ONE header and base-classify, WITHOUT downloading the full series."""
    item = {"series": series, "object_names": object_names, "n_slices": len(object_names)}
    if not PRUNE:
        item.update(keep_candidate=True, reason="no-prune", plane="No Plane",
                    window="Unknown", is_bone=False, modality="", image_type="",
                    series_desc="")
        return item
    try:
        ds_hdr = fetch_header(object_names[0])
        item.update(classify_header(ds_hdr, len(object_names)))
    except Exception as e:
        print(f"  ~ classify failed for {series}: {e}; keeping to be safe")
        item.update(keep_candidate=True, reason="classify-error-kept",
                    plane="No Plane", window="Unknown", is_bone=False,
                    modality="", image_type="", series_desc="")
    return item


def download_series(series, object_names, series_dir):
    slices = []
    max_workers = min(WORKERS, len(object_names))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for z, name, ds in ex.map(fetch_slice, object_names):
            if ds is not None:
                slices.append((z, name, ds))
    if not slices:
        return 0

    # z-sorted first (ascending), then any slices lacking a z-position by name.
    with_z = sorted((s for s in slices if s[0] is not None), key=lambda s: s[0])
    no_z = sorted((s for s in slices if s[0] is None), key=lambda s: s[1])
    ordered = with_z + no_z

    os.makedirs(series_dir, exist_ok=True)
    for index, (_z, _name, ds) in enumerate(ordered, start=1):
        ds = anonymize_dataset(ds)
        ds.save_as(os.path.join(series_dir, f"{series}_{index}.dcm"))
    return len(ordered)


def select_keep_set(inspected):
    """Study-aware keep-set: drop junk + bone-kernel duplicates, never empty."""
    candidates = [it for it in inspected if it.get("keep_candidate")]
    junk = [it for it in inspected if not it.get("keep_candidate")]
    soft = [it for it in candidates if not it.get("is_bone")]

    keep_set, bone_fallback = [], False
    if not PRUNE:
        keep_set = candidates
    elif DROP_BONE_KERNEL:
        if soft:
            keep_set = soft                       # drop bone duplicates
            junk += [it for it in candidates if it.get("is_bone")]
        elif candidates:
            keep_set = candidates                 # bone-only -> keep (never empty)
            bone_fallback = True
    else:
        keep_set = candidates                     # --keep-bone

    for it in junk:
        if it.get("keep_candidate") and it.get("is_bone"):
            it["reason"] = "bone-kernel recon (duplicate)"
    return keep_set, junk, bone_fallback


# ---------------------------------------------------------------- entry
def download_study(uid_or_url, directory=DATA_DOWNLOAD_DIR, skip_existing=False):
    study_uid = extract_study_uid(uid_or_url)
    study_path = resolve_study_path(study_uid)
    study_id = study_path.rstrip("/").split("/")[-1]  # folder = last segment of study_path

    out_root = os.path.join(directory, study_id)
    if skip_existing and os.path.isdir(out_root) and os.listdir(out_root):
        print(f"skip {study_id} ({study_uid}) (already downloaded)", flush=True)
        return

    series_map = list_study_objects(study_path)
    if not series_map:
        print(f"No series found for {study_uid} (path {study_path}) in {BUCKET_NAME}")
        return
    print(f"{study_uid} -> {study_path} (id {study_id}): {len(series_map)} series "
          f"({sum(len(v) for v in series_map.values())} objects)  "
          f"prune={'ON' if PRUNE else 'OFF'} drop_bone={DROP_BONE_KERNEL and PRUNE}")

    # Pass 1: inspect every series (one header each, parallel)
    items = list(series_map.items())
    with futures.ThreadPoolExecutor(max_workers=min(WORKERS, len(items))) as ex:
        inspected = list(ex.map(lambda kv: inspect_series(kv[0], kv[1]), items))

    # Pass 2: study-level keep-set
    keep_set, junk, bone_fallback = select_keep_set(inspected)

    for it in junk:
        print(f"  prune {it['series']}: {it.get('reason','')} "
              f"(n={it['n_slices']}, {it.get('series_desc','')!r})")

    if not keep_set:
        print("Nothing to download after pruning.")
        return

    # Pass 3: download only the kept series
    total = 0
    for it in keep_set:
        series = it["series"]
        reason = "bone-kernel kept (only series)" if bone_fallback else it.get("reason", "")
        series_dir = os.path.join(out_root, series)
        n = download_series(series, it["object_names"], series_dir)
        total += n
        print(f"  keep  {series}: {n}/{it['n_slices']} slices [{reason}] -> {series_dir}")
    print(f"Done: {len(keep_set)} series, {total} slices under {out_root}")


def download_from_csv(csv_path, directory=DATA_DOWNLOAD_DIR, skip_existing=True, jobs=1):
    """Read StudyInstanceUIDs from the `study_iuid` column and download each.
    `jobs` studies are processed concurrently (each still downloads its own
    slices in parallel)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if UID_COLUMN not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} has no '{UID_COLUMN}' column; "
                             f"found {reader.fieldnames}")
        seen, uids = set(), []
        for r in reader:
            v = str(r.get(UID_COLUMN, "")).strip()
            if v and v not in seen:
                seen.add(v)
                uids.append(v)

    print(f"{len(uids)} unique study_iuid values in {csv_path} (jobs={jobs})", flush=True)

    done = {"n": 0}
    total = len(uids)

    def handle(uid):
        try:
            download_study(uid, directory, skip_existing=skip_existing)
        except Exception as e:
            print(f"  !! failed {extract_study_uid(uid)}: {e}", flush=True)

    if jobs <= 1:
        for uid in uids:
            done["n"] += 1
            handle(uid)
        return

    with futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(handle, uid) for uid in uids]
        for fut in futures.as_completed(futs):
            done["n"] += 1
            try:
                fut.result()
            except Exception as e:
                print(f"  !! study task error: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resolve StudyInstanceUID(s), prune, and download from OCI secure-dcm.")
    parser.add_argument("target", nargs="?",
                        help="A StudyInstanceUID or api.5cnetwork.com download URL")
    parser.add_argument("--csv", help=f"CSV file; downloads every UID in the '{UID_COLUMN}' column")
    parser.add_argument("-o", "--out-dir", default=DATA_DOWNLOAD_DIR)
    parser.add_argument("--no-prune", action="store_true", help="Download every series.")
    parser.add_argument("--keep-bone", action="store_true",
                        help="Keep bone-kernel recons too (default: drop as duplicates).")
    parser.add_argument("--min-slices", type=int, default=DEFAULT_MIN_SLICES)
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="(CSV mode) re-download studies whose out dir already exists.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="(CSV mode) number of studies to download concurrently.")
    args = parser.parse_args()

    if not args.target and not args.csv:
        parser.error("provide a StudyInstanceUID/URL or --csv <file>")

    PRUNE = not args.no_prune
    DROP_BONE_KERNEL = not args.keep_bone
    MIN_SLICES = args.min_slices

    if args.csv:
        download_from_csv(args.csv, args.out_dir, skip_existing=not args.no_skip_existing,
                          jobs=args.jobs)
    else:
        download_study(args.target, args.out_dir)
