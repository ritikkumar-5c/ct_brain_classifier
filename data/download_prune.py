"""
Download + prune CT brain studies in one pass (study-aware).

For each study in data/csvs/ct_brain_{class}.csv:
  1. discover its series in the bucket
  2. read ONE DICOM header per series and classify it
  3. decide the keep-set at the STUDY level:
       - drop junk (localizer/scout, dose, secondary, derived/reformatted,
         non-axial, too-few-slices) entirely
       - drop bone-kernel reconstructions as duplicates (the brain/subdural/bone
         display channels are produced by windowing the soft-kernel series' HU
         values in preprocessing)
       - NEVER empty a study: if a study has only bone-kernel series, keep them
  4. download only the kept series

Bone vs soft is decided from the reconstruction kernel (ConvolutionKernel),
falling back to the display window only when the kernel tag is absent. Junk and
bone series are never downloaded, so they cost no bandwidth and no disk.

Output layout:   data/{class}/{study_id}/{series}/{series}_{n}.dcm
                 study_id = last segment of study_path

Usage:
    python3 download_prune.py normal
    python3 download_prune.py near_normal
    python3 download_prune.py abnormal
    python3 download_prune.py normal --keep-bone   # keep bone-kernel recons too
    python3 download_prune.py normal --no-prune     # download every series

Requires: pandas, minio, pydicom, numpy, tqdm, python-dotenv, urllib3
"""

import csv
import os
import re
import argparse
import logging
import concurrent.futures as futures
from io import BytesIO

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager
import pydicom
from pydicom import dcmread
from pydicom.multival import MultiValue

# ----------------------------------------------------------------------------- static config
BASE_DIR    = "/root/ritikkumar/"
CSV_DIR     = os.path.join(BASE_DIR, "data/csvs")
DATA_DIR    = os.path.join(BASE_DIR, "data")
LOG_DIR     = os.path.join(BASE_DIR, "gcp/logs")
env_path    = os.path.join(BASE_DIR, "gcp/pwd.env")
bucket_name = "5cnetwork-newserver-dicom"

CLASSES = {
    "normal":      "ct_brain_normal.csv",
    "near_normal": "ct_brain_near_normal.csv",
    "abnormal":    "ct_brain_abnormal.csv",
}

CLOUD_PRIORITY = ["YOTTA", "E2E"]
SKIP_EXISTING  = True

# Concurrency: studies processed in parallel x slices downloaded in parallel.
STUDY_WORKERS  = 6
SLICE_WORKERS  = 16

# --- pruning policy ---
PRUNE = True                     # --no-prune disables all pruning
DROP_BONE_KERNEL = True          # --keep-bone disables bone-kernel dropping
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

# Populated by configure()
input_csv_path = download_dir = series_csv = failed_csv = remaining_csv = skipped_csv = None
min_slices = DEFAULT_MIN_SLICES
logger = logging.getLogger(__name__)

load_dotenv(env_path)
http_client = PoolManager(maxsize=1024)
_client_cache = {}


def configure(cls):
    global input_csv_path, download_dir, series_csv, failed_csv, remaining_csv, skipped_csv
    input_csv_path = os.path.join(CSV_DIR, CLASSES[cls])
    download_dir   = os.path.join(DATA_DIR, cls)
    log_file       = os.path.join(LOG_DIR, f"{cls}_download.log")
    series_csv     = os.path.join(LOG_DIR, f"{cls}_series.csv")
    failed_csv     = os.path.join(LOG_DIR, f"{cls}_failed.csv")
    remaining_csv  = os.path.join(LOG_DIR, f"{cls}_remaining.csv")
    skipped_csv    = os.path.join(LOG_DIR, f"{cls}_pruned_series.csv")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s :: %(levelname)s :: %(message)s")

    if not os.path.exists(series_csv):
        with open(series_csv, "w", newline="") as f:
            csv.writer(f).writerow(["study_path", "series", "plane", "window", "is_bone", "n_slices"])
    if not os.path.exists(failed_csv):
        with open(failed_csv, "w", newline="") as f:
            csv.writer(f).writerow(["study_path", "series", "reason"])
    if not os.path.exists(skipped_csv):
        with open(skipped_csv, "w", newline="") as f:
            csv.writer(f).writerow(["study_path", "series", "reason", "n_slices",
                                    "modality", "plane", "window", "image_type", "series_desc"])


def _get_client(cloud):
    if cloud in _client_cache:
        return _client_cache[cloud]
    endpoint   = os.getenv(f"{cloud}_ENDPOINT")
    access_key = os.getenv(f"{cloud}_ACCESS_KEY")
    secret_key = os.getenv(f"{cloud}_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        logger.error(f"Missing credentials for {cloud}")
        _client_cache[cloud] = None
        return None
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key,
                   secure=True, http_client=http_client)
    _client_cache[cloud] = client
    return client


# ----------------------------------------------------------------------------- bucket helpers
def directory_exists(cloud, directory_path):
    try:
        if directory_path is None or not str(directory_path).strip():
            return False
        directory_path = str(directory_path).strip()
        client = _get_client(cloud)
        if client is None:
            return False
        for obj in client.list_objects(bucket_name, prefix=directory_path, recursive=True):
            if obj and getattr(obj, "object_name", None) and str(obj.object_name).startswith(directory_path):
                return True
        return False
    except S3Error as err:
        logger.error(f"S3 error checking directory in {cloud}: {err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in directory_exists ({cloud}, '{directory_path}'): {e}")
        return False


def list_objects(cloud, directory_path):
    try:
        if directory_path is None or not str(directory_path).strip():
            return None, None
        directory_path = str(directory_path).strip()
        client = _get_client(cloud)
        if client is None:
            return None, None
        objects = client.list_objects(bucket_name, prefix=directory_path, recursive=True)
        return objects, client
    except S3Error as err:
        logger.error(f"S3 error listing objects in {cloud}: {err}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error in list_objects ({cloud}, '{directory_path}'): {e}")
        return None, None


def get_objects_and_client(directory_path):
    if directory_path is None or not str(directory_path).strip():
        logger.error(f"Invalid directory_path: {directory_path}")
        return [], None
    directory_path = str(directory_path).strip()
    for cloud in CLOUD_PRIORITY:
        if directory_exists(cloud, directory_path):
            lists, session = list_objects(cloud, directory_path)
            if lists and session:
                return list(lists), session
    logger.error(f"{directory_path} not present in any bucket {CLOUD_PRIORITY}")
    return [], None


def discover_series(study_path):
    if not study_path or pd.isna(study_path):
        return [], None
    study_path = str(study_path).strip()
    prefix = study_path + "/"
    for cloud in CLOUD_PRIORITY:
        if not directory_exists(cloud, prefix):
            continue
        lists, session = list_objects(cloud, prefix)
        if not (lists and session):
            continue
        seen, series_list = set(), []
        for obj in lists:
            if not getattr(obj, "object_name", None):
                continue
            obj_name = str(obj.object_name)
            if not obj_name.startswith(prefix):
                continue
            first = obj_name[len(prefix):].split("/")[0]
            if first and first not in seen:
                seen.add(first)
                series_list.append(first)
        if series_list:
            logger.info(f"Found {len(series_list)} series in {study_path} from {cloud}")
            return series_list, session
    logger.warning(f"No series found for {study_path} in any bucket")
    return [], None


# ----------------------------------------------------------------------------- classification
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
    if not iop:
        return None
    try:
        r = [round(float(x)) for x in iop]
        n = [abs(x) for x in np.cross(r[0:3], r[3:6])]
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
    if n_slices < min_slices:
        return res(False, f"too few slices ({n_slices} < {min_slices})")
    return res(True, "primary axial CT brain")


def fetch_header(object_name, client):
    response = client.get_object(bucket_name, object_name)
    data = BytesIO(response.read())
    response.close()
    response.release_conn()
    return dcmread(data, stop_before_pixels=True, force=True)


# ----------------------------------------------------------------------------- download
def anonymize_dataset(dataset):
    for tag in ("PatientName", "InstitutionName", "PatientID", "OtherPatientIDs",
                "InstitutionAddress", "ReferringPhysicianName", "StudyDescription",
                "PerformedProcedureStepDescription"):
        if hasattr(dataset, tag):
            setattr(dataset, tag, "anonymized")
    return dataset


def process_single_dicom(obj, session):
    try:
        data = session.get_object(bucket_name, obj.object_name)
        dicom_bytes = data.read()
        data.close()
        data.release_conn()
        ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes))
        if hasattr(ds, "ImagePositionPatient"):
            return (float(ds.ImagePositionPatient[2]), dicom_bytes)
        return (0.0, dicom_bytes)
    except Exception as e:
        logger.error(f"Error processing object {obj.object_name}: {e}")
        return None


def download_one_series(study_path, series, study_dir, lists, session):
    """Download all slices of an already-listed series."""
    try:
        dicom_files = []
        with futures.ThreadPoolExecutor(max_workers=min(SLICE_WORKERS, len(lists))) as ex:
            fmap = {ex.submit(process_single_dicom, obj, session): obj for obj in lists}
            for fut in futures.as_completed(fmap):
                r = fut.result()
                if r:
                    dicom_files.append(r)
        if not dicom_files:
            with open(failed_csv, "a", newline="") as f:
                csv.writer(f).writerow([study_path, series, "no_dicom_read"])
            return False
        dicom_files.sort(key=lambda x: x[0])
        os.makedirs(study_dir, exist_ok=True)
        for index, (_z, b) in enumerate(dicom_files, start=1):
            ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(b))
            ds = anonymize_dataset(ds)
            ds.save_as(os.path.join(study_dir, f"{series}_{index}.dcm"))
        logger.info(f"Downloaded {len(dicom_files)} files for {study_path}/{series}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {study_path}/{series}: {e}")
        return False


def inspect_series(study_path, series):
    """List + read one header + base-classify, WITHOUT downloading.
    Returns dict with lists/session and classification, or None on failure."""
    lists, session = get_objects_and_client(study_path + "/" + series + "/")
    if not session or not lists:
        with open(failed_csv, "a", newline="") as f:
            csv.writer(f).writerow([study_path, series, "no_objects"])
        return None
    item = {"series": series, "lists": lists, "session": session, "n_slices": len(lists)}
    if PRUNE:
        try:
            ds_hdr = fetch_header(lists[0].object_name, session)
            item.update(classify_header(ds_hdr, len(lists)))
        except Exception as e:
            logger.warning(f"Classify failed for {study_path}/{series}: {e}; keeping to be safe")
            item.update(keep_candidate=True, reason="classify-error-kept",
                        plane="No Plane", window="Unknown", is_bone=False,
                        modality="", image_type="", series_desc="")
    else:
        item.update(keep_candidate=True, reason="no-prune", plane="No Plane",
                    window="Unknown", is_bone=False, modality="", image_type="", series_desc="")
    return item


def process_study(study_path, study_id, directory):
    """Inspect all series of a study, decide the keep-set (study-aware), download it."""
    try:
        series_list, _ = discover_series(study_path)
        if not series_list:
            return "no_series"

        # Pass 1: inspect every series (parallel, no download)
        inspected = []
        with futures.ThreadPoolExecutor(max_workers=min(SLICE_WORKERS, len(series_list))) as ex:
            for it in ex.map(lambda s: inspect_series(study_path, s), series_list):
                if it is not None:
                    inspected.append(it)
        if not inspected:
            return "no_series"

        # Pass 2: study-level keep-set
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

        # log pruned (junk + dropped bone)
        for it in junk:
            reason = it.get("reason", "")
            if it.get("keep_candidate") and it.get("is_bone"):
                reason = "bone-kernel recon (duplicate)"
            with open(skipped_csv, "a", newline="") as f:
                csv.writer(f).writerow([study_path, it["series"], reason, it.get("n_slices", ""),
                                        it.get("modality", ""), it.get("plane", ""),
                                        it.get("window", ""), it.get("image_type", ""),
                                        it.get("series_desc", "")])
            logger.info(f"Pruned (not downloaded) {study_path}/{it['series']}: {reason}")

        # Pass 3: download the keep-set
        for it in keep_set:
            reason = it.get("reason", "")
            if bone_fallback:
                reason = "bone-kernel kept (only series in study)"
            with open(series_csv, "a", newline="") as f:
                csv.writer(f).writerow([study_path, it["series"], it.get("plane", ""),
                                        it.get("window", ""), it.get("is_bone", ""), it["n_slices"]])
            study_dir = os.path.join(directory, study_id, str(it["series"]))
            download_one_series(study_path, it["series"], study_dir, it["lists"], it["session"])
        return "ok"
    except Exception as e:
        logger.error(f"Error processing study {study_path}: {e}")
        return "error"


# ----------------------------------------------------------------------------- main
def main(file, directory):
    studies = pd.read_csv(file).reset_index(drop=True)
    print(f"Loaded {studies.shape[0]} studies from {file}")
    print(f"Pruning: {'ON' if PRUNE else 'OFF'}  drop_bone_kernel={DROP_BONE_KERNEL and PRUNE}")

    completed = []
    try:
        with tqdm(total=len(studies), unit="study") as pbar:
            with futures.ThreadPoolExecutor(max_workers=STUDY_WORKERS) as executor:
                fut_to_idx = {}
                for i, row in studies.iterrows():
                    study_path = row.get("study_path")
                    if study_path is None or pd.isna(study_path) or not str(study_path).strip():
                        completed.append(i); pbar.update(); continue
                    study_path = str(study_path).strip()
                    study_id = study_path.rstrip("/").split("/")[-1]

                    out_dir = os.path.join(directory, study_id)
                    if SKIP_EXISTING and os.path.isdir(out_dir) and os.listdir(out_dir):
                        completed.append(i); pbar.update(); continue

                    fut = executor.submit(process_study, study_path, study_id, directory)

                    def make_cb(idx):
                        def cb(f):
                            pbar.update()
                            completed.append(idx)
                        return cb
                    fut.add_done_callback(make_cb(i))
                    fut_to_idx[fut] = i

                futures.wait(list(fut_to_idx.keys()))

        remaining = studies.drop(studies.index[sorted(set(completed))])
        remaining.to_csv(remaining_csv, index=False)
        print(f"Done. {len(set(completed))} studies handled; {len(remaining)} remaining -> {remaining_csv}")
    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted! Saving remaining studies...")
        remaining = studies.drop(studies.index[sorted(set(completed))])
        remaining.to_csv(remaining_csv, index=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download + prune CT brain studies (study-aware).")
    parser.add_argument("klass", choices=list(CLASSES.keys()))
    parser.add_argument("--no-prune", action="store_true", help="Download every series.")
    parser.add_argument("--keep-bone", action="store_true",
                        help="Keep bone-kernel recons too (default: drop them as duplicates).")
    parser.add_argument("--min-slices", type=int, default=DEFAULT_MIN_SLICES)
    args = parser.parse_args()

    PRUNE = not args.no_prune
    DROP_BONE_KERNEL = not args.keep_bone
    min_slices = args.min_slices
    configure(args.klass)
    main(input_csv_path, download_dir)
