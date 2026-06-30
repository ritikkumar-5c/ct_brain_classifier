"""
Download a single CT study from the YOTTA/E2E DICOM bucket given its
StudyInstanceUID (or the api.5cnetwork.com download URL).

The StudyInstanceUID is not the bucket prefix directly: it is first resolved
to a storage path via the 5C storage-path API, then every series under that
path is pulled, z-sorted, anonymized, and written to disk.

Usage (run with the ct_brain venv):
    PY=/root/ritikkumar/ct_brain/bin/python

    # by full URL
    $PY download_study_by_uid.py http://api.5cnetwork.com/dicom/download/1.2.840.113619.2.415.3.2831157761.783.1769483505.53

    # or by bare StudyInstanceUID
    $PY download_study_by_uid.py 1.2.840.113619.2.415.3.2831157761.783.1769483505.53

    # optional 2nd arg overrides the output directory
    $PY download_study_by_uid.py <uid|url> /root/ritikkumar/data
"""

import os
import sys
import csv
import json
import logging
import urllib.request
import concurrent.futures as futures
from io import BytesIO

import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.multival import MultiValue
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager

# ---------------------------------------------------------------- config
ENV_PATH          = "/root/ritikkumar/gcp/pwd.env"
DATA_DOWNLOAD_DIR = "/root/ritikkumar/data"
FAILED_CSV        = "/root/ritikkumar/failed_downloads.csv"
LOG_FILE          = "/root/ritikkumar/yotta_download.log"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s :: %(levelname)s :: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(ENV_PATH)

BUCKET_NAME          = os.getenv("YOTTA_BUCKET", "5cnetwork-newserver-dicom")
STUDY_PATH_API       = os.getenv("STUDY_PATH_API")
STUDY_PATH_API_TOKEN = os.getenv("STUDY_PATH_API_TOKEN")

http_client = PoolManager(maxsize=1024)


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
    logger.info(f"Resolved {study_uid} -> {study_path}")
    return study_path


# ---------------------------------------------------------------- bucket access
def list_objects(cloud, directory_path):
    """List objects under a prefix in the given cloud; return (objects, client)."""
    directory_path = str(directory_path).strip()
    if not directory_path:
        return None, None
    endpoint   = os.getenv(f"{cloud}_ENDPOINT")
    access_key = os.getenv(f"{cloud}_ACCESS_KEY")
    secret_key = os.getenv(f"{cloud}_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        logger.error(f"Missing credentials for {cloud}")
        return None, None
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key,
                       secure=True, http_client=http_client)
        objects = list(client.list_objects(BUCKET_NAME, prefix=directory_path,
                                            recursive=True))
        return objects, client
    except S3Error as err:
        logger.error(f"S3 error listing {directory_path} in {cloud}: {err}")
        return None, None
    except Exception as e:
        logger.error(f"Error listing {directory_path} in {cloud}: {e}")
        return None, None


def get_objects_and_client(directory_path):
    """Find the prefix in YOTTA (then E2E) and return its objects + client."""
    for cloud in ("YOTTA", "E2E"):
        objs, client = list_objects(cloud, directory_path)
        if objs and client:
            logger.info(f"{directory_path} found in {cloud} ({len(objs)} objects)")
            return objs, client
    logger.error(f"{directory_path} not present in YOTTA or E2E")
    return [], None


def discover_series(study_path):
    """Discover series sub-directories directly under a study path."""
    prefix = str(study_path).strip() + "/"
    objs, _ = get_objects_and_client(prefix)
    if not objs:
        return []
    seen, series_list = set(), []
    for obj in objs:
        if not obj.object_name or not obj.object_name.startswith(prefix):
            continue
        first = obj.object_name[len(prefix):].split("/")[0]
        if first and first not in seen:
            seen.add(first)
            series_list.append(first)
    return series_list


# ---------------------------------------------------------------- dicom helpers
def find_window(window_center, window_width):
    return "Bone" if abs(window_center - window_width) > 1000 else "Plain"


def get_window(object_name, client):
    response = client.get_object(BUCKET_NAME, object_name)
    dicom_data = BytesIO(response.read())
    response.close()
    response.release_conn()
    dicom_file = dcmread(dicom_data)
    window = "No Window"
    if getattr(dicom_file, "Modality", None) == "CT" \
            and getattr(dicom_file, "WindowCenter", None) is not None \
            and getattr(dicom_file, "WindowWidth", None) is not None:
        wc, ww = dicom_file.WindowCenter, dicom_file.WindowWidth
        if isinstance(wc, MultiValue):
            wc = wc[0]
        if isinstance(ww, MultiValue):
            ww = ww[0]
        window = find_window(wc, ww)
    return "No Plane", window


def anonymize_dataset(dataset):
    """Anonymize patient-specific information in the DICOM dataset."""
    for tag in ("PatientName", "InstitutionName", "PatientID", "OtherPatientIDs",
                "InstitutionAddress", "ReferringPhysicianName", "StudyDescription",
                "PerformedProcedureStepDescription"):
        if hasattr(dataset, tag):
            setattr(dataset, tag, "anonymized")
    return dataset


def process_single_dicom(obj, client):
    """Download one object and return (z_position, dicom_bytes)."""
    try:
        data = client.get_object(BUCKET_NAME, obj.object_name)
        dicom_bytes = data.read()
        data.close()
        data.release_conn()
        ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes))
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2]), dicom_bytes
        return None
    except Exception as e:
        logger.error(f"Error processing {obj.object_name}: {e}")
        return None


def download_series(study_path, series, study_dir):
    """Download every slice of one series, z-sorted + anonymized."""
    try:
        lists, client = get_objects_and_client(study_path + "/" + series + "/")
        if not client or not lists:
            return False

        try:
            _, window = get_window(lists[0].object_name, client)
        except Exception as e:
            logger.warning(f"window lookup failed for {study_path}/{series}: {e}")
            window = "No Window"
        with open(FAILED_CSV, "a", newline="") as f:
            csv.writer(f).writerow([study_path, series, "No Plane", window])

        dicom_files = []
        max_workers = min(32, len(lists))
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(lambda o: process_single_dicom(o, client), lists):
                if result:
                    dicom_files.append(result)

        dicom_files.sort(key=lambda x: x[0])
        os.makedirs(study_dir, exist_ok=True)
        for index, (_z, dicom_bytes) in enumerate(dicom_files, start=1):
            ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes))
            ds = anonymize_dataset(ds)
            ds.save_as(os.path.join(study_dir, f"{series}_{index}.dcm"))

        logger.info(f"Downloaded {len(dicom_files)} files for {study_path}/{series}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {study_path}/{series}: {e}")
        return False


# ---------------------------------------------------------------- entry
def download_study(uid_or_url, directory=DATA_DOWNLOAD_DIR):
    study_uid = extract_study_uid(uid_or_url)
    study_path = resolve_study_path(study_uid)
    series_list = discover_series(study_path)
    if not series_list:
        print(f"No series found for {study_uid} (path {study_path})")
        return
    print(f"{study_uid} -> {study_path}: {len(series_list)} series")
    for series in series_list:
        study_dir = os.path.join(directory, study_uid, str(series))
        ok = download_series(study_path, series, study_dir)
        print(f"  series {series}: {'OK' if ok else 'FAILED'} -> {study_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    target = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DATA_DOWNLOAD_DIR

    if not os.path.exists(FAILED_CSV):
        with open(FAILED_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["study_path", "series", "plane", "window"])

    download_study(target, out_dir)
