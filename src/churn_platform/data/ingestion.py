"""Download and load the IBM sample Telco churn dataset."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_dataset(destination: Path, *, timeout: int = 30) -> str:
    """Download the IBM-hosted CSV and return the downloaded file's SHA-256."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "churn-platform/0.1"})
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checksum = digest.hexdigest()
    LOGGER.info("Downloaded dataset to %s (sha256=%s)", destination, checksum)
    return checksum


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV from disk without mutating the raw file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `churn-platform download-data` first.")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV dataset at {path}: {exc}") from exc
    LOGGER.info("Loaded %d rows and %d columns from %s", *frame.shape, path)
    return frame
