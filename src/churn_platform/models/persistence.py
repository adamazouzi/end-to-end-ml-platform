"""Safe model loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

LOGGER = logging.getLogger(__name__)


def load_pipeline(path: Path):
    """Load a locally generated trusted joblib pipeline, with a helpful error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found at {path}. Run `churn-platform train` first.")
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        LOGGER.exception("Could not load model from %s", path)
        raise RuntimeError(f"Could not load model artifact at {path}: {exc}") from exc
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"Model artifact at {path} does not provide probability predictions.")
    return pipeline
