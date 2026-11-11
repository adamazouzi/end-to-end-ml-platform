"""Visualization runtime configuration shared by report generators."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib() -> None:
    """Use a writable temporary Matplotlib cache unless the user configured one."""
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "churn-platform-matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
