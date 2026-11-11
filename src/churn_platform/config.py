"""Application configuration and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Environment-configurable locations and application settings."""

    data_path: Path
    model_path: Path
    reports_dir: Path
    api_url: str = "http://127.0.0.1:8000"
    random_seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with project-local defaults."""
        return cls(
            data_path=Path(os.getenv("CHURN_DATA_PATH", PROJECT_ROOT / "data/raw/Telco-Customer-Churn.csv")),
            model_path=Path(os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "models/churn_pipeline.joblib")),
            reports_dir=Path(os.getenv("CHURN_REPORTS_DIR", PROJECT_ROOT / "reports")),
            api_url=os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            random_seed=int(os.getenv("CHURN_RANDOM_SEED", "42")),
        )
