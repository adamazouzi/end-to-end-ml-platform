"""Command-line workflows for dataset validation, analysis, and training."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from churn_platform.config import Settings
from churn_platform.data.ingestion import download_dataset, load_dataset
from churn_platform.data.validation import validate_dataset


def _load_validated(settings: Settings):
    frame = load_dataset(settings.data_path)
    report = validate_dataset(frame)
    logging.info("Dataset validation: %s", json.dumps(asdict(report), sort_keys=True))
    return frame


def main() -> None:
    """Run a project workflow command."""
    parser = argparse.ArgumentParser(prog="churn-platform")
    parser.add_argument("command", choices=["download-data", "validate", "eda", "train", "explain"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if args.command == "download-data":
        checksum = download_dataset(settings.data_path)
        print(json.dumps({"path": str(settings.data_path), "sha256": checksum}, indent=2))
        return
    frame = _load_validated(settings)
    if args.command == "validate":
        print(json.dumps(asdict(validate_dataset(frame)), indent=2))
    elif args.command == "eda":
        from churn_platform.data.eda import run_eda

        print(json.dumps(run_eda(frame, settings.reports_dir), indent=2))
    elif args.command == "train":
        from churn_platform.models.training import train_and_evaluate

        print(json.dumps(train_and_evaluate(frame, settings), indent=2))
    elif args.command == "explain":
        from churn_platform.models.explainability import explain_model

        result = explain_model(frame, settings.model_path, settings.reports_dir, seed=settings.random_seed)
        print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
