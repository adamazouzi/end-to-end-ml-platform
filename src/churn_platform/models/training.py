"""Cross-validated model comparison and final holdout evaluation."""

from __future__ import annotations

import json
import logging
import platform
import hashlib
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from churn_platform.config import Settings
from churn_platform.features.preprocessing import build_pipeline, split_features_target

LOGGER = logging.getLogger(__name__)


def _models(seed: int):
    """Create candidate estimators with compact parameter grids."""
    return {
        "majority_baseline": (
            DummyClassifier(strategy="prior"),
            {},
        ),
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            {"classifier__C": [0.5, 1.0, 2.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_estimators=250, n_jobs=-1, random_state=seed),
            {"classifier__max_depth": [None, 12], "classifier__min_samples_leaf": [1, 3]},
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                n_estimators=250, n_jobs=1, random_state=seed,
            ),
            {"classifier__max_depth": [3, 5], "classifier__learning_rate": [0.05, 0.1]},
        ),
    }


def _evaluate(y_true, predicted, probabilities) -> dict:
    """Calculate held-out classification metrics at the fixed 0.5 threshold."""
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "threshold": 0.5,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(frame: pd.DataFrame, settings: Settings | None = None) -> dict:
    """Tune candidate models on train folds, compare on holdout, and save best pipeline."""
    settings = settings or Settings.from_env()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.random_seed,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_seed)
    comparisons = []
    best_by_name = {}
    for name, (estimator, parameter_grid) in _models(settings.random_seed).items():
        LOGGER.info("Tuning %s", name)
        search = GridSearchCV(
            build_pipeline(estimator), parameter_grid, scoring="roc_auc", cv=cv,
            n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype("int8")
        metrics = _evaluate(y_test, predictions, probabilities)
        comparisons.append({
            "model": name,
            "cv_roc_auc": float(search.best_score_),
            "best_params": search.best_params_,
            **metrics,
        })
        best_by_name[name] = (pipeline, search, metrics)

    comparison = sorted(comparisons, key=lambda row: row["cv_roc_auc"], reverse=True)
    winner_name = comparison[0]["model"]
    winner_pipeline, winner_search, winner_metrics = best_by_name[winner_name]
    joblib.dump(winner_pipeline, settings.model_path)
    try:
        dataset_hash = hashlib.sha256(settings.data_path.read_bytes()).hexdigest()
    except OSError:
        dataset_hash = None
    package_versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost-cpu"]:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            if package == "xgboost-cpu":
                try:
                    package_versions["xgboost"] = version("xgboost")
                except PackageNotFoundError:
                    package_versions["xgboost"] = "unknown"
            else:
                package_versions[package] = "unknown"
    results = {
        "dataset": "IBM Telco Customer Churn",
        "dataset_sha256": dataset_hash,
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_seed": settings.random_seed,
        "cv_folds": 3,
        "selection_metric": "mean cross-validated ROC-AUC",
        "selected_model": winner_name,
        "selected_best_params": winner_search.best_params_,
        "selected_cv_roc_auc": float(winner_search.best_score_),
        "selected_test_metrics": winner_metrics,
        "models": comparison,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "model_file": settings.model_path.name,
    }
    (settings.reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).drop(columns=["confusion_matrix"]).to_csv(settings.reports_dir / "model_comparison.csv", index=False)
    _plot_evaluation(winner_metrics["confusion_matrix"], winner_name, settings.reports_dir)
    LOGGER.info("Selected %s; holdout ROC-AUC=%.4f", winner_name, winner_metrics["roc_auc"])
    return results


def _plot_evaluation(matrix: list[list[int]], model_name: str, output_dir: Path) -> None:
    """Save the selected model's holdout confusion matrix."""
    from churn_platform.visualization import configure_matplotlib

    configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    axis.set(title=f"{model_name.replace('_', ' ').title()} holdout confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
