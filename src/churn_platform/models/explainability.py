"""SHAP global explanation artifact generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churn_platform.features.preprocessing import split_features_target
from churn_platform.models.persistence import load_pipeline

LOGGER = logging.getLogger(__name__)


def explain_model(frame: pd.DataFrame, model_path: Path, output_dir: Path, *, sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
    """Use SHAP to produce a global feature-importance summary and plot."""
    from churn_platform.visualization import configure_matplotlib

    configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(model_path)
    X, _ = split_features_target(frame)
    sample = X.sample(min(sample_size, len(X)), random_state=seed)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(sample)
    feature_names = list(preprocessor.get_feature_names_out())
    classifier = pipeline.named_steps["classifier"]
    masker = shap.maskers.Independent(transformed, max_samples=len(transformed))
    explainer = shap.Explainer(classifier, masker, feature_names=feature_names)
    values = explainer(transformed)
    contributions = np.asarray(values.values)
    if contributions.ndim == 3:
        contributions = contributions[:, :, 1]
    importance = np.abs(contributions).mean(axis=0)
    result = pd.DataFrame({"feature": feature_names, "mean_absolute_shap": importance}).sort_values(
        "mean_absolute_shap", ascending=False,
    )
    result.to_csv(output_dir / "shap_global_importance.csv", index=False)
    shap.summary_plot(contributions, transformed, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    LOGGER.info("SHAP global explanation saved to %s", output_dir)
    return result
