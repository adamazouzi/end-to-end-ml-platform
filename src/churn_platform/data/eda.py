"""Reproducible exploratory data analysis summaries and plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from churn_platform.visualization import configure_matplotlib

configure_matplotlib()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def run_eda(frame: pd.DataFrame, output_dir: Path) -> dict:
    """Write dataset summary and a small set of descriptive visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_values = {str(k): int(v) for k, v in frame.isna().sum().items() if v}
    blank_totals = int(pd.to_numeric(
        frame["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce",
    ).isna().sum())
    if blank_totals:
        missing_values["TotalCharges"] = blank_totals
    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "churn_counts": {str(k): int(v) for k, v in frame["Churn"].value_counts().items()},
        "churn_rate": float(frame["Churn"].eq("Yes").mean()),
        "missing_values": missing_values,
        "numeric_describe": json.loads(frame[["tenure", "MonthlyCharges", "TotalCharges"]].apply(pd.to_numeric, errors="coerce").describe().to_json()),
        "categorical_counts": {
            column: {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}
            for column in ["Contract", "InternetService", "PaymentMethod"]
        },
        "churn_rate_by": {
            column: {
                str(category): float(group["Churn"].eq("Yes").mean())
                for category, group in frame.groupby(column, sort=True, observed=True)
            }
            for column in ["Contract", "InternetService", "PaymentMethod"]
        },
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=frame, x="Churn", ax=axes[0], hue="Churn", legend=False, palette="Set2")
    axes[0].set_title("Customer churn classes")
    grouped_tenure = [frame.loc[frame["Churn"].eq(label), "tenure"].to_numpy() for label in ["No", "Yes"]]
    axes[1].boxplot(grouped_tenure)
    axes[1].set_xticks([1, 2], ["No churn", "Churn"])
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Tenure (months)")
    axes[1].set_title("Tenure by churn status")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    contract_rates = pd.Series(summary["churn_rate_by"]["Contract"]).sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    contract_rates.plot(kind="bar", ax=axis, color=sns.color_palette("Set2", len(contract_rates)))
    axis.set(title="Observed churn rate by contract", xlabel="Contract", ylabel="Share marked churned", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "churn_rate_by_contract.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("EDA summary and overview saved to %s", output_dir)
    return summary
