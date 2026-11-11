"""Tests for reproducible EDA summaries."""

import json

from churn_platform.data.eda import run_eda


def test_eda_writes_summary_and_plot(customer_frame, tmp_path):
    summary = run_eda(customer_frame, tmp_path)
    persisted = json.loads((tmp_path / "eda_summary.json").read_text())
    assert summary["rows"] == 80
    assert summary["missing_values"]["TotalCharges"] > 0
    assert set(summary["churn_rate_by"]["Contract"]) == {"Month-to-month", "One year", "Two year"}
    assert persisted["churn_counts"] == {"No": 40, "Yes": 40}
    assert (tmp_path / "eda_overview.png").is_file()
    assert (tmp_path / "churn_rate_by_contract.png").is_file()
