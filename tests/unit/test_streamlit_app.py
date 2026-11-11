"""Smoke tests for the Streamlit dashboard's key user-facing workflows."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_metrics_charts_and_scoring_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHURN_REPORTS_DIR", str(tmp_path))
    (tmp_path / "metrics.json").write_text(
        '{"selected_test_metrics":{"precision":0.6,"recall":0.5,"f1":0.55,'
        '"roc_auc":0.8,"average_precision":0.65,"threshold":0.5,'
        '"confusion_matrix":[[10,2],[3,5]]},"selected_model":"logistic_regression",'
        '"test_rows":20,"models":[{"model":"logistic_regression","cv_roc_auc":0.79,'
        '"precision":0.6,"recall":0.5,"f1":0.55,"roc_auc":0.8},'
        '{"model":"random_forest","cv_roc_auc":0.77,"precision":0.58,'
        '"recall":0.48,"f1":0.52,"roc_auc":0.78}]}',
        encoding="utf-8",
    )
    (tmp_path / "eda_summary.json").write_text(
        '{"rows":20,"churn_rate":0.25,"churn_counts":{"No":15,"Yes":5},'
        '"categorical_counts":{"Contract":{"Month-to-month":10,"One year":6,"Two year":4}},'
        '"churn_rate_by":{"Contract":{"Month-to-month":0.4,"One year":0.15,"Two year":0.0}}}',
        encoding="utf-8",
    )
    (tmp_path / "shap_global_importance.csv").write_text(
        "feature,mean_absolute_shap\nContract,0.12\ntenure,0.08\n", encoding="utf-8"
    )
    app_path = Path(__file__).resolve().parents[2] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Precision" in labels
    assert "Recall" in labels
    assert "F1 score" in labels
    assert "ROC-AUC" in labels
    assert "Average precision" in labels
    assert "Customers in source data" in labels
    assert "Observed churn rate" in labels
    # Streamlit's AppTest represents Vega-Lite charts as UnknownElement.
    assert sum(type(element).__name__ == "UnknownElement" for element in app) >= 4
    assert app.get_by_key("batch_csv")
    assert app.get_by_key("customer_form")
    assert any(button.label == "Predict churn" for button in app.button)
