"""Interactive customer churn operations dashboard."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from churn_platform.config import Settings

settings = Settings.from_env()
st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide")
st.html("""
<style>
.block-container {padding-top: 2rem; max-width: 1320px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
  background: linear-gradient(115deg, #102a43, #176b87); margin-bottom: 1.2rem;}
.hero p {color: #d7e8f3; margin-bottom: 0;}
.st-key-metric_kpi_precision, .st-key-metric_kpi_recall,
.st-key-metric_kpi_f1, .st-key-metric_kpi_roc_auc,
.st-key-metric_kpi_average_precision, .st-key-metric_dataset_customers,
.st-key-metric_dataset_churn_rate {
  background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 14px;
  padding: 0.5rem 0.8rem; color: #102a43;
}
.st-key-metric_kpi_precision *, .st-key-metric_kpi_recall *,
.st-key-metric_kpi_f1 *, .st-key-metric_kpi_roc_auc *,
.st-key-metric_kpi_average_precision *, .st-key-metric_dataset_customers *,
.st-key-metric_dataset_churn_rate * {color: #102a43 !important;}
</style>
""")
st.markdown(
    "<div class='hero'><h1>Customer Churn Intelligence</h1>"
    "<p>Explore the model, score a customer, or submit a small batch for prioritization.</p></div>",
    unsafe_allow_html=True,
)


def api_request(path: str, payload: dict | None = None):
    """Call the configured FastAPI service using the standard library."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        settings.api_url + path, data=data,
        headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = str(exc)
        return None, f"API returned {exc.code}: {detail}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"Could not reach the prediction API at {settings.api_url}: {exc}"


health, health_error = api_request("/health")
if health_error:
    st.warning(health_error)
elif not health.get("model_loaded"):
    st.warning("The API is running, but no trained model is loaded. Run the training workflow and restart the API.")
else:
    st.success("Prediction API is ready")

metrics_path = settings.reports_dir / "metrics.json"
summary_path = settings.reports_dir / "eda_summary.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None

overview, score_one, score_batch = st.tabs(["Model overview", "Score a customer", "Batch scoring"])

METRIC_HELP = {
    "Precision": "Among customers predicted to churn, the share who actually churned.",
    "Recall": "Among customers who churned, the share detected by the model.",
    "F1 score": "The harmonic mean of precision and recall; balances missed churners and false alarms.",
    "ROC-AUC": "How well the model ranks churners above non-churners across possible thresholds; 0.50 is no better than random ranking.",
    "Average precision": "A summary of precision across recall levels, with emphasis on the churn class.",
}


def chart(data: pd.DataFrame, spec: dict, *, key: str, alt: str) -> None:
    """Render an accessible, explicitly labelled Vega-Lite chart."""
    st.vega_lite_chart(data, spec, key=key, width="stretch", alt=alt)

with overview:
    st.subheader("Model performance")
    if not metrics:
        st.info("Model evaluation will appear here after `churn-platform train` completes.")
    else:
        selected = metrics["selected_test_metrics"]
        columns = st.columns(5)
        kpis = [
            ("Precision", "precision", "metric_kpi_precision"),
            ("Recall", "recall", "metric_kpi_recall"),
            ("F1 score", "f1", "metric_kpi_f1"),
            ("ROC-AUC", "roc_auc", "metric_kpi_roc_auc"),
            ("Average precision", "average_precision", "metric_kpi_average_precision"),
        ]
        for column, (label, metric_key, widget_key) in zip(columns, kpis):
            with column.container(key=widget_key, border=True):
                st.metric(label, f"{selected[metric_key]:.3f}", help=METRIC_HELP[label])
        st.caption(
            f"Selected {metrics['selected_model'].replace('_', ' ').title()} using 3-fold CV ROC-AUC; "
            f"holdout has {metrics['test_rows']:,} customers."
        )
        threshold = selected["threshold"]
        st.info(
            f"Classification threshold: **{threshold:.2f}**. A customer is labelled churn when the model's "
            f"estimated probability is at least {threshold:.2f}; a lower probability is labelled no churn. "
            "The threshold is a decision cut-off, not a change to the probability itself."
        )
        comparison = pd.DataFrame(metrics["models"])
        st.markdown("#### Candidate model comparison")
        chart_columns = ["model", "cv_roc_auc", "precision", "recall", "f1", "roc_auc"]
        st.dataframe(comparison[chart_columns].set_index("model").style.format("{:.3f}"), width="stretch")
        roc_data = comparison[["model", "cv_roc_auc", "roc_auc"]].melt(
            id_vars="model", var_name="Evaluation set", value_name="ROC-AUC"
        )
        roc_data["Evaluation set"] = roc_data["Evaluation set"].map(
            {"cv_roc_auc": "Cross-validation", "roc_auc": "Holdout test"}
        )
        st.markdown("#### ROC-AUC by model and evaluation set")
        chart(roc_data, {
            "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
            "encoding": {
                "x": {"field": "model", "type": "nominal", "title": "Model", "axis": {"labelAngle": -25}},
                "xOffset": {"field": "Evaluation set"},
                "y": {"field": "ROC-AUC", "type": "quantitative", "title": "ROC-AUC", "scale": {"domain": [0, 1]}, "axis": {"format": ".2f", "tickCount": 6}},
                "color": {"field": "Evaluation set", "type": "nominal", "title": "Evaluation set", "scale": {"range": ["#38bdf8", "#2dd4bf"]}},
                "tooltip": [
                    {"field": "model", "type": "nominal", "title": "Model"},
                    {"field": "Evaluation set", "type": "nominal"},
                    {"field": "ROC-AUC", "type": "quantitative", "format": ".3f"},
                ],
            },
            "height": 300,
        }, key="roc_auc_grouped", alt="Grouped bars compare cross-validation and holdout ROC-AUC by model, on a zero-to-one scale.")
        matrix = selected["confusion_matrix"]
        st.markdown("#### Holdout confusion matrix")
        st.dataframe(pd.DataFrame(matrix, index=["Actual no churn", "Actual churn"],
                                  columns=["Predicted no churn", "Predicted churn"]), width="stretch")
    if summary:
        st.markdown("#### Dataset profile")
        left, right = st.columns(2)
        with left.container(key="metric_dataset_customers", border=True):
            st.metric("Customers in source data", f"{summary['rows']:,}", help="Number of customer records in the source dataset.")
        with right.container(key="metric_dataset_churn_rate", border=True):
            st.metric("Observed churn rate", f"{summary['churn_rate']:.1%}", help="Share of source records labelled as churned; descriptive of this dataset.")
        chart_data = pd.DataFrame.from_dict(summary["churn_counts"], orient="index", columns=["Customers"])
        churn_chart = chart_data.rename_axis("Churn class").reset_index()
        chart(churn_chart, {
            "mark": {"type": "bar", "color": "#2dd4bf", "cornerRadiusEnd": 4},
            "encoding": {
                "x": {"field": "Churn class", "type": "nominal", "title": "Churn class"},
                "y": {"field": "Customers", "type": "quantitative", "title": "Customers", "axis": {"format": ",.0f", "tickCount": 5}},
                "tooltip": [{"field": "Churn class", "type": "nominal"}, {"field": "Customers", "type": "quantitative", "format": ",.0f"}],
            }, "height": 230,
        }, key="dataset_churn_counts", alt="Customer count by observed churn class; the vertical axis shows whole customers.")
        contract_counts = summary.get("categorical_counts", {}).get("Contract", {})
        if contract_counts:
            st.markdown("#### Contract mix")
            contract_data = pd.DataFrame({"Contract": list(contract_counts), "Customers": list(contract_counts.values())})
            chart(contract_data, {
                "mark": {"type": "bar", "color": "#38bdf8", "cornerRadiusEnd": 4},
                "encoding": {
                    "y": {"field": "Contract", "type": "nominal", "title": "Contract", "sort": "-x"},
                    "x": {"field": "Customers", "type": "quantitative", "title": "Customers", "axis": {"format": ",.0f"}},
                    "tooltip": [{"field": "Contract", "type": "nominal"}, {"field": "Customers", "type": "quantitative", "format": ",.0f"}],
                }, "height": 170,
            }, key="contract_mix", alt="Horizontal bars show the number of customers in each contract category.")
        contract_rates = summary.get("churn_rate_by", {}).get("Contract", {})
        if contract_rates:
            st.markdown("#### Observed churn rate by contract")
            rate_data = pd.DataFrame({"Contract": list(contract_rates), "Churn rate": list(contract_rates.values())})
            chart(rate_data, {
                "mark": {"type": "bar", "color": "#fb923c", "cornerRadiusEnd": 4},
                "encoding": {
                    "y": {"field": "Contract", "type": "nominal", "title": "Contract", "sort": "-x"},
                    "x": {"field": "Churn rate", "type": "quantitative", "title": "Observed churn rate", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%", "tickCount": 6}},
                    "tooltip": [{"field": "Contract", "type": "nominal"}, {"field": "Churn rate", "type": "quantitative", "format": ".1%"}],
                }, "height": 170,
            }, key="churn_rate_by_contract", alt="Horizontal bars show the observed churn rate for each contract, on a zero-to-one-hundred-percent scale.")
            st.caption("Descriptive rates in this sample; differences do not establish cause or predict intervention impact.")
    shap_path = settings.reports_dir / "shap_global_importance.csv"
    if shap_path.is_file():
        st.markdown("#### Global model drivers")
        shap_frame = pd.read_csv(shap_path).head(12).rename(columns={"mean_abs_shap": "Mean absolute SHAP value"})
        shap_value = next(column for column in shap_frame.columns if column != "feature")
        shap_frame = shap_frame.rename(columns={shap_value: "Mean absolute SHAP value"})
        chart(shap_frame, {
            "mark": {"type": "bar", "color": "#a78bfa", "cornerRadiusEnd": 4},
            "encoding": {
                "y": {"field": "feature", "type": "nominal", "title": "Feature", "sort": "-x"},
                "x": {"field": "Mean absolute SHAP value", "type": "quantitative", "title": "Mean absolute SHAP value", "axis": {"format": ".3f"}},
                "tooltip": [{"field": "feature", "type": "nominal", "title": "Feature"}, {"field": "Mean absolute SHAP value", "type": "quantitative", "format": ".4f"}],
            }, "height": 340,
        }, key="shap_importance", alt="Horizontal bars rank the twelve features by their mean absolute SHAP contribution.")
        st.caption("Mean absolute SHAP contribution summarizes model behavior; it does not establish cause.")
    else:
        st.info("Global SHAP feature importance appears here after `churn-platform explain` completes.")

with score_one:
    st.subheader("Individual customer prediction")
    st.caption("Values are sent to the configured FastAPI service. This is a demonstration score, not a retention decision.")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c3:
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
            total = st.number_input("Total charges", min_value=0.0, value=840.0, step=10.0)
        submitted = st.form_submit_button("Predict churn", type="primary")
    if submitted:
        record = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        result, error = api_request("/predict", record)
        if error:
            st.error(error)
        else:
            probability = result["churn_probability"]
            prediction = result["prediction"]
            threshold = result["threshold"]
            churn_predicted = prediction.lower() == "yes"
            st.markdown("#### Prediction result")
            probability_col, class_col = st.columns(2)
            with probability_col.container(border=True):
                st.metric("Estimated churn probability", f"{probability:.1%}", help="Model-estimated probability for this customer; not a guaranteed outcome.")
            with class_col.container(border=True):
                st.metric("Predicted class", "Churn" if churn_predicted else "No churn", help=f"Class assigned using the {threshold:.2f} probability threshold.")
            st.progress(probability, text=f"Churn probability: {probability:.1%}")
            if churn_predicted:
                st.warning(f"Interpretation: the estimated probability is at or above {threshold:.2f}, so the model classifies this customer as churn.")
            else:
                st.success(f"Interpretation: the estimated probability is below {threshold:.2f}, so the model classifies this customer as no churn.")

with score_batch:
    st.subheader("Batch predictions")
    st.caption("Upload up to 100 rows with the IBM Telco feature columns. Customer ID and Churn columns are optional and ignored.")
    uploaded = st.file_uploader("Customer feature CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded).drop(columns=["customerID", "Churn"], errors="ignore")
            if len(batch) > 100:
                st.error("The API accepts at most 100 customers per request.")
            elif st.button("Score uploaded customers", type="primary"):
                customers = batch.astype(object).where(pd.notna(batch), None).to_dict(orient="records")
                result, error = api_request("/predict/batch", {"customers": customers})
                if error:
                    st.error(error)
                else:
                    output = batch.copy()
                    output["prediction"] = [item["prediction"] for item in result["predictions"]]
                    output["churn_probability"] = [item["churn_probability"] for item in result["predictions"]]
                    st.dataframe(output, width="stretch")
                    st.download_button("Download scored CSV", output.to_csv(index=False), "churn_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

st.divider()
st.caption("Educational demonstration using a public IBM sample dataset. Probabilities may not generalize to any live customer population.")
