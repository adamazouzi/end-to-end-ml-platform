"""Schema and data quality checks for the IBM Telco churn dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_COLUMNS = (
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
)
ALLOWED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    },
}
REQUIRED_COLUMNS = (ID_COLUMN, *CATEGORICAL_COLUMNS[:4], "SeniorCitizen", *CATEGORICAL_COLUMNS[4:], "tenure", "MonthlyCharges", "TotalCharges", TARGET)
EXPECTED_COLUMNS = set(REQUIRED_COLUMNS)


@dataclass
class ValidationReport:
    """Summary of non-fatal data checks."""

    rows: int
    columns: int
    churn_rate: float
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    warnings: list[str] = field(default_factory=list)


def validate_dataset(frame: pd.DataFrame) -> ValidationReport:
    """Validate schema, key integrity, target labels, and known numeric domains."""
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")
    missing_columns = sorted(EXPECTED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    unexpected = sorted(set(frame.columns) - EXPECTED_COLUMNS)
    if unexpected:
        raise ValueError(f"Dataset has unexpected columns: {unexpected}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("customerID must be present for every row.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("customerID values must be unique.")
    target_values = set(frame[TARGET].dropna().unique())
    if frame[TARGET].isna().any() or not target_values.issubset({"Yes", "No"}) or target_values != {"Yes", "No"}:
        raise ValueError("Churn must contain both 'Yes' and 'No' and no missing or other values.")

    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(frame[column].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
        invalid_text = frame[column].notna() & frame[column].astype(str).str.strip().ne("") & numeric.isna()
        if invalid_text.any():
            raise ValueError(f"Column {column} contains non-numeric values.")
        if column != "TotalCharges" and numeric.isna().any():
            raise ValueError(f"Column {column} contains missing values.")
        if column == "SeniorCitizen" and not numeric.dropna().isin([0, 1]).all():
            raise ValueError("SeniorCitizen must contain only 0 or 1.")
        if column == "tenure" and not numeric.dropna().between(0, 72).all():
            raise ValueError("tenure must be between 0 and 72 months.")
        if column in {"MonthlyCharges", "TotalCharges"} and (numeric.dropna() < 0).any():
            raise ValueError(f"{column} must be non-negative.")

    for column in CATEGORICAL_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Column {column} contains missing values.")
        invalid_values = set(frame[column].unique()) - ALLOWED_CATEGORIES[column]
        if invalid_values:
            raise ValueError(f"Column {column} has unsupported values: {sorted(invalid_values)}")
    total_missing = pd.to_numeric(
        frame["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce",
    ).isna()
    if (total_missing & pd.to_numeric(frame["tenure"], errors="coerce").ne(0)).any():
        raise ValueError("Blank TotalCharges values are only expected for customers with zero tenure.")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        raise ValueError(f"Dataset contains {duplicate_rows} duplicate rows.")
    missing = frame.isna().sum()
    missing["TotalCharges"] = int(total_missing.sum())
    missing_values = {str(k): int(v) for k, v in missing[missing > 0].items()}
    warnings = []
    if missing_values.get("TotalCharges", 0):
        warnings.append("Blank TotalCharges values are expected for new customers and will be imputed by the model pipeline.")
    return ValidationReport(
        rows=len(frame), columns=len(frame.columns), churn_rate=float(frame[TARGET].eq("Yes").mean()),
        missing_values=missing_values, duplicate_rows=duplicate_rows, warnings=warnings,
    )
