"""Tests for input dataset schema and quality rules."""

import pytest

from churn_platform.data.validation import validate_dataset


def test_accepts_expected_schema_and_known_blank_totals(customer_frame):
    report = validate_dataset(customer_frame)
    assert report.rows == 80
    assert report.missing_values["TotalCharges"] > 0
    assert report.warnings


def test_rejects_missing_required_column(customer_frame):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(customer_frame.drop(columns="Contract"))


def test_rejects_invalid_target(customer_frame):
    customer_frame.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Churn must"):
        validate_dataset(customer_frame)


def test_rejects_duplicate_customer_id(customer_frame):
    customer_frame.loc[1, "customerID"] = customer_frame.loc[0, "customerID"]
    with pytest.raises(ValueError, match="customerID values must be unique"):
        validate_dataset(customer_frame)


def test_rejects_invalid_tenure(customer_frame):
    customer_frame.loc[0, "tenure"] = 90
    with pytest.raises(ValueError, match="tenure must"):
        validate_dataset(customer_frame)


def test_rejects_unknown_category(customer_frame):
    customer_frame.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(ValueError, match="unsupported values"):
        validate_dataset(customer_frame)


def test_rejects_blank_total_for_existing_customer(customer_frame):
    customer_frame.loc[1, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="zero tenure"):
        validate_dataset(customer_frame)
