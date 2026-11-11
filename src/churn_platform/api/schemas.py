"""Typed FastAPI request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    """Customer features accepted by the prediction service."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def blank_total_charges_are_missing(cls, value):
        """Map the source CSV's whitespace-only new-customer totals to missing."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PredictionResponse(BaseModel):
    """Churn prediction response for one customer."""

    prediction: Literal["No", "Yes"]
    churn_probability: float = Field(ge=0, le=1)
    threshold: float


class BatchPredictionRequest(BaseModel):
    """A bounded set of customer rows for batch scoring."""

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Predictions in the same order as the input customer list."""

    predictions: list[PredictionResponse]
