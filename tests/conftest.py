"""Small deterministic fixtures shared by the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def customer_frame():
    """Return valid synthetic Telco-shaped rows, including blank new-customer totals."""
    count = 80
    index = np.arange(count)
    frame = pd.DataFrame({
        "customerID": [f"TEST-{i:04d}" for i in index],
        "gender": np.where(index % 2, "Female", "Male"),
        "SeniorCitizen": (index % 5 == 0).astype(int),
        "Partner": np.where(index % 2, "Yes", "No"),
        "Dependents": np.where(index % 3 == 0, "Yes", "No"),
        "tenure": index % 73,
        "PhoneService": np.where(index % 4 == 0, "No", "Yes"),
        "MultipleLines": np.where(index % 4 == 0, "No phone service", np.where(index % 2, "Yes", "No")),
        "InternetService": np.where(index % 3 == 0, "DSL", np.where(index % 3 == 1, "Fiber optic", "No")),
        "OnlineSecurity": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "Yes", "No")),
        "OnlineBackup": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "No", "Yes")),
        "DeviceProtection": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "Yes", "No")),
        "TechSupport": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "No", "Yes")),
        "StreamingTV": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "Yes", "No")),
        "StreamingMovies": np.where(index % 3 == 2, "No internet service", np.where(index % 2, "No", "Yes")),
        "Contract": np.where(index % 3 == 0, "Month-to-month", np.where(index % 3 == 1, "One year", "Two year")),
        "PaperlessBilling": np.where(index % 2, "Yes", "No"),
        "PaymentMethod": np.where(index % 4 == 0, "Electronic check", np.where(index % 4 == 1, "Mailed check", np.where(index % 4 == 2, "Bank transfer (automatic)", "Credit card (automatic)"))),
        "MonthlyCharges": 20.0 + (index % 100),
        "TotalCharges": (20.0 + (index % 100)) * (index % 73),
        "Churn": np.where(index % 2, "Yes", "No"),
    })
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[frame["tenure"] == 0, "TotalCharges"] = " "
    return frame
