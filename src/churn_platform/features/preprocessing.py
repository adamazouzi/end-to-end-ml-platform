"""Reusable input cleaning and model preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.data.validation import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET

FEATURE_COLUMNS = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and a 0/1 target, excluding the identifier and label."""
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    target = frame[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build train-fitted imputing, scaling, and one-hot encoding transformations."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_COLUMNS)), ("categorical", categorical, list(CATEGORICAL_COLUMNS))],
        remainder="drop", verbose_feature_names_out=False,
    )


def build_pipeline(classifier) -> Pipeline:
    """Combine shared preprocessing and a supplied classifier into one pipeline."""
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
