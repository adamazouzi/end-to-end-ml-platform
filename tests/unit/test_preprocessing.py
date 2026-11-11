"""Tests for the shared feature transformation pipeline."""

from sklearn.linear_model import LogisticRegression

from churn_platform.features.preprocessing import build_pipeline, split_features_target


def test_preprocessor_fits_and_ignores_unseen_categories(customer_frame):
    features, target = split_features_target(customer_frame)
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(features, target)
    unseen = features.iloc[[0]].copy()
    unseen["gender"] = "Unseen category"
    result = pipeline.predict_proba(unseen)
    assert result.shape == (1, 2)
    assert 0 <= result[0, 1] <= 1


def test_blank_total_charges_become_missing_for_train_fitted_imputer(customer_frame):
    features, _ = split_features_target(customer_frame)
    assert features["TotalCharges"].isna().sum() > 0
