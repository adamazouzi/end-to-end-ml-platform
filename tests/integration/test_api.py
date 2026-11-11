"""FastAPI contract integration tests with a locally fitted synthetic pipeline."""

from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn_platform.api.main import create_app
from churn_platform.config import Settings
from churn_platform.features.preprocessing import build_pipeline, split_features_target


def _ready_client(customer_frame):
    X, y = split_features_target(customer_frame)
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    app = create_app(pipeline=pipeline)
    return TestClient(app)


def _payload(customer_frame):
    return customer_frame.iloc[0].drop(labels=["customerID", "Churn"]).to_dict()


def test_health_and_single_prediction_contract(customer_frame):
    with _ready_client(customer_frame) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=_payload(customer_frame))
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] in {"Yes", "No"}
    assert 0 <= prediction.json()["churn_probability"] <= 1


def test_batch_prediction_contract(customer_frame):
    with _ready_client(customer_frame) as client:
        response = client.post("/predict/batch", json={"customers": [_payload(customer_frame)] * 2})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_batch_predictions_match_local_pipeline_for_distinct_customers(customer_frame):
    X, y = split_features_target(customer_frame)
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    indices = [0, 17, 63]
    rows = [customer_frame.iloc[index].drop(labels=["customerID", "Churn"]).to_dict() for index in indices]
    expected = pipeline.predict_proba(X.iloc[indices])[:, 1]

    with TestClient(create_app(pipeline=pipeline)) as client:
        response = client.post("/predict/batch", json={"customers": rows})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == len(indices)
    assert [item["churn_probability"] for item in predictions] == expected.tolist()
    assert all(item["prediction"] == ("Yes" if item["churn_probability"] >= 0.5 else "No") for item in predictions)


def test_metrics_endpoint_returns_report_from_configured_directory(tmp_path, monkeypatch):
    expected = {"selected_model": "test_model", "test_rows": 3}
    (tmp_path / "metrics.json").write_text('{"selected_model":"test_model","test_rows":3}', encoding="utf-8")
    settings = Settings(data_path=tmp_path / "data.csv", model_path=tmp_path / "model.joblib", reports_dir=tmp_path)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))

    with TestClient(create_app(pipeline=object())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == expected


def test_invalid_request_is_rejected(customer_frame):
    payload = _payload(customer_frame)
    payload["tenure"] = 99
    with _ready_client(customer_frame) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_unavailable_returns_service_unavailable(customer_frame):
    payload = _payload(customer_frame)
    with TestClient(create_app(model_path="/no/such/model.joblib")) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = client.post("/predict", json=payload)
    assert response.status_code == 503
