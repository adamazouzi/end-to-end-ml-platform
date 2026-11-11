"""FastAPI application for churn predictions."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn_platform.api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, CustomerFeatures, PredictionResponse,
)
from churn_platform.config import Settings
from churn_platform.models.persistence import load_pipeline

LOGGER = logging.getLogger(__name__)


def create_app(*, model_path: Path | None = None, pipeline=None) -> FastAPI:
    """Build an app instance; a pipeline may be injected for tests."""
    settings = Settings.from_env()
    resolved_model_path = Path(model_path or settings.model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            try:
                app.state.pipeline = load_pipeline(resolved_model_path)
            except Exception:
                LOGGER.exception("Model startup failed")
                app.state.pipeline = None
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        description="Predict churn probability from IBM Telco-style customer account features.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def predict_rows(request: Request, records: list[CustomerFeatures]) -> list[PredictionResponse]:
        active_pipeline = getattr(request.app.state, "pipeline", None)
        if active_pipeline is None:
            raise HTTPException(status_code=503, detail="Model is unavailable. Train a model and restart the API.")
        rows = pd.DataFrame([record.model_dump() for record in records])
        rows["TotalCharges"] = pd.to_numeric(rows["TotalCharges"], errors="coerce")
        try:
            probabilities = active_pipeline.predict_proba(rows)[:, 1]
        except Exception as exc:
            LOGGER.exception("Model prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed for the supplied customer data.") from exc
        return [
            PredictionResponse(
                prediction="Yes" if float(probability) >= 0.5 else "No",
                churn_probability=float(probability), threshold=0.5,
            )
            for probability in probabilities
        ]

    @app.get("/health")
    def health(request: Request):
        """Return process health and model readiness."""
        return {"status": "ok", "model_loaded": getattr(request.app.state, "pipeline", None) is not None}

    @app.get("/metrics")
    def metrics():
        """Expose the latest local training evaluation summary."""
        metrics_path = settings.reports_dir / "metrics.json"
        if not metrics_path.is_file():
            raise HTTPException(status_code=404, detail="Training metrics are not available yet.")
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.exception("Could not read metrics file")
            raise HTTPException(status_code=500, detail="Training metrics could not be read.") from exc

    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerFeatures, request: Request):
        """Predict churn for one customer."""
        return predict_rows(request, [customer])[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest, request: Request):
        """Predict churn for a bounded batch of customers."""
        return BatchPredictionResponse(predictions=predict_rows(request, payload.customers))

    return app


app = create_app()
