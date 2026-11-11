# Project development instructions

## Project goal

Build a modular customer churn prediction platform around the IBM Telco Customer Churn dataset. The platform will cover data ingestion and validation, EDA, preprocessing and feature engineering, model comparison, evaluation and SHAP explainability, a FastAPI prediction service, a Streamlit dashboard, automated tests, and Docker packaging.

## Environment and dependencies

- Target Python 3.12 (`>=3.12,<3.13`).
- Declare runtime and development dependencies in `pyproject.toml`; keep dependency changes intentional and documented.
- Use only the agreed project stack unless a change is explained in the README and reviewed before adoption.
- Do not commit credentials, `.env` files, local virtual environments, raw customer data, or generated model artifacts.

## Code conventions

- Keep production code under `src/churn_platform/`; keep the Streamlit entry point at the repository root in `streamlit_app.py`.
- Separate data access/validation, feature preparation, training/evaluation, explanation, API, and dashboard concerns.
- Prefer small, typed functions with clear inputs and outputs. Add docstrings to public modules and functions where they clarify behavior.
- Keep preprocessing fitted on training data and serialize the complete fitted preprocessing/model pipeline so inference uses the same transformations.
- Make randomness reproducible with explicit random seeds and record model/data configuration used for runs.
- Do not hard-code local paths, credentials, or measured model results. Never report metrics that have not been produced by an actual run.
- Keep data, reports, and model artifacts out of source control; commit only empty directory markers where needed.

## Testing and changes

- Add focused Pytest coverage for data validation, preprocessing, training/evaluation behavior, and API contracts as those components are implemented.
- Keep tests deterministic and use small synthetic fixtures; do not require the private/local dataset to run unit tests.
- Update the README and roadmap when project capabilities or phase status changes.
- Before claiming verification, state the exact checks run and their outcome. Do not add unrequested test suites or claim checks were run when they were not.

## Initial architecture

- `src/churn_platform/data/`: dataset loading and schema/quality validation.
- `src/churn_platform/features/`: reusable preprocessing and feature engineering.
- `src/churn_platform/models/`: training, model comparison, evaluation, and explainability.
- `src/churn_platform/api/`: FastAPI app and request/response schemas.
- `streamlit_app.py`: interactive dashboard entry point, calling shared package code/API.
- `tests/`: unit and API contract tests.
- `notebooks/`: exploratory analysis and documented experiments; reusable production logic belongs in `src/`.
- `data/`, `models/`, and `reports/`: local/generated artifacts, excluded from version control except directory markers.
- `docs/`: architecture and phase roadmap.
