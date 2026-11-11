# Implementation roadmap

This plan defines the sequence used to build the Customer Churn Prediction Platform. A phase is complete only when its deliverables and acceptance checks are documented; no model metrics should be stated before they are measured.

**Current status:** Phases 0–9 implemented. Verification includes the measured training run, SHAP report generation, Pytest, Streamlit AppTest, API/dashboard workflow checks, and Docker image build. On 2026-11-11 the `api` and `dashboard` Compose services were confirmed healthy on host ports 8001 and 8501 and using the same `customer-churn-platform:latest` image ID. The image digest was `sha256:fbd3f85769e3966dfdda2d7d6aab853f8246710a60860c1dd1868d8ca301ed75` at verification time. A synthetic three-row batch scoring workflow was also validated. This does not establish production readiness.

## Phase 0 — Architecture and project foundation (complete)

**Deliverables**

- Establish the modular source, tests, data, model, report, notebook, and docs layout.
- Define Python 3.12 and the agreed dependency stack in `pyproject.toml`.
- Add repository conventions, Git/Docker ignore rules, README, and this roadmap.

**Acceptance**

- Project layout and responsibilities are documented.
- Local virtual environments and data/artifacts are excluded from Git.
- Python/dependency metadata parses and package installation can be attempted in the target environment.

## Phase 1 — Dataset acquisition, ingestion, and validation (complete)

**Deliverables**

- Confirm an authoritative/usable IBM Telco dataset source, license, checksum/version, and expected columns.
- Implement configurable local CSV ingestion without embedding a download credential or machine-specific path.
- Validate required columns, types, target labels, missing values, duplicate records, and basic domain constraints.
- Produce a concise data dictionary and validation report; preserve raw input unchanged.

**Acceptance**

- Valid sample data passes; malformed schemas and invalid target/domain values fail with actionable messages.
- Tests use synthetic fixtures and do not require the full dataset.

## Phase 2 — Exploratory data analysis (complete)

**Deliverables**

- Add a reproducible EDA command that profiles distributions, missingness, target balance, and key categorical/numeric summaries.
- Save reproducible plots and a JSON report; document data limitations and hypotheses.

**Acceptance**

- Analysis can be rerun from validated data and clearly distinguishes observation from interpretation.
- No unsupported causal claims are made.

## Phase 3 — Preprocessing and feature engineering (complete)

**Deliverables**

- Define target separation, categorical/numeric handling, missing-value strategy, and relevant feature derivations.
- Use a scikit-learn `Pipeline`/`ColumnTransformer` fitted only on training folds.
- Record feature and input schema requirements for inference.

**Acceptance**

- Transformations fit on training data only, handle expected inputs, and are serialized with the model.
- Tests cover missing/unknown categories and representative edge cases.

## Phase 4 — Baseline training and model comparison (complete)

**Deliverables**

- Establish stratified, reproducible train/validation/test splits and a majority-class/simple baseline.
- Compare a small, justified set of scikit-learn classifiers and XGBoost with consistent preprocessing and evaluation.
- Track parameters, random seeds, and dataset version in run outputs.

**Acceptance**

- Comparisons use a held-out test set only for final evaluation and avoid leakage.
- Report metrics actually produced by the run, including class-aware measures and threshold context.

## Phase 5 — Evaluation and explainability (complete)

**Deliverables**

- Assess discrimination, calibration/threshold tradeoffs, confusion matrix, and error patterns.
- Add a global SHAP feature contribution summary appropriate to the selected model and transformed feature space.
- Document limitations and intended decision-support use.

**Acceptance**

- Figures and reported values are generated from the saved experiment/model.
- Explanation outputs are checked against model inputs and do not imply causation.

## Phase 6 — FastAPI prediction service (complete)

**Deliverables**

- Create request/response schemas, health and prediction routes, startup model loading, and clear validation errors.
- Return a churn probability and predicted class; add explanation output only if it is practical and consistent with Phase 5.
- Document local run and API usage.

**Acceptance**

- Contract tests cover valid inputs, invalid inputs, missing artifacts, and prediction output shape.
- API uses the exact saved preprocessing/model pipeline.

## Phase 7 — Streamlit dashboard (complete)

**Deliverables**

- Build a clear dashboard for project context, model performance, cohort exploration, and individual prediction.
- Call shared package functionality or the documented API; avoid duplicating feature logic.
- Display explanatory limitations and model/data version where available.

**Acceptance**

- Main workflows work with representative inputs and gracefully report unavailable artifacts/API errors.
- UI never presents unmeasured or stale metrics as current results.

## Phase 8 — Automated tests and quality hardening (complete)

**Deliverables**

- Complete unit/integration coverage for validation, preprocessing, training helpers, API contract, and dashboard-facing logic where testable.
- Add reproducible local quality commands and a documented test strategy.

**Acceptance**

- Pytest suite passes without access to private data or external services.
- Failure modes are actionable and the project installs cleanly under Python 3.12.

## Phase 9 — Docker and release documentation (complete)

**Deliverables**

- Add a lean Docker image and clear service startup instructions, including model artifact expectations.
- Document architecture, data source/handling, setup, training, evaluation, API, dashboard, tests, and limitations in README.
- Review ignore rules and verify no credentials, private data, or generated artifacts are tracked.

**Acceptance**

- Container build and documented run path work in a clean environment once dependencies/artifacts are available.
- README contains only verified commands and measured results.
