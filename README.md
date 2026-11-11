# Customer Churn Intelligence

An end-to-end machine learning demonstration that profiles IBM's Telco Customer Churn sample, compares churn classifiers, and serves predictions through FastAPI and Streamlit. It includes data validation, exploratory analysis, train-fitted preprocessing, cross-validated model selection, evaluation, SHAP feature summaries, individual and CSV batch scoring, automated tests, and Docker Compose.

This is an educational demonstration. It is not a validated customer-retention or production decision system.

## Dashboard preview

No dashboard screenshots are included yet. Add genuine screenshots made with synthetic inputs at:

- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/batch-scoring.png`

## Architecture and technology

```text
IBM-hosted CSV / local CSV
        │
        ▼
ingestion and schema/domain validation
        ├──► EDA summaries and plots
        ▼
train/test split ──► preprocessing fitted inside cross-validation folds
        ▼
Logistic Regression / Random Forest / XGBoost + majority baseline
        ▼
held-out metrics and SHAP global feature importance
        ▼
serialized preprocessing + model pipeline
        ├──► FastAPI health, metrics, individual and batch prediction
        └──► Streamlit model overview and customer scoring dashboard
```

The project uses Python 3.12, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, SHAP, FastAPI, Uvicorn, Streamlit, Pytest and Docker Compose. On Linux the image uses XGBoost's CPU-only package; other supported platforms use the regular XGBoost package. The API loads the same serialized pipeline produced by training. The dashboard calls the API for predictions and reads generated local reports for charts.

## Prerequisites and local installation

- Python 3.12
- `uv` 0.13.7
- Git

Install the pinned `uv` version, then install exactly the versions and hashes in `uv.lock` from the repository root:

```bash
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
uv sync --locked --extra dev
```

The raw dataset, trained model and generated reports are local artifacts and are not included in the repository. Download the IBM CSV and prepare a model before starting prediction services:

```bash
uv run churn-platform download-data
uv run churn-platform validate
uv run churn-platform eda
uv run churn-platform train
uv run churn-platform explain
```

The default paths are `data/raw/Telco-Customer-Churn.csv`, `models/churn_pipeline.joblib` and `reports/`. `download-data` needs network access to the IBM-hosted CSV and prints the fetched file's SHA-256.

## Run locally

Start the API in one terminal:

```bash
uv run uvicorn churn_platform.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start Streamlit in another terminal:

```bash
CHURN_API_URL=http://127.0.0.1:8000 uv run streamlit run streamlit_app.py
```

Open the dashboard at `http://localhost:8501` and the interactive API docs at `http://127.0.0.1:8000/docs`. The API provides `GET /health`, `GET /metrics`, `POST /predict` and `POST /predict/batch`.

### Docker Compose

Requirements: Docker Engine and the Compose plugin. Build the image, download and prepare the dataset through the trainer service, then start the API and dashboard:

```bash
docker compose build
docker compose --profile training run --rm trainer download-data
docker compose --profile training run --rm trainer validate
docker compose --profile training run --rm trainer eda
docker compose --profile training run --rm trainer train
docker compose --profile training run --rm trainer explain
docker compose up -d api dashboard
```

The dashboard is at `http://localhost:8501`; the Compose API is at `http://localhost:8001/docs`. The API listens on port 8000 inside the container and is published on host port 8001. Both ports bind to loopback. Compose mounts `data/`, `models/` and `reports/` from the local workspace; the raw dataset and model are not baked into the image. Stop only these services with `docker compose stop api dashboard`.

## Use the prediction system

### Individual prediction

The dashboard's **Score a customer** form submits a customer to FastAPI and displays the predicted class, churn probability and interpretation. The API expects all 19 feature fields. For example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender":"Female", "SeniorCitizen":0, "Partner":"No", "Dependents":"No",
    "tenure":12, "PhoneService":"Yes", "MultipleLines":"No",
    "InternetService":"DSL", "OnlineSecurity":"No", "OnlineBackup":"Yes",
    "DeviceProtection":"No", "TechSupport":"No", "StreamingTV":"No",
    "StreamingMovies":"No", "Contract":"Month-to-month", "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check", "MonthlyCharges":70.0, "TotalCharges":840.0
  }'
```

For a local Compose API, use `http://127.0.0.1:8001/predict` instead.

The default classification threshold is **0.50**: probabilities at or above 0.50 are labelled `Yes` (churn); lower probabilities are labelled `No`. This is a configurable business decision point in concept, not an optimized threshold. The appropriate threshold depends on intervention costs and capacity.

### CSV batch scoring

In **Batch scoring**, upload a CSV containing the 19 Telco feature columns documented in [docs/data.md](docs/data.md). `customerID` and `Churn` are optional and ignored. A blank `TotalCharges` is accepted for a zero-tenure customer. The API accepts 1–100 rows per request. The dashboard displays each predicted class and churn probability and offers a downloadable result CSV.

Example synthetic row (no personal data):

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
SYN-001,Female,0,No,No,0,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,,Yes
```

## Training, results and tests

Training uses a reproducible stratified 80/20 split, a majority-class baseline, and 3-fold stratified cross-validation on the training partition. Preprocessing is fitted within each fold. The selected estimator is the candidate with the highest mean CV ROC-AUC; the held-out test set is used for final reporting, not model selection. The `explain` command generates global SHAP feature-importance output for the saved pipeline.

The following results were measured on the downloaded CSV with SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, random seed 42, and a 1,409-row stratified holdout. Test precision, recall and F1 use the 0.50 threshold; ROC-AUC and average precision use prediction scores.

| Model (best CV parameters) | CV ROC-AUC | Test precision | Test recall | Test F1 | Test ROC-AUC | Test average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline (prior) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |
| Logistic Regression (`C=2.0`) | 0.8458 | 0.5060 | 0.7861 | 0.6157 | 0.8413 | 0.6304 |
| Random Forest (`max_depth=12`, `min_samples_leaf=3`) | 0.8423 | 0.5342 | 0.7513 | 0.6244 | 0.8379 | 0.6472 |
| **XGBoost** (`max_depth=3`, `learning_rate=0.05`) | **0.8472** | **0.6523** | 0.5267 | 0.5828 | **0.8441** | **0.6596** |

XGBoost was selected by mean CV ROC-AUC. Its holdout confusion matrix (actual rows: no churn, churn; predicted columns: no churn, churn) was `[[930, 105], [177, 197]]`; accuracy was 0.7999. On this single split, Logistic Regression and Random Forest had higher churn recall and F1 at the same threshold. These measured sample results are not estimates of production performance.

Run automated tests from the project root:

```bash
uv run pytest -q
```

The tests use synthetic fixtures and do not require the IBM dataset or network access. The latest recorded run completed with 18 passing tests; it emitted one Starlette/httpx deprecation warning from `starlette.testclient`.

## Dataset provenance and rights

The CSV is sourced from IBM's [archived telco-customer-churn-on-icp4d repository](https://github.com/IBM/telco-customer-churn-on-icp4d), which includes the [dataset file](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) and describes it as a Telco customer churn dataset. IBM's README applies Apache-2.0 to its code pattern but does not clearly state a distinct license grant for the CSV. This project does not redistribute the raw dataset; do not infer data rights from the code-pattern license. See [docs/data.md](docs/data.md) and verify upstream terms before redistribution or commercial use.

This project itself has **no `LICENSE` file**. A source-code license has not been selected. The model and reports are generated locally and excluded from Git alongside the raw dataset.

## Limitations

- This is an educational demonstration using a small Telco churn sample; representativeness for current customers or other markets has not been established.
- Scores are measured on one stratified holdout. There is no external validation or evidence of production performance.
- The 0.50 threshold is a default, not a cost-optimized decision threshold.
- SHAP values describe model behavior; they do not establish causal effects.
- The application has no authentication, rate limiting, monitoring, drift detection or production data governance.
- The dataset's separate redistribution rights and this project's source-code license remain unresolved.

## Repository structure and documentation

```text
src/churn_platform/   Data ingestion, validation, features, models, API and CLI
streamlit_app.py      Streamlit dashboard
tests/                Unit and API integration tests with synthetic fixtures
docs/data.md          Dataset fields, provenance and rights notes
docs/roadmap.md       Implementation phases and verification status
notebooks/            Reserved for exploratory notebooks
data/                 Local data paths; raw/processed content is ignored
models/               Local serialized pipeline; ignored
reports/              Local metrics and plots; ignored
```

See [docs/data.md](docs/data.md) for the input schema and source notes, [docs/roadmap.md](docs/roadmap.md) for the project phases, and [AGENTS.md](AGENTS.md) for development conventions.
