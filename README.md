# Data Science Workbench

An end-to-end data science platform for dataset ingestion, analysis, preprocessing, model training, experiment tracking, explainability, and deployment.

## Features

- CSV, Excel, and JSON dataset loading
- Exploratory data analysis and visualizations
- Missing-value handling and feature engineering
- Classification and regression model training
- MLflow experiment tracking
- SHAP and LIME model explainability
- PostgreSQL-backed saved datasets and SQL queries
- FastAPI prediction and batch-prediction endpoints
- Streamlit, FastAPI, PostgreSQL, and MLflow Docker services

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
streamlit run app/app.py
```

For PostgreSQL and Docker, copy `.env.example` to `.env` and replace the
placeholder password with a strong local password before starting services.
If `.env` already exists, preserve it and add the settings from `.env.example`.
The local application otherwise uses SQLite by default.

Open:

- Streamlit: http://localhost:8501
- API documentation: http://localhost:8000/docs
- MLflow: http://localhost:5000

## Run with Docker

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and replace the placeholder password before continuing.
docker compose up --build
```

Services:

- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432

## API Usage

Model information:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/model-info"
```

Prediction:

```powershell
$body = @{
    features = @{
        area = 120
        rooms = 3
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

## Tests

```powershell
pytest
```

## Sample Dataset

The sample dataset is available at [datasets/sample_datasets/sample_regression.csv](datasets/sample_datasets/sample_regression.csv).

## Submission Materials

- [Architecture](docs/ARCHITECTURE.md)
- [Project report](docs/PROJECT_REPORT.md)
- [Viva questions](docs/VIVA_QUESTIONS.md)
- [Screenshot guide](screenshots/README.md)