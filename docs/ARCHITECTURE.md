# System Architecture

```mermaid
flowchart LR
    User[User] --> Streamlit[Streamlit Workbench]
    Streamlit --> PostgreSQL[(PostgreSQL)]
    Streamlit --> MLflow[MLflow Server]
    Streamlit --> Model[(Joblib Model)]
    Streamlit --> FastAPI[FastAPI Backend]
    FastAPI --> Model
    MLflow --> Artifacts[(MLflow Artifacts)]
```

## Components

- **Streamlit:** User interface for uploading datasets, EDA, cleaning, feature engineering, training, evaluation, explanation, SQL, and prediction.
- **FastAPI:** REST API for model metadata, single predictions, and batch predictions.
- **PostgreSQL:** Persistent storage for datasets and SQL analytics.
- **MLflow:** Experiment metadata and model/artifact tracking.
- **Joblib:** Serialization format for the latest trained model bundle.
- **Docker Compose:** Runs the application services in an isolated network.

