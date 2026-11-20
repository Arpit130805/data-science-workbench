# Viva Questions and Suggested Answers

## 1. What is this project?

It is an end-to-end data science workbench for data preparation, machine-learning training, evaluation, experiment tracking, explainability, and prediction deployment.

## 2. Which Python libraries are used?

Streamlit provides the UI, Pandas performs data manipulation, Scikit-learn trains models, Plotly creates charts, FastAPI serves predictions, SQLAlchemy and PostgreSQL provide persistence, and MLflow tracks experiments.

## 3. Which tasks does the platform support?

It supports classification and regression. It also includes EDA, cleaning, feature engineering, SQL analytics, SHAP/LIME explainability, and REST prediction.

## 4. How are trained models saved?

The latest model and its metadata are serialized as a Joblib bundle in `models/latest_model.joblib`. FastAPI loads this bundle for prediction.

## 5. What is MLflow used for?

MLflow logs parameters, metrics, tags, models, and experiment artifacts. This makes runs reproducible and comparable.

## 6. Why is Docker used?

Docker packages services and dependencies consistently. Docker Compose starts the complete system, including PostgreSQL and MLflow.

## 7. How is the API tested?

Pytest uses FastAPI's TestClient to validate endpoint responses and model behavior.

## 8. What is SHAP?

SHAP assigns each feature a contribution value to a prediction and supports local and global explainability.

## 9. What is LIME?

LIME creates a local interpretable approximation around an individual prediction.

## 10. How can the project be improved?

Possible improvements include authentication, role-based access, stronger data validation, automated hyperparameter tuning, CI/CD, cloud deployment, and model monitoring.

