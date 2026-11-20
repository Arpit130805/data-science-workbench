# Data Science Workbench - Project Report

## 1. Introduction

Data Science Workbench is a unified web application that supports a complete machine-learning workflow. It helps users load datasets, explore and clean data, engineer features, train and evaluate models, track experiments, explain predictions, query saved datasets, and deploy the latest model through an API.

## 2. Objectives

- Provide an accessible end-to-end data science environment.
- Support both classification and regression workflows.
- Persist datasets and trained experiment metadata.
- Compare MLflow runs and inspect model explanations.
- Expose trained models through a REST API.
- Deploy all major components using Docker.

## 3. Technology Stack

Python, Streamlit, Pandas, Scikit-learn, Plotly, FastAPI, PostgreSQL, SQLAlchemy, MLflow, SHAP, LIME, Joblib, Docker, and Docker Compose.

## 4. Workflow

1. Upload and save a dataset.
2. Explore distributions and relationships.
3. Clean and transform the data.
4. Select features and target.
5. Train a model.
6. Evaluate metrics and visualizations.
7. Track the run in MLflow.
8. Explain the trained model.
9. Save the model for API use.
10. Generate predictions through Streamlit or FastAPI.

## 5. Testing

Pytest verifies API routing, health reporting, model metadata behavior, missing-model handling, and batch prediction behavior.

## 6. Deployment

Docker Compose launches PostgreSQL, MLflow, FastAPI, and Streamlit. Each service has an explicit port and persistent storage is configured for PostgreSQL and MLflow data.

## 7. Future Enhancements

- Authentication and role-based access
- Hyperparameter search
- Dataset/model versioning
- Cloud artifact storage
- HTTPS and secrets management
- Automated data validation

