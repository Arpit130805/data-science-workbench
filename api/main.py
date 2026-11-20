from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "latest_model.joblib"

app = FastAPI(
    title="Data Science Workbench API",
    description="Prediction API for trained machine-learning models.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, float]] = Field(min_length=1)


def convert_to_python_value(value: Any):
    """Convert NumPy values into JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()

    return value


@lru_cache
def load_model_bundle():
    """Load the latest model and its metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No saved model was found. Train a model in Streamlit first."
        )

    return joblib.load(MODEL_PATH)


def get_model_bundle():
    """Return the saved model bundle or a clear API error."""
    try:
        return load_model_bundle()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load saved model: {error}",
        ) from error


def create_feature_dataframe(records, required_features):
    """Validate input records and convert them to a DataFrame."""
    cleaned_records = []

    for record_number, record in enumerate(records, start=1):
        missing_features = [
            feature
            for feature in required_features
            if feature not in record
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Record {record_number} is missing required features: "
                    f"{', '.join(missing_features)}"
                ),
            )

        cleaned_record = {
            feature: record[feature]
            for feature in required_features
        }

        cleaned_records.append(cleaned_record)

    return pd.DataFrame(
        cleaned_records,
        columns=required_features,
    )


@app.get("/")
def home():
    return {
        "message": "Data Science Workbench Prediction API is running.",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
    }


@app.get("/model-info")
def model_info():
    model_bundle = get_model_bundle()

    return {
        "model_name": model_bundle["model_name"],
        "task_type": model_bundle["task_type"],
        "target_column": model_bundle["target"],
        "required_features": model_bundle["features"],
        "model_file": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model_bundle = get_model_bundle()

    model = model_bundle["model"]
    required_features = model_bundle["features"]
    task_type = model_bundle["task_type"]

    input_df = create_feature_dataframe(
        [request.features],
        required_features,
    )

    try:
        prediction = model.predict(input_df)[0]

        response = {
            "prediction": convert_to_python_value(prediction),
            "model_name": model_bundle["model_name"],
            "task_type": task_type,
        }

        if task_type == "Classification" and hasattr(
            model,
            "predict_proba",
        ):
            probabilities = model.predict_proba(input_df)[0]

            response["class_probabilities"] = {
                str(class_name): float(probability)
                for class_name, probability in zip(
                    model.classes_,
                    probabilities,
                )
            }

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error


@app.post("/batch-predict")
def batch_predict(request: BatchPredictionRequest):
    model_bundle = get_model_bundle()

    model = model_bundle["model"]
    required_features = model_bundle["features"]

    input_df = create_feature_dataframe(
        request.records,
        required_features,
    )

    try:
        predictions = model.predict(input_df)

        return {
            "predictions": [
                convert_to_python_value(prediction)
                for prediction in predictions
            ],
            "total_predictions": len(predictions),
            "model_name": model_bundle["model_name"],
            "task_type": model_bundle["task_type"],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {error}",
        ) from error