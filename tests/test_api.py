from fastapi.testclient import TestClient

from api.main import app
import api.main as api_main


client = TestClient(app)


def test_api_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_api_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "healthy"
    assert "model_available" in response_data


def test_model_info(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_model_bundle",
        lambda: {
            "model_name": "Test Model",
            "task_type": "Regression",
            "target": "price",
            "features": ["area", "rooms"],
        },
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["required_features"] == ["area", "rooms"]


def test_missing_model_returns_not_found(monkeypatch):
    def raise_missing_model():
        raise api_main.HTTPException(
            status_code=404,
            detail="No saved model was found.",
        )

    monkeypatch.setattr(api_main, "get_model_bundle", raise_missing_model)

    response = client.get("/model-info")

    assert response.status_code == 404


def test_batch_prediction(monkeypatch):
    class DummyModel:
        def predict(self, input_df):
            return [float(row.area + row.rooms) for row in input_df.itertuples()]

    monkeypatch.setattr(
        api_main,
        "get_model_bundle",
        lambda: {
            "model": DummyModel(),
            "model_name": "Test Model",
            "task_type": "Regression",
            "target": "price",
            "features": ["area", "rooms"],
        },
    )

    response = client.post(
        "/batch-predict",
        json={
            "records": [
                {"area": 100, "rooms": 2},
                {"area": 150, "rooms": 3},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["total_predictions"] == 2
