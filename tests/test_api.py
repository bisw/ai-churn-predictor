import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_prediction(client):
    payload = {
        "tenure_months": 3,
        "monthly_charges": 2500,
        "support_calls": 5,
        "contract_type": "monthly",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in [
        "low",
        "medium",
        "high",
    ]

    expected_prediction = int(
        result["churn_probability"] >= 0.5
    )
    assert (
        result["churn_prediction"]
        == expected_prediction
    )


def test_invalid_contract(client):
    payload = {
        "tenure_months": 3,
        "monthly_charges": 2500,
        "support_calls": 5,
        "contract_type": "invalid",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_charges(client):
    payload = {
        "tenure_months": 3,
        "monthly_charges": -100,
        "support_calls": 5,
        "contract_type": "monthly",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_field(client):
    payload = {
        "tenure_months": 3,
        "monthly_charges": 2500,
        "contract_type": "monthly",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422