import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Create a test client that runs the FastAPI lifespan.
    This allows the model and preprocessor to load before tests run.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_home_endpoint(client):
    """Test that the home endpoint responds successfully."""
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint(client):
    """Test that the health endpoint responds successfully."""
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction_endpoint(client):
    """Test that the prediction endpoint returns a valid prediction."""

    payload = {
        "Patient_Age": 45,
        "Patient_Gender": "Female",
        "Diagnosis_Code": "I10",
        "Procedure_Code": 120,
        "Claim_Amount": 1500.0,
        "Insurance_Type": "Private",
        "Provider_Specialty": "Cardiology",
        "Patient_State": "CA",
        "Visit_Type": "Outpatient",
        "Days_Between_Service_and_Claim": 3,
        "Number_of_Claims_Per_Provider_Monthly": 15,
        "Length_of_Stay": 2,
        "Chronic_Condition_Flag": 1,
        "Prior_Visits_12m": 2,
        "Claim_Submission_Date": "2026-10-20",
    }

    response = client.post("/predict", json=payload)

    # The API should successfully process the claim.
    assert response.status_code == 200, response.text

    result = response.json()

    # Check that the expected response fields exist.
    assert "prediction" in result
    assert "fraud_probability" in result

    # Prediction must be either 0 (not flagged) or 1 (flagged).
    assert result["prediction"] in [0, 1]

    # Probability must be between 0 and 1 when provided.
    probability = result["fraud_probability"]

    if probability is not None:
        assert isinstance(probability, (int, float))
        assert 0 <= probability <= 1


def test_prediction_rejects_missing_fields(client):
    """Test that an empty request is rejected by input validation."""

    response = client.post("/predict", json={})

    assert response.status_code == 422
