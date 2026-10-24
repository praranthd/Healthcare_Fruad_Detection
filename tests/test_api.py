from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction_endpoint():
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

    assert response.status_code == 200, response.text

    result = response.json()
    assert "prediction" in result
    assert "fraud_probability" in result
    assert result["prediction"] in [0, 1]

    probability = result["fraud_probability"]
    if probability is not None:
        assert 0 <= probability <= 1


def test_prediction_rejects_missing_fields():
    response = client.post("/predict", json={})
    assert response.status_code == 422
