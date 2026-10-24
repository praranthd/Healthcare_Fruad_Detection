
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


# Skip loading local MLflow model artifacts in CI tests.
@asynccontextmanager
async def ci_lifespan(app):
    yield


api_main.app.router.lifespan_context = ci_lifespan


@pytest.fixture(scope="module")
def client():
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction_rejects_missing_fields(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
