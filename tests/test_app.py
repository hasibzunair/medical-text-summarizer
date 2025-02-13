import sys
import os

from fastapi.testclient import TestClient

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_path)

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert "status" in data
    assert data["status"] == "this works"


def test_feedback_endpoint():
    response = client.post("/feedback", json={"feedback": "Great service"})
    data = response.json()

    assert response.status_code == 200
    assert "message" in data
    assert data["message"] == "Got feedback."


def test_summarize_endpoint():
    test_text = (
        "The patient has a fever and cough. "
        "It may be pneumonia. The patient requires an X-ray and blood tests."
    )
    response = client.post(
        "/summarize", json={"text": test_text, "clinical_role": "physician"}
    )
    data = response.json()

    assert response.status_code == 200
    assert "summary" in data
    assert "tokens" in data
    assert "references" in data
    assert "processing_time" in data

    # check data types
    assert isinstance(data["summary"], str)
    assert isinstance(data["tokens"], int)
    assert isinstance(data["references"], list)
    assert isinstance(data["processing_time"], (float, int))


def test_summarize_endpoint_non_medical():
    non_medical_text = "I like to be close to nature during sunny days."
    response = client.post(
        "/summarize", json={"text": non_medical_text, "clinical_role": "physician"}
    )
    assert response.status_code == 200
    text = response.json()["summary"]
    assert (
        text == "<|no_medical_text|>"
    ), f"Should be <|no_medical_text|>, but got {text}."
