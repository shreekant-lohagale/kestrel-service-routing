from fastapi.testclient import TestClient

from src.api import (
    app,
    REVIEW_MARGIN_THRESHOLD,
)


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True

    assert (
        data["review_margin_threshold"]
        == REVIEW_MARGIN_THRESHOLD
    )


def test_clear_repair_request():
    payload = {
        "request_text": (
            "My water purifier stopped working "
            "and is showing error code E2"
        ),
        "channel": "chat",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_team"] == "Repairs"

    assert data["routing_status"] == "auto_route"

    assert (
        data["needs_human_review"]
        is False
    )

    assert (
        data["decision_margin"]
        >= REVIEW_MARGIN_THRESHOLD
    )

    assert len(data["reasons"]) >= 1


def test_ambiguous_request_requires_review():
    payload = {
        "request_text": (
            "Please call me about my purifier"
        ),
        "channel": "chat",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["routing_status"]
        == "human_review"
    )

    assert (
        data["needs_human_review"]
        is True
    )

    assert (
        data["decision_margin"]
        < REVIEW_MARGIN_THRESHOLD
    )


def test_invalid_empty_request():
    payload = {
        "request_text": "",
        "channel": "chat",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_response_contains_required_fields():
    payload = {
        "request_text": (
            "My mixer grinder motor smells burnt"
        ),
        "channel": "email",
        "product_family": "Mixer Grinder",
        "warranty_status": "in_warranty",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "predicted_team",
        "routing_status",
        "needs_human_review",
        "decision_margin",
        "review_threshold",
        "reasons",
        "note",
    }

    assert required_fields.issubset(
        data.keys()
    )