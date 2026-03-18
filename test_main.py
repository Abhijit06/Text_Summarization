import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)


# ── Health check ──────────────────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Input validation ──────────────────────────────────────────────────────────
def test_empty_text_returns_400():
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 400


def test_short_text_returns_400():
    response = client.post("/summarize", json={"text": "Too short."})
    assert response.status_code == 400


def test_text_too_long_returns_400():
    response = client.post("/summarize", json={"text": "a" * 100_001})
    assert response.status_code == 400


# ── Successful summarization (mocked) ────────────────────────────────────────
@patch("main.summarize_text")
def test_summarize_success(mock_summarize):
    mock_summarize.return_value = {
        "summary": "This is a mocked summary.",
        "model_used": "facebook/bart-large-cnn",
        "chunks_processed": 1,
    }
    payload = {
        "text": "A" * 200,
        "length": "medium",
        "model": "bart"
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "compression_ratio" in data
    assert data["model_used"] == "facebook/bart-large-cnn"


@patch("main.summarize_text")
def test_short_length_preset(mock_summarize):
    mock_summarize.return_value = {
        "summary": "Short summary.",
        "model_used": "facebook/bart-large-cnn",
        "chunks_processed": 1,
    }
    response = client.post("/summarize", json={"text": "B" * 200, "length": "short"})
    assert response.status_code == 200
    mock_summarize.assert_called_once_with(
        text="B" * 200, length="short", model_name="bart"
    )


# ── Compression ratio ─────────────────────────────────────────────────────────
@patch("main.summarize_text")
def test_compression_ratio_calculated(mock_summarize):
    original = "W" * 1000
    summary = "W" * 100
    mock_summarize.return_value = {
        "summary": summary,
        "model_used": "facebook/bart-large-cnn",
        "chunks_processed": 1,
    }
    response = client.post("/summarize", json={"text": original})
    assert response.status_code == 200
    assert response.json()["compression_ratio"] == 0.1
