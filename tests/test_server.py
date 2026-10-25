"""Tests for the FastAPI server endpoints.

Uses FastAPI's TestClient (backed by httpx) to test:
  - GET /api/health
  - POST /api/analyze with text input
  - POST /api/analyze with file upload
  - POST /api/analyze with missing input (422)
  - GET / serves frontend HTML
"""

import os
import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)

SAMPLES_DIR = (
    os.path.join(os.path.dirname(__file__), "..", "backend", "samples")
    if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "backend", "samples"))
    else os.path.join(os.path.dirname(__file__), "..", "samples")
)


# ── Health Check ────────────────────────────────────────────────────────────

def test_health_endpoint():
    """GET /api/health should return 200 with status=ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── Frontend Serving ────────────────────────────────────────────────────────

def test_root_serves_html():
    """GET / should serve the frontend index.html."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "MedExplain" in resp.text


# ── Analyze: Text Input ────────────────────────────────────────────────────

def test_analyze_with_text():
    """POST /api/analyze with pasted text should return structured results."""
    report_text = (
        "Hemoglobin: 10.2 g/dL (Ref: 13.0 - 17.0)\n"
        "WBC: 6500 /uL (Ref: 4000 - 11000)\n"
        "Glucose: 105 mg/dL (Ref: 70 - 99)\n"
    )
    resp = client.post("/api/analyze", data={"text": report_text, "language": "English"})
    assert resp.status_code == 200

    data = resp.json()

    # Must have an explanation
    assert "explanation" in data
    assert len(data["explanation"]) > 50

    # Must have flags
    assert "flags" in data
    assert len(data["flags"]) >= 2  # At least hemoglobin and glucose

    # Hemoglobin should be LOW
    hb_flags = [f for f in data["flags"] if "hemoglobin" in f["test"].lower()]
    assert len(hb_flags) >= 1
    assert hb_flags[0]["flag"] == "LOW"

    # WBC should be NORMAL
    wbc_flags = [f for f in data["flags"] if "wbc" in f["test"].lower()]
    if wbc_flags:
        assert wbc_flags[0]["flag"] == "NORMAL"

    # Must have stop_reason
    assert data["stop_reason"] == "completed"

    # Must have disclaimer in explanation
    assert "not medical advice" in data["explanation"].lower() or "discuss" in data["explanation"].lower()


# ── Analyze: File Upload ───────────────────────────────────────────────────

def test_analyze_with_file_upload():
    """POST /api/analyze with a .txt file upload should return results."""
    sample_file = os.path.join(SAMPLES_DIR, "report_normal_cbc.txt")
    if not os.path.exists(sample_file):
        pytest.skip("Sample file not found")

    with open(sample_file, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("report_normal_cbc.txt", f, "text/plain")},
            data={"language": "English"},
        )

    assert resp.status_code == 200
    data = resp.json()

    assert data["stop_reason"] == "completed"
    assert len(data["flags"]) >= 4  # Normal CBC has 5 tests
    # All should be NORMAL in this sample
    flag_types = set(f["flag"] for f in data["flags"])
    assert "NORMAL" in flag_types


# ── Analyze: Critical Values ──────────────────────────────────────────────

def test_analyze_critical_values():
    """POST /api/analyze with critical potassium should return critical findings."""
    sample_file = os.path.join(SAMPLES_DIR, "report_critical_potassium.txt")
    if not os.path.exists(sample_file):
        pytest.skip("Sample file not found")

    with open(sample_file, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("report_critical_potassium.txt", f, "text/plain")},
            data={"language": "English"},
        )

    assert resp.status_code == 200
    data = resp.json()

    # Must detect critical findings
    assert len(data["critical_findings"]) >= 1
    # Potassium should be critical
    critical_tests = [cf["test"].lower() for cf in data["critical_findings"]]
    assert any("potassium" in t for t in critical_tests)


# ── Analyze: Missing Input ─────────────────────────────────────────────────

def test_analyze_missing_input_returns_422():
    """POST /api/analyze without text or file should return 422."""
    resp = client.post("/api/analyze", data={"language": "English"})
    assert resp.status_code == 422


# ── Analyze: Response Shape ────────────────────────────────────────────────

def test_analyze_response_shape():
    """Verify all expected fields exist in the response."""
    resp = client.post(
        "/api/analyze",
        data={
            "text": "Platelets: 260000 /uL (Ref: 150000 - 450000)",
            "language": "English",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    expected_keys = {"explanation", "flags", "critical_findings", "stop_reason", "steps", "tokens_used", "trace_file"}
    assert expected_keys.issubset(set(data.keys()))

    # Types
    assert isinstance(data["explanation"], str)
    assert isinstance(data["flags"], list)
    assert isinstance(data["critical_findings"], list)
    assert isinstance(data["stop_reason"], str)
    assert isinstance(data["steps"], int)
    assert isinstance(data["tokens_used"], int)


# ── Chatbot Endpoint Tests ──────────────────────────────────────────────────

def test_chat_overview_intent():
    """POST /api/chat with 'what is going on' should return report breakdown."""
    payload = {
        "message": "What's going on with my report?",
        "flags": [
            {"test": "potassium", "value": 6.8, "unit": "mmol/L", "flag": "HIGH", "reference_low": 3.5, "reference_high": 5.0},
            {"test": "hemoglobin", "value": 14.5, "unit": "g/dL", "flag": "NORMAL", "reference_low": 13.0, "reference_high": 17.0}
        ],
        "critical_findings": [
            {"test": "potassium", "value": 6.8, "unit": "mmol/L", "threshold": 6.5, "direction": "HIGH", "message": "CRITICAL: potassium severely high"}
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "suggestions" in data
    assert "potassium" in data["reply"].lower()
    assert len(data["suggestions"]) > 0


def test_chat_lifestyle_suggestions_intent():
    """POST /api/chat asking for diet/lifestyle suggestions."""
    payload = {
        "message": "What dietary suggestions or foods should I eat?",
        "flags": [
            {"test": "hemoglobin", "value": 9.2, "unit": "g/dL", "flag": "LOW", "reference_low": 13.0, "reference_high": 17.0}
        ],
        "critical_findings": []
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "iron" in data["reply"].lower() or "nutrition" in data["reply"].lower() or "lifestyle" in data["reply"].lower()


def test_chat_doctor_questions_intent():
    """POST /api/chat asking what to ask the doctor."""
    payload = {
        "message": "What questions should I ask my doctor?",
        "flags": [
            {"test": "potassium", "value": 6.8, "unit": "mmol/L", "flag": "HIGH", "reference_low": 3.5, "reference_high": 5.0}
        ],
        "critical_findings": []
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "doctor" in data["reply"].lower() or "physician" in data["reply"].lower()
