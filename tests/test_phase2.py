"""Tests for Phase 2: parse_report, extract_values, range parser, and normalize_units."""

import os
import pytest

from med_agent.tools.parse import parse_report
from med_agent.tools.extract import parse_reference_range, extract_values
from med_agent.tools.units import normalize_units, UnitRowInput


def _get_sample_path(filename: str) -> str:
    candidates = [
        os.path.join("backend", "samples", filename),
        os.path.join("samples", filename),
        os.path.join(os.path.dirname(__file__), "..", "backend", "samples", filename),
        os.path.join(os.path.dirname(__file__), "..", "samples", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def test_parse_report_txt():
    """Verify parse_report parses sample text files accurately."""
    sample_path = _get_sample_path("report_normal_cbc.txt")
    res = parse_report(sample_path)
    assert res.get("method") == "plain_text"
    assert "Hemoglobin" in res.get("text", "")
    assert res.get("char_count", 0) > 100

    # Non-existent file
    err_res = parse_report("nonexistent_file.pdf")
    assert "error" in err_res


def test_range_parser_formats():
    """Verify deterministic range parser handles all required formats."""
    # Standard interval with spaces
    assert parse_reference_range("13.0 - 17.0") == (13.0, 17.0)
    # Standard interval without spaces
    assert parse_reference_range("13-17") == (13.0, 17.0)
    # Upper bound with less-than
    assert parse_reference_range("< 200") == (None, 200.0)
    assert parse_reference_range("<=5.7") == (None, 5.7)
    assert parse_reference_range("up to 150") == (None, 150.0)
    assert parse_reference_range("less than 100") == (None, 100.0)
    # Lower bound with greater-than
    assert parse_reference_range("> 40") == (40.0, None)
    assert parse_reference_range(">= 60") == (60.0, None)
    assert parse_reference_range("greater than 40") == (40.0, None)
    # Zero to high bound
    assert parse_reference_range("0 - 200") == (0.0, 200.0)
    # Commas in numbers
    assert parse_reference_range("4,000 - 11,000") == (4000.0, 11000.0)
    # En-dash or em-dash
    assert parse_reference_range("13.0 – 17.0") == (13.0, 17.0)
    assert parse_reference_range("13.0 — 17.0") == (13.0, 17.0)
    # Empty or null
    assert parse_reference_range(None) == (None, None)
    assert parse_reference_range("") == (None, None)
    assert parse_reference_range("Negative") == (None, None)


def test_normalize_units():
    """Verify unit conversions and preservation of original values."""
    rows = [
        # Glucose: 8.2 mmol/L -> ~147.7 mg/dL
        UnitRowInput(test="Fast Glucose", value=8.2, unit="mmol/L", ref_low=3.9, ref_high=5.5),
        # Hemoglobin: 115 g/L -> 11.5 g/dL
        UnitRowInput(test="Hemoglobin", value=115.0, unit="g/L", ref_low=130.0, ref_high=170.0),
        # Creatinine: 160 umol/L -> 1.81 mg/dL
        UnitRowInput(test="Serum Creatinine", value=160.0, unit="umol/L", ref_low=53.0, ref_high=106.0),
        # Standard unit that shouldn't change
        UnitRowInput(test="Potassium", value=4.5, unit="mmol/L", ref_low=3.5, ref_high=5.0),
    ]

    res = normalize_units(rows)
    assert res["normalized_count"] == 3
    norm_rows = res["rows"]

    # Glucose check
    assert norm_rows[0]["unit"] == "mg/dL"
    assert norm_rows[0]["original_unit"] == "mmol/L"
    assert norm_rows[0]["original_value"] == 8.2
    assert abs(norm_rows[0]["value"] - 147.7) < 0.2

    # Hemoglobin check
    assert norm_rows[1]["unit"] == "g/dL"
    assert norm_rows[1]["value"] == 11.5
    assert norm_rows[1]["ref_low"] == 13.0
    assert norm_rows[1]["ref_high"] == 17.0

    # Creatinine check
    assert norm_rows[2]["unit"] == "mg/dL"
    assert abs(norm_rows[2]["value"] - 1.81) < 0.05

    # Potassium check (unchanged)
    assert norm_rows[3]["unit"] == "mmol/L"
    assert norm_rows[3]["is_normalized"] is False


def test_sample_report_to_structured_rows():
    """Verify that a sample report becomes correct structured rows."""
    sample_path = _get_sample_path("report_normal_cbc.txt")
    parse_result = parse_report(sample_path)
    text = parse_result["text"]

    extraction = extract_values(text=text)
    assert extraction["count"] >= 4
    assert extraction["needs_review"] is False

    test_names = [r["test"].lower() for r in extraction["rows"]]
    assert any("hemoglobin" in t for t in test_names)
    assert any("platelet" in t for t in test_names)
