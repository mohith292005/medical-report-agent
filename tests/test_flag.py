"""Tests for deterministic flag_abnormal tool."""

import pytest
from med_agent.tools.flag import flag_abnormal, LabValueInput


def test_flag_normal_and_abnormal():
    """Verify normal, low, and high flagging based on report reference limits."""
    rows = [
        LabValueInput(
            test="Hemoglobin",
            value=14.5,
            unit="g/dL",
            ref_low=13.0,
            ref_high=17.0,
        ),
        LabValueInput(
            test="Hemoglobin",
            value=10.2,
            unit="g/dL",
            ref_low=13.0,
            ref_high=17.0,
        ),
        LabValueInput(
            test="Fasting Glucose",
            value=145.0,
            unit="mg/dL",
            ref_low=70.0,
            ref_high=99.0,
        ),
    ]

    result = flag_abnormal(rows)
    assert result["total_tests"] == 3
    assert result["abnormal_count"] == 2
    assert result["has_abnormal"] is True

    flagged = result["flagged_rows"]
    assert flagged[0]["flag"] == "NORMAL"
    assert flagged[0]["source"] == "report"

    assert flagged[1]["flag"] == "LOW"
    assert flagged[1]["source"] == "report"

    assert flagged[2]["flag"] == "HIGH"
    assert flagged[2]["source"] == "report"


def test_fallback_reference_ranges():
    """Verify fallback database is queried when report provides no reference range."""
    rows = [
        # Potassium fallback range: 3.5 - 5.0 mmol/L
        LabValueInput(
            test="Potassium",
            value=6.1,
            unit="mmol/L",
            ref_low=None,
            ref_high=None,
        ),
        # Platelets fallback range: 150000 - 450000 /uL
        LabValueInput(
            test="Platelet Count",
            value=250000.0,
            unit="/uL",
            ref_low=None,
            ref_high=None,
        ),
    ]

    result = flag_abnormal(rows)
    flagged = result["flagged_rows"]

    assert flagged[0]["flag"] == "HIGH"
    assert flagged[0]["source"] == "fallback"
    assert flagged[0]["ref_high"] == 5.0

    assert flagged[1]["flag"] == "NORMAL"
    assert flagged[1]["source"] == "fallback"
    assert flagged[1]["ref_low"] == 150000.0


def test_single_sided_and_no_range():
    """Verify single-sided ranges (e.g. < 200) and unknown tests without ranges."""
    rows = [
        # Upper-limit only: LDL < 100
        LabValueInput(
            test="LDL",
            value=125.0,
            unit="mg/dL",
            ref_low=None,
            ref_high=100.0,
        ),
        # Upper-limit only, within limit:
        LabValueInput(
            test="LDL",
            value=85.0,
            unit="mg/dL",
            ref_low=None,
            ref_high=100.0,
        ),
        # Completely unknown test with no range
        LabValueInput(
            test="Unknown Experimental Marker XYZ",
            value=42.0,
            unit="units",
            ref_low=None,
            ref_high=None,
        ),
    ]

    result = flag_abnormal(rows)
    flagged = result["flagged_rows"]

    assert flagged[0]["flag"] == "HIGH"
    assert flagged[1]["flag"] == "NORMAL"
    assert flagged[2]["flag"] == "NO_RANGE"
