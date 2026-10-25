"""Tests for Phase 3 tools: critical_check, glossary, trend, questions, and translate."""

import pytest
from med_agent.tools.critical import critical_check, CriticalRowInput, URGENT_BANNER_TEXT
from med_agent.tools.glossary import explain_term
from med_agent.tools.trend import compare_trend, TrendRowInput
from med_agent.tools.questions import doctor_questions, FlaggedItemInput
from med_agent.tools.translate import translate_summary


def test_critical_check_panic_thresholds():
    """Verify panic value detection and urgent banner generation."""
    # Critical Potassium: 6.8 mmol/L > 6.5
    rows_critical = [
        CriticalRowInput(test="Potassium", value=6.8, unit="mmol/L"),
        CriticalRowInput(test="Sodium", value=140.0, unit="mmol/L"),
    ]
    res_crit = critical_check(rows_critical)
    assert res_crit["has_critical"] is True
    assert res_crit["critical_count"] == 1
    assert res_crit["urgent_banner"] == URGENT_BANNER_TEXT
    finding = res_crit["critical_findings"][0]
    assert finding["breach"] == "CRITICALLY_HIGH"
    assert finding["canonical_test"] == "potassium"

    # Critical low hemoglobin: 6.2 g/dL < 7.0
    rows_low_hb = [
        CriticalRowInput(test="Hemoglobin", value=6.2, unit="g/dL"),
    ]
    res_low_hb = critical_check(rows_low_hb)
    assert res_low_hb["has_critical"] is True
    assert res_low_hb["critical_findings"][0]["breach"] == "CRITICALLY_LOW"

    # Normal values: no panic breach
    rows_normal = [
        CriticalRowInput(test="Potassium", value=4.2, unit="mmol/L"),
        CriticalRowInput(test="Hemoglobin", value=14.0, unit="g/dL"),
    ]
    res_norm = critical_check(rows_normal)
    assert res_norm["has_critical"] is False
    assert res_norm["urgent_banner"] is None


def test_explain_term_glossary():
    """Verify keyword and alias retrieval from glossary."""
    # Direct match
    res_hb = explain_term("Hemoglobin")
    assert res_hb["found"] is True
    assert res_hb["source"] == "glossary"
    assert "oxygen" in res_hb["what_it_measures"].lower()

    # Alias match ("HbA1c" or "a1c")
    res_a1c = explain_term("A1C")
    assert res_a1c["found"] is True
    assert "blood sugar" in res_a1c["what_it_measures"].lower()

    # Non-existent term
    res_unk = explain_term("supercalifragilistic_protein")
    assert res_unk["found"] is False
    assert res_unk["explanation"] is None


def test_compare_trend():
    """Verify longitudinal trend calculation and range movement."""
    current = [
        TrendRowInput(test="Hemoglobin", value=13.5, unit="g/dL", ref_low=13.0, ref_high=17.0),
        TrendRowInput(test="LDL Cholesterol", value=175.0, unit="mg/dL", ref_low=None, ref_high=100.0),
    ]
    previous = [
        TrendRowInput(test="Hemoglobin", value=11.0, unit="g/dL", ref_low=13.0, ref_high=17.0),
        TrendRowInput(test="LDL Cholesterol", value=140.0, unit="mg/dL", ref_low=None, ref_high=100.0),
    ]

    res = compare_trend(current, previous)
    assert res["compared_count"] == 2
    trends = res["trends"]

    # Hemoglobin: was 11.0 (LOW), now 13.5 (NORMAL) -> moved toward_range, up
    hb_trend = trends[0]
    assert hb_trend["test"] == "Hemoglobin"
    assert hb_trend["delta"] == 2.5
    assert hb_trend["direction"] == "up"
    assert hb_trend["range_movement"] == "toward_range"

    # LDL: was 140.0 (HIGH), now 175.0 (HIGHER) -> moved away_from_range, up
    ldl_trend = trends[1]
    assert ldl_trend["delta"] == 35.0
    assert ldl_trend["direction"] == "up"
    assert ldl_trend["range_movement"] == "away_from_range"


def test_doctor_questions():
    """Verify template-based non-diagnostic question generation."""
    flags = [
        FlaggedItemInput(test="Hemoglobin", flag="LOW", value=9.2, unit="g/dL"),
        FlaggedItemInput(test="Serum Ferritin", flag="LOW", value=8.0, unit="ng/mL"),
    ]
    res = doctor_questions(flags)
    assert 3 <= res["count"] <= 6
    qs = res["questions"]
    # Check that question addresses hemoglobin/iron
    assert any("hemoglobin" in q.lower() or "iron" in q.lower() for q in qs)
    # Check that questions do not offer diagnosis or dosage
    for q in qs:
        assert "you have" not in q.lower()
        assert "mg" not in q.lower()


def test_translate_english_noop():
    """Verify English translation returns the input text verbatim."""
    sample = "Hemoglobin is 14.8 g/dL. This is an explanation, not medical advice."
    res = translate_summary(sample, "English")
    assert res["translated_text"] == sample
    assert res["is_translated"] is False
