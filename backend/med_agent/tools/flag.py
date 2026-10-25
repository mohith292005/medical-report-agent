"""Deterministic tool to flag lab values as LOW, NORMAL, HIGH, or NO_RANGE.

Core Principle:
Code decides the numbers; the LLM only explains them.
The LLM must never decide whether a value is high or low.
Flags come exclusively from this deterministic evaluation tool.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from med_agent.harness.registry import default_registry

def get_data_filepath(filename: str) -> str:
    """Finds path to a data file, checking med_agent/data, root data, or relative paths."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", filename),
        os.path.join("data", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

FALLBACK_RANGES_PATH = get_data_filepath("reference_ranges.json")


def load_fallback_ranges() -> Dict[str, Any]:
    """Loads fallback adult reference ranges from data/reference_ranges.json."""
    if os.path.exists(FALLBACK_RANGES_PATH):
        try:
            with open(FALLBACK_RANGES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("ranges", {})
        except Exception:
            return {}
    return {}


class LabValueInput(BaseModel):
    """Single laboratory test row to be evaluated."""
    test: str = Field(description="Name or standard abbreviation of the laboratory test")
    value: float = Field(description="Numeric value obtained in the test")
    unit: Optional[str] = Field(default=None, description="Measurement unit (e.g., g/dL, mg/dL)")
    ref_text: Optional[str] = Field(default=None, description="Original reference interval text from report")
    ref_low: Optional[float] = Field(default=None, description="Lower reference limit if present")
    ref_high: Optional[float] = Field(default=None, description="Upper reference limit if present")


class FlagAbnormalInput(BaseModel):
    """Input payload for flag_abnormal tool."""
    rows: List[LabValueInput] = Field(description="List of extracted test value rows")


class FlaggedRow(BaseModel):
    """Output evaluation for a single laboratory test."""
    test: str
    value: float
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    ref_text: Optional[str] = None
    flag: str  # "LOW", "NORMAL", "HIGH", "NO_RANGE"
    source: str  # "report" or "fallback"
    explanation: str


def _match_fallback_key(test_name: str, fallback_ranges: Dict[str, Any]) -> Optional[str]:
    """Matches a test name (e.g. 'Fasting Glucose' or 'Hb') to the fallback ranges key."""
    norm = test_name.lower().strip()
    
    # Direct match
    if norm in fallback_ranges:
        return norm

    # Key matching heuristics
    alias_map = {
        "hb": "hemoglobin",
        "hgb": "hemoglobin",
        "haemoglobin": "hemoglobin",
        "wbc": "wbc",
        "white blood cell": "wbc",
        "white blood cells": "wbc",
        "total leukocyte count": "wbc",
        "tlc": "wbc",
        "leukocytes": "wbc",
        "rbc": "rbc",
        "red blood cell": "rbc",
        "red blood cells": "rbc",
        "platelet": "platelets",
        "platelets": "platelets",
        "plt": "platelets",
        "hematocrit": "hematocrit",
        "hct": "hematocrit",
        "pcv": "hematocrit",
        "glucose": "glucose",
        "fasting glucose": "glucose",
        "blood sugar": "glucose",
        "fbs": "glucose",
        "hba1c": "hba1c",
        "a1c": "hba1c",
        "creatinine": "creatinine",
        "serum creatinine": "creatinine",
        "bun": "bun",
        "blood urea nitrogen": "bun",
        "urea": "bun",
        "egfr": "egfr",
        "gfr": "egfr",
        "cholesterol": "total_cholesterol",
        "total cholesterol": "total_cholesterol",
        "ldl": "ldl",
        "ldl cholesterol": "ldl",
        "hdl": "hdl",
        "hdl cholesterol": "hdl",
        "triglycerides": "triglycerides",
        "triglyceride": "triglycerides",
        "potassium": "potassium",
        "k+": "potassium",
        "sodium": "sodium",
        "na+": "sodium",
        "calcium": "calcium",
        "chloride": "chloride",
        "alt": "alt",
        "sgpt": "alt",
        "ast": "ast",
        "sgot": "ast",
        "bilirubin": "bilirubin",
        "total bilirubin": "bilirubin",
        "alkaline phosphatase": "alkaline_phosphatase",
        "alp": "alkaline_phosphatase",
        "tsh": "tsh",
        "free t4": "free_t4",
        "ft4": "free_t4",
        "vitamin d": "vitamin_d",
        "vit d": "vitamin_d",
        "25-oh vitamin d": "vitamin_d",
        "vitamin b12": "vitamin_b12",
        "b12": "vitamin_b12",
        "ferritin": "ferritin",
    }

    for alias, target in alias_map.items():
        if alias in norm:
            if target in fallback_ranges:
                return target

    return None


@default_registry.tool(
    name="flag_abnormal",
    description="Determines whether laboratory values are LOW, NORMAL, HIGH, or NO_RANGE based on reference limits. Uses fallback adult ranges if none are in the report.",
    input_model=FlagAbnormalInput,
)
def flag_abnormal(rows: List[LabValueInput]) -> Dict[str, Any]:
    """Evaluates each row against its reference range.
    
    Returns:
        Dict with 'flagged_rows' list, 'abnormal_count', and 'summary'.
    """
    fallback_ranges = load_fallback_ranges()
    results = []

    for item in rows:
        val = item.value
        low = item.ref_low
        high = item.ref_high
        source = "report"

        # If report did not supply ranges, check fallback database
        if low is None and high is None:
            matched_key = _match_fallback_key(item.test, fallback_ranges)
            if matched_key and matched_key in fallback_ranges:
                fb = fallback_ranges[matched_key]
                low = fb.get("low")
                high = fb.get("high")
                source = "fallback"

        # Determine flag
        if low is None and high is None:
            flag = "NO_RANGE"
            explanation = "No reference range available from report or standard fallbacks."
        elif low is not None and val < low:
            flag = "LOW"
            explanation = f"Value {val} is below the lower reference limit of {low}."
        elif high is not None and val > high:
            flag = "HIGH"
            explanation = f"Value {val} exceeds the upper reference limit of {high}."
        else:
            flag = "NORMAL"
            explanation = "Value is within the expected reference range."

        results.append({
            "test": item.test,
            "value": item.value,
            "unit": item.unit,
            "ref_low": low,
            "ref_high": high,
            "ref_text": item.ref_text,
            "flag": flag,
            "source": source,
            "explanation": explanation,
        })

    abnormal_count = sum(1 for r in results if r["flag"] in ("LOW", "HIGH"))

    return {
        "flagged_rows": results,
        "total_tests": len(results),
        "abnormal_count": abnormal_count,
        "has_abnormal": abnormal_count > 0,
    }
