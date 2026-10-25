"""Deterministic tool for checking laboratory values against critical panic thresholds.

Core Safety Rule:
Critical values represent acute physiological danger (e.g. severe hyperkalemia,
extreme hypoglycemia, profound anemia).
If ANY critical value is detected:
1. `has_critical` is set to True.
2. An urgent banner string is returned:
   "One or more values are in a range that usually needs prompt medical attention. Please contact your doctor or a hospital today."
3. The harness verifier will strictly enforce that this urgent banner appears
   at the very start of the final patient explanation.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from med_agent.harness.registry import default_registry

URGENT_BANNER_TEXT = (
    "One or more values are in a range that usually needs prompt medical attention. "
    "Please contact your doctor or a hospital today."
)


def get_data_filepath(filename: str) -> str:
    """Finds path to a data file across package and root directories."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", filename),
        os.path.join("data", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_critical_values() -> Dict[str, Any]:
    """Loads panic thresholds from data/critical_values.json."""
    path = get_data_filepath("critical_values.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("tests", {})
        except Exception:
            return {}
    return {}


class CriticalRowInput(BaseModel):
    """Input row representing a laboratory test value."""
    test: str = Field(description="Name or standard label of the test")
    value: float = Field(description="Numeric test result")
    unit: Optional[str] = Field(default=None, description="Measurement unit")


class CriticalCheckInput(BaseModel):
    """Input payload for critical_check tool."""
    rows: List[CriticalRowInput] = Field(description="List of lab test rows to inspect for critical panic values")


def _match_critical_key(test_name: str, critical_tests: Dict[str, Any]) -> Optional[str]:
    """Matches a test name to a critical value entry."""
    norm = test_name.lower().strip()
    if norm in critical_tests:
        return norm

    alias_map = {
        "potassium": "potassium",
        "k+": "potassium",
        "glucose": "glucose",
        "sugar": "glucose",
        "fbs": "glucose",
        "hemoglobin": "hemoglobin",
        "hb": "hemoglobin",
        "hgb": "hemoglobin",
        "sodium": "sodium",
        "na+": "sodium",
        "platelet": "platelets",
        "platelets": "platelets",
        "plt": "platelets",
        "wbc": "wbc",
        "white blood": "wbc",
        "calcium": "calcium",
    }
    for alias, target in alias_map.items():
        if alias in norm:
            if target in critical_tests:
                return target
    return None


@default_registry.tool(
    name="critical_check",
    description="Inspects test results against panic/critical thresholds (e.g. potassium > 6.5, hemoglobin < 7.0). Returns critical findings and urgent banner if any threshold is breached.",
    input_model=CriticalCheckInput,
)
def critical_check(rows: List[CriticalRowInput]) -> Dict[str, Any]:
    """Checks rows against panic thresholds."""
    critical_db = load_critical_values()
    findings: List[Dict[str, Any]] = []

    for r in rows:
        key = _match_critical_key(r.test, critical_db)
        if not key:
            continue

        crit_info = critical_db[key]
        val = r.value
        panic_low = crit_info.get("panic_low")
        panic_high = crit_info.get("panic_high")
        urgency_msg = crit_info.get("urgency_message", "")

        is_critical = False
        breach_direction = None

        if panic_low is not None and val < panic_low:
            is_critical = True
            breach_direction = "CRITICALLY_LOW"
        elif panic_high is not None and val > panic_high:
            is_critical = True
            breach_direction = "CRITICALLY_HIGH"

        if is_critical:
            findings.append({
                "test": r.test,
                "canonical_test": key,
                "value": val,
                "unit": r.unit,
                "breach": breach_direction,
                "panic_low": panic_low,
                "panic_high": panic_high,
                "urgency_message": urgency_msg,
            })

    has_critical = len(findings) > 0
    return {
        "has_critical": has_critical,
        "critical_count": len(findings),
        "critical_findings": findings,
        "urgent_banner": URGENT_BANNER_TEXT if has_critical else None,
    }
