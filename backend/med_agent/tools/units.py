"""Tool for normalizing laboratory measurement units to standard canonical units.

Canonical conversions:
1. Glucose: mmol/L -> mg/dL (multiplied by 18.016)
2. Cholesterol / Lipids: mmol/L -> mg/dL (multiplied by 38.67)
3. Creatinine: umol/L -> mg/dL (divided by 88.4)
4. Hemoglobin: g/L -> g/dL (divided by 10.0)

Design:
Preserves the original value, unit, and reference bounds alongside the normalized
figures so auditability and provenance are 100% maintained.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from med_agent.harness.registry import default_registry


class UnitRowInput(BaseModel):
    """Input row representing a laboratory test."""
    test: str = Field(description="Name of the test")
    value: float = Field(description="Numeric result value")
    unit: Optional[str] = Field(default=None, description="Reported measurement unit")
    ref_low: Optional[float] = Field(default=None, description="Lower reference limit")
    ref_high: Optional[float] = Field(default=None, description="Upper reference limit")
    ref_text: Optional[str] = Field(default=None, description="Original reference interval text")
    confidence: Optional[float] = Field(default=1.0, description="Extraction confidence")


class NormalizeUnitsInput(BaseModel):
    """Input payload for normalize_units tool."""
    rows: List[UnitRowInput] = Field(description="List of extracted test rows to normalize")


def _normalize_single_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies canonical unit conversions to a single test result row."""
    test = row_dict.get("test", "").lower().strip()
    raw_unit = (row_dict.get("unit") or "").lower().strip()
    val = float(row_dict.get("value", 0.0))
    low = row_dict.get("ref_low")
    high = row_dict.get("ref_high")

    normalized = dict(row_dict)
    normalized["original_value"] = val
    normalized["original_unit"] = row_dict.get("unit")
    normalized["is_normalized"] = False

    # 1. Glucose: mmol/L -> mg/dL (x 18.016)
    if "glucose" in test or "sugar" in test or test in ("glu", "fbs"):
        if raw_unit in ("mmol/l", "mmol/litre", "mmoll"):
            factor = 18.016
            normalized["value"] = round(val * factor, 1)
            normalized["unit"] = "mg/dL"
            normalized["ref_low"] = round(low * factor, 1) if low is not None else None
            normalized["ref_high"] = round(high * factor, 1) if high is not None else None
            normalized["is_normalized"] = True

    # 2. Cholesterol & Lipids: mmol/L -> mg/dL (x 38.67)
    elif any(k in test for k in ("cholesterol", "ldl", "hdl", "triglyceride", "lipid")):
        if raw_unit in ("mmol/l", "mmol/litre", "mmoll"):
            factor = 38.67
            normalized["value"] = round(val * factor, 1)
            normalized["unit"] = "mg/dL"
            normalized["ref_low"] = round(low * factor, 1) if low is not None else None
            normalized["ref_high"] = round(high * factor, 1) if high is not None else None
            normalized["is_normalized"] = True

    # 3. Creatinine: umol/L -> mg/dL (/ 88.4)
    elif "creatinine" in test or test in ("cr", "creat"):
        if raw_unit in ("umol/l", "µmol/l", "umol/litre", "micromol/l"):
            factor = 88.4
            normalized["value"] = round(val / factor, 2)
            normalized["unit"] = "mg/dL"
            normalized["ref_low"] = round(low / factor, 2) if low is not None else None
            normalized["ref_high"] = round(high / factor, 2) if high is not None else None
            normalized["is_normalized"] = True

    # 4. Hemoglobin: g/L -> g/dL (/ 10.0)
    elif any(k in test for k in ("hemoglobin", "haemoglobin", "hb", "hgb")):
        if raw_unit in ("g/l", "g/litre", "gram/l"):
            factor = 10.0
            normalized["value"] = round(val / factor, 1)
            normalized["unit"] = "g/dL"
            normalized["ref_low"] = round(low / factor, 1) if low is not None else None
            normalized["ref_high"] = round(high / factor, 1) if high is not None else None
            normalized["is_normalized"] = True

    return normalized


@default_registry.tool(
    name="normalize_units",
    description="Converts non-standard or alternative measurement units (mmol/L, umol/L, g/L) into standard canonical units (mg/dL, g/dL). Preserves original values.",
    input_model=NormalizeUnitsInput,
)
def normalize_units(rows: List[UnitRowInput]) -> Dict[str, Any]:
    """Normalizes all rows to canonical medical units."""
    converted_rows = []
    normalized_count = 0

    for r in rows:
        row_dict = r.model_dump()
        converted = _normalize_single_row(row_dict)
        if converted.get("is_normalized"):
            normalized_count += 1
        converted_rows.append(converted)

    return {
        "rows": converted_rows,
        "total": len(converted_rows),
        "normalized_count": normalized_count,
    }
