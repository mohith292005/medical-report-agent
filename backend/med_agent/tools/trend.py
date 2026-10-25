"""Deterministic tool to compare test results between current and previous lab reports.

Calculations:
- Absolute Delta: `current_value - previous_value`
- Percentage Change: `((current_value - previous_value) / previous_value) * 100`
- Direction: "up", "down", or "stable" (within +/- 1%)
- Range Movement:
  - "stayed_in_range" if both values are inside normal reference limits.
  - "toward_range" if moving closer to or back into normal limits.
  - "away_from_range" if moving further outside normal limits or leaving normal limits.
  - "unknown" if reference limits are unavailable.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from med_agent.harness.registry import default_registry


class TrendRowInput(BaseModel):
    """Input row representing a laboratory test value."""
    test: str = Field(description="Name of the test")
    value: float = Field(description="Numeric value")
    unit: Optional[str] = Field(default=None, description="Measurement unit")
    ref_low: Optional[float] = Field(default=None, description="Lower reference limit")
    ref_high: Optional[float] = Field(default=None, description="Upper reference limit")


class CompareTrendInput(BaseModel):
    """Input payload for compare_trend tool."""
    current_rows: List[TrendRowInput] = Field(description="Test rows from the current report")
    previous_rows: List[TrendRowInput] = Field(description="Test rows from a prior comparison report")


def _normalize_name(name: str) -> str:
    """Simplifies test name for robust cross-report matching."""
    norm = name.lower().strip()
    norm = norm.replace("count", "").replace("serum", "").replace("total", "").strip()
    return norm


def _calc_range_movement(
    cur_val: float,
    prev_val: float,
    low: Optional[float],
    high: Optional[float],
) -> str:
    """Calculates whether the value moved toward, away, or stayed within normal reference limits."""
    if low is None and high is None:
        return "unknown"

    prev_in_range = (low is None or prev_val >= low) and (high is None or prev_val <= high)
    cur_in_range = (low is None or cur_val >= low) and (high is None or cur_val <= high)

    if prev_in_range and cur_in_range:
        return "stayed_in_range"
    if not prev_in_range and cur_in_range:
        return "toward_range"
    if prev_in_range and not cur_in_range:
        return "away_from_range"

    # Both are out of range: check distance from nearest boundary
    def get_distance(val: float) -> float:
        d = 0.0
        if low is not None and val < low:
            d = low - val
        elif high is not None and val > high:
            d = val - high
        return d

    prev_dist = get_distance(prev_val)
    cur_dist = get_distance(cur_val)

    if cur_dist < prev_dist:
        return "toward_range"
    elif cur_dist > prev_dist:
        return "away_from_range"
    else:
        return "stable_outside_range"


@default_registry.tool(
    name="compare_trend",
    description="Compares current lab results against a previous report. Calculates delta, percentage change, direction, and whether values moved toward or away from normal ranges.",
    input_model=CompareTrendInput,
)
def compare_trend(
    current_rows: List[TrendRowInput],
    previous_rows: List[TrendRowInput],
) -> Dict[str, Any]:
    """Compares current rows with previous rows."""
    # Index previous rows by simplified test name
    prev_map: Dict[str, TrendRowInput] = {}
    for p in previous_rows:
        key = _normalize_name(p.test)
        prev_map[key] = p

    trends: List[Dict[str, Any]] = []

    for c in current_rows:
        key = _normalize_name(c.test)
        if key not in prev_map:
            continue

        prev_item = prev_map[key]
        cur_val = c.value
        prev_val = prev_item.value

        delta = round(cur_val - prev_val, 3)
        pct_change = round(((delta) / prev_val) * 100.0, 1) if prev_val != 0 else 0.0

        if abs(pct_change) < 1.0 or delta == 0.0:
            direction = "stable"
        elif delta > 0:
            direction = "up"
        else:
            direction = "down"

        # Determine reference range to use
        low = c.ref_low if c.ref_low is not None else prev_item.ref_low
        high = c.ref_high if c.ref_high is not None else prev_item.ref_high
        movement = _calc_range_movement(cur_val, prev_val, low, high)

        trends.append({
            "test": c.test,
            "current_value": cur_val,
            "previous_value": prev_val,
            "unit": c.unit or prev_item.unit,
            "delta": delta,
            "percent_change": pct_change,
            "direction": direction,
            "range_movement": movement,
        })

    return {
        "compared_count": len(trends),
        "trends": trends,
    }
