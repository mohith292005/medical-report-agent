"""Tool for extracting structured laboratory test rows from unstructured report text.

Pipeline:
1. LLM Extraction: Sends prompt with json_mode=True to extract test names,
   numeric values, units, reference text, and confidence scores.
2. Retry on Schema Failure: If JSON is invalid or fails Pydantic validation,
   retries once feeding back the exact validation error.
3. Deterministic Range Parser: Independently parses `ref_low` and `ref_high`
   from `ref_text`, handling intervals ("13.0 - 17.0", "13-17"), one-sided bounds
   ("< 200", "> 40", "<=5.7", "up to 150"), and formatting variations.
4. Confidence & Review Check: Calculates overall_confidence; if below 0.6,
   sets `needs_review=True`.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

from med_agent.harness.config import DEFAULT_CONFIG
from med_agent.harness.llm import OpenRouterLLM
from med_agent.harness.registry import default_registry

logger = logging.getLogger(__name__)


def parse_reference_range(ref_text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Deterministically extracts lower and upper numerical bounds from reference text.
    
    Examples:
        "13.0 - 17.0"   -> (13.0, 17.0)
        "13-17"         -> (13.0, 17.0)
        "< 200"         -> (None, 200.0)
        "<=5.7"         -> (None, 5.7)
        "up to 150"     -> (None, 150.0)
        "> 40"          -> (40.0, None)
        ">= 60"         -> (60.0, None)
        "4,000 - 11,000"-> (4000.0, 11000.0)
        "0 - 200"       -> (0.0, 200.0)
    """
    if not ref_text or not isinstance(ref_text, str):
        return None, None

    clean = ref_text.strip().replace(",", "")
    # Normalize unicode dashes (en-dash, em-dash) to standard minus
    clean = re.sub(r"[–—−]", "-", clean)

    # 1. Check for upper-bound only: "<= 150", "< 200", "up to 150", "less than 100", "max 150"
    upper_pattern = re.search(
        r"(?:<=|<|up\s*to|less\s*than|max|maximum)\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE,
    )
    if upper_pattern and not re.search(r"\d+\s*-\s*\d+", clean):
        try:
            return None, float(upper_pattern.group(1))
        except ValueError:
            pass

    # 2. Check for lower-bound only: ">= 40", "> 40", "greater than 40", "min 50", "at least 60"
    lower_pattern = re.search(
        r"(?:>=|>|greater\s*than|min|minimum|at\s*least)\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE,
    )
    if lower_pattern and not re.search(r"\d+\s*-\s*\d+", clean):
        try:
            return float(lower_pattern.group(1)), None
        except ValueError:
            pass

    # 3. Two-sided intervals: "13.0 - 17.0", "13-17", "13.0 to 17.0"
    interval_pattern = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to)\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE,
    )
    if interval_pattern:
        try:
            low = float(interval_pattern.group(1))
            high = float(interval_pattern.group(2))
            return low, high
        except ValueError:
            pass

    return None, None


class RawExtractedRow(BaseModel):
    """Row extracted from report by the LLM."""
    test: str = Field(description="Name or standard label of the test")
    value: float = Field(description="Numerical test result value")
    unit: Optional[str] = Field(default=None, description="Measurement unit (e.g. g/dL, mg/dL)")
    ref_text: Optional[str] = Field(default=None, description="Reference interval printed on the report")
    ref_low: Optional[float] = Field(default=None, description="Lower reference bound")
    ref_high: Optional[float] = Field(default=None, description="Upper reference bound")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this row's extraction")


class ExtractionResponse(BaseModel):
    """Schema expected from the LLM JSON response."""
    rows: List[RawExtractedRow]
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: Optional[str] = None


class ExtractValuesInput(BaseModel):
    """Input payload for extract_values tool."""
    text: str = Field(description="Raw text content of the medical report")


EXTRACT_SYSTEM_PROMPT = """You are a medical laboratory data extraction engine.
Given the raw text of a medical lab report, extract all quantitative test results into valid JSON.

Return ONLY a JSON object with this exact structure:
{
  "rows": [
    {
      "test": "Hemoglobin",
      "value": 14.2,
      "unit": "g/dL",
      "ref_text": "13.0 - 17.0",
      "confidence": 0.95
    }
  ],
  "overall_confidence": 0.95,
  "notes": "Clear CBC report"
}

Extraction Rules:
1. Extract only quantitative tests with numeric values. Ignore qualitative comments like 'Normal appearance'.
2. The 'value' must be a numeric float (e.g. 14.2, not '14.2 g/dL'). Remove commas (e.g. '4,500' -> 4500.0).
3. Extract 'unit' and 'ref_text' exactly as printed.
4. Provide confidence between 0.0 and 1.0 based on clarity and formatting.
5. If text is completely unreadable or missing lab values, return "rows": [] and "overall_confidence": 0.1.
"""


def _heuristic_regex_extract(text: str) -> List[RawExtractedRow]:
    """Deterministic regex extraction fallback for standard lab report lines."""
    rows: List[RawExtractedRow] = []
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith(("-", "=", "#", "*", "_")):
            continue

        lower = line.lower()
        if any(h in lower for h in (
            "test name", "reference range", "patient name", "patient id",
            "ordering physician", "specimen", "status:", "date", "dr.", "doctor",
            "remarks:", "interpretation:", "reviewing pathologist", "lab director", "tech:",
            "critical value", "alert:", "location:", "hematology evaluation", "biochemistry",
            "stat biochemistry", "screening", "risk panel", "metabolic & electrolyte",
            "complete blood count"
        )):
            continue

        # Format 1: Colon format (e.g. "Fast Glucose: 8.2 mmol/L (3.9 - 5.5)" or "Platelet Count: 180000")
        colon_match = re.search(
            r"^([^:]+):\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z0-9\/\^\%\-_]+)?(?:\s*[\(\[]?([0-9\.\-\s<>=–—to]+(?:[A-Za-z\/\%]+)?)[\)\]]?)?",
            line,
        )
        if colon_match:
            test = colon_match.group(1).strip()
            val = float(colon_match.group(2))
            unit = colon_match.group(3)
            ref = colon_match.group(4)
            rows.append(RawExtractedRow(test=test, value=val, unit=unit, ref_text=ref, confidence=0.9))
            continue

        # Format 2: Tabular whitespace format (e.g. "Hemoglobin   14.8   g/dL   13.0 - 17.0")
        m = re.search(
            r"^([A-Za-z][A-Za-z0-9\s\+\-\(\)\/_]+?)\s+([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z0-9\/\^\%\-_]+)?(?:\s+([0-9\.\-\s<>=–—to]+(?:[A-Za-z\/\%]+)?))?$",
            line,
        )
        if m:
            test = m.group(1).strip()
            val = float(m.group(2))
            unit = m.group(3)
            ref = m.group(4)
            rows.append(RawExtractedRow(test=test, value=val, unit=unit, ref_text=ref, confidence=0.9))

    return rows


@default_registry.tool(
    name="extract_values",
    description="Extracts test names, numeric values, units, and reference ranges from medical report text into structured rows.",
    input_model=ExtractValuesInput,
)
def extract_values(
    text: str,
    llm: Optional[OpenRouterLLM] = None,
) -> Dict[str, Any]:
    """Extracts test rows using an LLM JSON call, with validation retry and range parsing."""
    client = llm or OpenRouterLLM()
    
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract lab tests from this medical report:\n\n{text}"},
    ]

    extraction_obj: Optional[ExtractionResponse] = None

    # Check if we should attempt the LLM call or if no API key is provided and no mock
    can_call_llm = bool(client.config.api_key) or client.mock_handler is not None

    if can_call_llm:
        try:
            resp = client.chat(messages=messages, json_mode=True)
            content = resp.message.get("content", "{}")
            try:
                data = json.loads(content)
                extraction_obj = ExtractionResponse(**data)
            except (json.JSONDecodeError, ValidationError) as parse_err:
                logger.warning("LLM extraction failed validation: %s. Retrying once...", parse_err)
                # Retry once with validation error feedback
                retry_messages = list(messages)
                retry_messages.append({"role": "assistant", "content": content})
                retry_messages.append({
                    "role": "user",
                    "content": f"Your JSON output failed validation with error: {parse_err}. Please output valid JSON matching the schema.",
                })
                retry_resp = client.chat(messages=retry_messages, json_mode=True)
                retry_content = retry_resp.message.get("content", "{}")
                retry_data = json.loads(retry_content)
                extraction_obj = ExtractionResponse(**retry_data)
        except Exception as e:
            logger.warning("LLM extraction encountered error (%s); falling back to heuristic parsing", e)

    # If LLM extraction wasn't possible or yielded empty rows on clear text, try deterministic parsing
    if extraction_obj is None or len(extraction_obj.rows) == 0:
        heuristic_rows = _heuristic_regex_extract(text)
        if heuristic_rows:
            extraction_obj = ExtractionResponse(
                rows=heuristic_rows,
                overall_confidence=0.85,
                notes="Extracted via deterministic fallback parser",
            )
        elif extraction_obj is None:
            extraction_obj = ExtractionResponse(
                rows=[],
                overall_confidence=0.0,
                notes="Could not extract any lab values from report text.",
            )

    # Deterministic range parsing for every extracted row
    final_rows = []
    for row in extraction_obj.rows:
        row_dict = row.model_dump()
        # Parse ref_low and ref_high using deterministic range parser
        parsed_low, parsed_high = parse_reference_range(row.ref_text)
        if parsed_low is not None:
            row_dict["ref_low"] = parsed_low
        if parsed_high is not None:
            row_dict["ref_high"] = parsed_high
        final_rows.append(row_dict)

    # Check confidence threshold
    needs_review = extraction_obj.overall_confidence < 0.6 or len(final_rows) == 0

    return {
        "rows": final_rows,
        "overall_confidence": extraction_obj.overall_confidence,
        "needs_review": needs_review,
        "count": len(final_rows),
        "notes": extraction_obj.notes,
    }
