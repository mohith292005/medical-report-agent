"""Tools package for the Medical Report Explanation Agent.

Importing this package automatically registers all 9 tools with the default ToolRegistry:
1. parse_report: Reads PDF, image, or text file.
2. extract_values: Structured test extraction with JSON mode & range parsing.
3. normalize_units: Converts mmol/L, umol/L, g/L to canonical medical units.
4. flag_abnormal: Deterministic evaluation of LOW, NORMAL, HIGH, NO_RANGE.
5. critical_check: Inspects panic thresholds and returns urgent banner.
6. explain_term: Trusted glossary retrieval for test explanations.
7. compare_trend: Longitudinal comparison of test results over time.
8. doctor_questions: Template-based collaborative questions for the patient.
9. translate_summary: Multi-language translation preserving numbers and units.
"""

from med_agent.tools.parse import parse_report
from med_agent.tools.extract import extract_values, parse_reference_range
from med_agent.tools.units import normalize_units
from med_agent.tools.flag import flag_abnormal
from med_agent.tools.critical import critical_check
from med_agent.tools.glossary import explain_term
from med_agent.tools.trend import compare_trend
from med_agent.tools.questions import doctor_questions
from med_agent.tools.translate import translate_summary

__all__ = [
    "parse_report",
    "extract_values",
    "parse_reference_range",
    "normalize_units",
    "flag_abnormal",
    "critical_check",
    "explain_term",
    "compare_trend",
    "doctor_questions",
    "translate_summary",
]
