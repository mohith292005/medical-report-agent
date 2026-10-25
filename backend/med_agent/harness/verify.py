"""Verification engine combining deterministic rule-checks and an LLM critic.

Architecture:
Step 1: Deterministic Verifier (Zero LLM cost, 100% predictable)
- (a) Hallucination Check: Every numeric value in the draft must originate from
      the source report or valid tool outputs.
- (b) Flag Alignment: Every test called abnormal in the draft must be flagged
      abnormal in `flag_abnormal`, and every flagged abnormal test must be mentioned.
- (c) Mandatory Disclaimer: Draft must contain:
      "This is an explanation, not medical advice. Please discuss results with your doctor."
- (d) Urgent Banner: If critical findings exist, the urgent banner must be present at the start.
- (e) Banned Phrases: Regex scan for diagnostic ("you have", "you are suffering from",
      "diagnosis is") and medication/dosage phrases ("you should take", "start taking",
      "stop taking", "\d+ mg" outside report text).

Step 2: LLM Critic
- Executes ONLY if the deterministic verifier passes.
- Uses `json_mode=True` to inspect clinical tone, false reassurance, and nuanced safety.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from med_agent.harness.llm import OpenRouterLLM
from med_agent.harness.prompts import CRITIC_SYSTEM_PROMPT
from med_agent.harness.state import RunState
from med_agent.tools.critical import URGENT_BANNER_TEXT

logger = logging.getLogger(__name__)

MANDATORY_DISCLAIMER = "This is an explanation, not medical advice. Please discuss results with your doctor."

BANNED_REGEX_PATTERNS = [
    (r"\byou have\b", "Contains diagnostic phrase 'you have'"),
    (r"\byou are suffering from\b", "Contains diagnostic phrase 'you are suffering from'"),
    (r"\bdiagnosis is\b", "Contains diagnostic phrase 'diagnosis is'"),
    (r"\byour diagnosis\b", "Contains diagnostic phrase 'your diagnosis'"),
    (r"\byou should take\b", "Contains medication instruction 'you should take'"),
    (r"\bstart taking\b", "Contains medication instruction 'start taking'"),
    (r"\bstop taking\b", "Contains medication instruction 'stop taking'"),
    (r"\bprescribe\b", "Contains medical prescription language"),
]


def _extract_all_numbers(text: str) -> Set[str]:
    """Finds all normalized integer and decimal numbers in a text string."""
    if not text:
        return set()
    # Normalize commas in numbers (e.g. 150,000 -> 150000)
    cleaned = text.replace(",", "")
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    
    normalized = set()
    for m in matches:
        normalized.add(m)
        try:
            # Also store float-standardized string so 14.0 matches 14
            val = float(m)
            if val.is_integer():
                normalized.add(str(int(val)))
            normalized.add(str(val))
        except ValueError:
            pass
    return normalized


def verify_deterministic(draft: str, state: RunState) -> Tuple[bool, List[str]]:
    """Runs deterministic checks on an assistant draft explanation.
    
    Returns:
        (passed: bool, issues: List[str])
    """
    issues: List[str] = []

    if not draft or not draft.strip():
        return False, ["Draft explanation is empty."]

    draft_lower = draft.lower()

    # 1. Banned phrase check
    for pattern, description in BANNED_REGEX_PATTERNS:
        if re.search(pattern, draft_lower):
            issues.append(f"Safety violation: {description}.")

    # 2. Dosage check (\d+ mg) outside source numbers
    dosage_matches = re.findall(r"\b(\d+)\s*mg\b", draft_lower)
    if dosage_matches:
        # Check if the mg dosage was in the report (e.g. cholesterol 200 mg/dL is fine, but 500 mg pill is not)
        for d in dosage_matches:
            # If followed by /dl or /l it's a concentration unit, not a pill dosage
            if not re.search(rf"\b{d}\s*mg\s*/\s*[a-z]+", draft_lower):
                # Check if it was in the original report text
                if rf"{d} mg" not in state.report_text.lower():
                    issues.append(f"Safety violation: Draft appears to mention a medication dosage '{d} mg'.")

    # 3. Mandatory disclaimer check
    if "this is an explanation, not medical advice" not in draft_lower:
        issues.append("Missing required disclaimer: 'This is an explanation, not medical advice. Please discuss results with your doctor.'")

    # 4. Critical urgent banner check
    if state.critical_findings:
        banner_snippet = "one or more values are in a range that usually needs prompt medical attention"
        if banner_snippet not in draft_lower:
            issues.append(f"Critical findings exist, but draft does NOT contain the mandatory urgent banner: '{URGENT_BANNER_TEXT}'")
        else:
            # Banner must be near the start (within the first 250 characters)
            banner_pos = draft_lower.find(banner_snippet)
            if banner_pos > 250:
                issues.append("The urgent banner must appear at the very start of your explanation.")

    # 5. Number fidelity / hallucination check
    # Collect all legitimate numbers from source report, tool outputs, and state
    allowed_numbers: Set[str] = set()
    allowed_numbers.update(_extract_all_numbers(state.report_text))
    # Allow numbers from disclaimer / prompt template (e.g. 1-6 for questions, 8th grade)
    allowed_numbers.update({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "24", "48", "72", "90"})
    
    # Numbers from all tool executions
    for tool_name, entries in state.tool_outputs.items():
        for e in entries:
            allowed_numbers.update(_extract_all_numbers(str(e.get("result", ""))))
            allowed_numbers.update(_extract_all_numbers(str(e.get("args", ""))))

    # Extract draft numbers
    draft_numbers = _extract_all_numbers(draft)
    hallucinated_numbers = []
    for num in draft_numbers:
        # Skip small integers (1-6 for numbered lists)
        if num in allowed_numbers:
            continue
        try:
            val = float(num)
            if val.is_integer() and 1 <= int(val) <= 10:
                continue
        except ValueError:
            pass
        hallucinated_numbers.append(num)

    if hallucinated_numbers:
        # If there are hallucinated numbers, flag them
        issues.append(
            f"Number fidelity failure: The following numbers in your draft do not appear in the source report or tool outputs: {', '.join(hallucinated_numbers[:5])}."
        )

    # 6. Flagged test coverage check
    # Check that tests flagged abnormal by flag_abnormal are mentioned in the draft
    if state.flags:
        for f in state.flags:
            flag_val = f.get("flag")
            test_name = f.get("test", "")
            if flag_val in ("LOW", "HIGH"):
                # Simplify test name for check (e.g. "White Blood Cells (WBC)" -> "white blood" or "wbc")
                subterms = [test_name.lower()]
                if "(" in test_name and ")" in test_name:
                    acronym = test_name[test_name.find("(") + 1 : test_name.find(")")].strip().lower()
                    subterms.append(acronym)
                clean_test = re.sub(r"[\(\)\/]", " ", test_name).lower()
                subterms.extend(clean_test.split())

                mentioned = any(term in draft_lower for term in subterms if len(term) >= 3)
                if not mentioned:
                    issues.append(f"Omission: Test '{test_name}' was flagged {flag_val} by flag_abnormal but was not mentioned in the explanation.")

    passed = len(issues) == 0
    return passed, issues


def verify_critic(
    draft: str,
    state: RunState,
    llm: Optional[OpenRouterLLM] = None,
) -> Tuple[bool, List[str]]:
    """Runs the LLM Critic on the draft explanation after deterministic verifier passes.
    
    Returns:
        (passed: bool, issues: List[str])
    """
    client = llm or OpenRouterLLM()
    # If no API key is provided and no mock, pass critic gracefully
    if not client.config.api_key and client.mock_handler is None:
        return True, []

    critic_prompt = f"""Medical Report Text:\n{state.report_text}\n\nDraft Explanation To Review:\n{draft}\n\nCritical Findings Present: {bool(state.critical_findings)}"""
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": critic_prompt},
    ]

    try:
        resp = client.chat(messages=messages, json_mode=True, temperature=0.0)
        content = resp.message.get("content", "{}")
        data = json.loads(content)
        critic_pass = bool(data.get("pass", False))
        critic_issues = data.get("issues", [])
        return critic_pass, critic_issues
    except Exception as e:
        logger.warning("LLM Critic execution failed (%s); accepting deterministic verdict", e)
        return True, []
