"""Deterministic tool to retrieve trusted plain-language explanations from the medical glossary.

Safety Principle:
The LLM must never freestyle or hallucinate biological descriptions.
It retrieves verified, peer-reviewed plain-language definitions from `data/glossary.json`.
If a test is not in the glossary, it returns `found=False` so the agent explicitly
informs the user that no verified explanation is on file.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from med_agent.harness.registry import default_registry


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


def load_glossary() -> Dict[str, Any]:
    """Loads glossary definitions from data/glossary.json."""
    path = get_data_filepath("glossary.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


class ExplainTermInput(BaseModel):
    """Input payload for explain_term tool."""
    term: str = Field(description="Name or abbreviation of the medical lab test/term to explain (e.g. 'Hemoglobin', 'LDL', 'HbA1c')")


def _search_glossary(term: str, glossary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Performs exact and alias-based keyword lookup in the glossary."""
    clean_term = term.lower().strip()

    # 1. Direct dictionary key match
    if clean_term in glossary:
        return glossary[clean_term]

    # 2. Check aliases and official names
    for key, data in glossary.items():
        name = data.get("name", "").lower()
        aliases = [a.lower() for a in data.get("aliases", [])]

        if clean_term == name or clean_term in aliases:
            return data
        
        # Word-boundary or substring check
        if clean_term in name or any(clean_term in a for a in aliases):
            return data

        # Check if key is a sub-term
        if key in clean_term:
            return data

    return None


@default_registry.tool(
    name="explain_term",
    description="Retrieves trusted, neutral, non-diagnostic plain-language explanations of medical tests and their general significance from the medical glossary.",
    input_model=ExplainTermInput,
)
def explain_term(term: str) -> Dict[str, Any]:
    """Looks up term in glossary.json and returns verified explanation."""
    glossary = load_glossary()
    entry = _search_glossary(term, glossary)

    if not entry:
        return {
            "term": term,
            "found": False,
            "explanation": None,
            "source": "glossary",
            "message": f"No verified explanation found for term '{term}' in the glossary.",
        }

    return {
        "term": term,
        "found": True,
        "name": entry.get("name"),
        "what_it_measures": entry.get("what_it_measures"),
        "general_meaning": entry.get("general_meaning", {}),
        "source": "glossary",
    }
