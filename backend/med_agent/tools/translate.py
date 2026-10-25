"""Tool for translating the verified plain-language summary into regional languages.

Supported Languages:
- English (default)
- Hindi
- Kannada
- Telugu
- Tamil

Hard Translation Invariant:
EVERY numeric value, unit of measurement (e.g. g/dL, mg/dL, /uL), reference interval,
and disclaimer statement MUST be preserved with 100% exact numerical fidelity.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from med_agent.harness.config import DEFAULT_CONFIG
from med_agent.harness.llm import OpenRouterLLM
from med_agent.harness.registry import default_registry

SUPPORTED_LANGUAGES = ["english", "hindi", "kannada", "telugu", "tamil"]

TRANSLATE_SYSTEM_PROMPT = """You are a medical report translation engine.
Translate the medical report explanation into the target Indian language ({language}).

CRITICAL INVARIANTS:
1. PRESERVE EVERY NUMBER EXACTLY: Do NOT translate, alter, or transliterate numeric figures (e.g., '14.8', '6500', '13.0 - 17.0' must remain exactly '14.8', '6500', '13.0 - 17.0').
2. PRESERVE EVERY MEASUREMENT UNIT EXACTLY: Keep 'g/dL', 'mg/dL', 'mmol/L', '/uL', '%', 'ng/mL' in English/Latin characters.
3. PRESERVE TEST NAMES: You may provide the English test name followed by the translation.
4. MAINTAIN NEUTRAL, CALM TONE: No medical diagnoses, no medication suggestions.
5. PRESERVE URGENT BANNER AND DISCLAIMER:
   - If an urgent banner is present at the beginning, translate its urgency faithfully.
   - The ending disclaimer must remain prominent.
"""


class TranslateSummaryInput(BaseModel):
    """Input payload for translate_summary tool."""
    text: str = Field(description="The complete English explanation summary text to translate")
    language: str = Field(description="Target language: English, Hindi, Kannada, Telugu, or Tamil")


@default_registry.tool(
    name="translate_summary",
    description="Translates the final medical explanation into Hindi, Kannada, Telugu, or Tamil while strictly keeping all numbers and measurement units unchanged.",
    input_model=TranslateSummaryInput,
)
def translate_summary(
    text: str,
    language: str,
    llm: Optional[OpenRouterLLM] = None,
) -> Dict[str, Any]:
    """Translates explanation into target language while preserving numbers and units."""
    lang_clean = language.strip().lower()
    
    # If English or unrecognized, return original text
    if lang_clean in ("english", "en") or not text.strip():
        return {
            "translated_text": text,
            "language": "English",
            "is_translated": False,
        }

    if lang_clean not in SUPPORTED_LANGUAGES:
        return {
            "error": f"Language '{language}' is not supported. Supported: English, Hindi, Kannada, Telugu, Tamil.",
            "translated_text": text,
            "language": language,
            "is_translated": False,
        }

    client = llm or OpenRouterLLM()
    # Check if client can make call
    if not client.config.api_key and client.mock_handler is None:
        return {
            "translated_text": text,
            "language": language,
            "is_translated": False,
            "note": "OpenRouter API key not configured; returned original English text.",
        }

    system_msg = TRANSLATE_SYSTEM_PROMPT.format(language=language.capitalize())
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Please translate this verified medical explanation into {language.capitalize()}:\n\n{text}"},
    ]

    try:
        resp = client.chat(messages=messages, temperature=0.2)
        translated = resp.message.get("content", text).strip()
        return {
            "translated_text": translated,
            "language": language.capitalize(),
            "is_translated": True,
        }
    except Exception as e:
        return {
            "error": f"Translation failed: {e}",
            "translated_text": text,
            "language": language,
            "is_translated": False,
        }
