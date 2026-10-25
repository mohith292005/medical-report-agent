"""Configuration settings for the Medical Report Explanation Agent.

This module loads environment variables via `python-dotenv` and defines
the operational constraints (budgets, retry limits, model parameters)
used by the agent loop.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()


@dataclass
class AgentConfig:
    """Agent runtime configuration and safety budgets.
    
    Hard constraints:
    - MAX_STEPS: Prevents the agent from looping indefinitely (infinite tool calls).
    - MAX_VERIFY_RETRIES: Prevents endless verifier-critic rejection loops.
    - MAX_TOTAL_TOKENS: Strict cost/usage budget per user report run.
    - MAX_OUTPUT_TOKENS: Cap on generated tokens per single LLM completion.
    - TOOL_TIMEOUT_S: Max seconds to wait for a tool execution before timeout.
    """
    
    # Model and API settings
    model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    api_key: str = (
        ""
        if os.getenv("OPENROUTER_API_KEY", "").strip() in ("", "your_openrouter_api_key_here", "your_key", "none")
        else os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    http_referer: str = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000")
    title: str = os.getenv("OPENROUTER_TITLE", "Medical Report Explanation Agent")

    # Safety and budget boundaries
    max_steps: int = 12
    max_verify_retries: int = 2
    max_total_tokens: int = 60000
    max_output_tokens: int = 2000
    tool_timeout_s: float = 20.0

    # Privacy / Tracing
    trace_redact: bool = os.getenv("TRACE_REDACT", "true").lower() in ("1", "true", "yes")
    trace_dir: str = os.getenv("TRACE_DIR", "traces")


# Global default configuration instance
DEFAULT_CONFIG = AgentConfig()
