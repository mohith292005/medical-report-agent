"""Agent execution state.

The RunState dataclass holds the complete lifecycle context of an agent run.
By storing raw reports, extracted values, tool results, conversational messages,
and budget counters in one place, the harness keeps state explicit, observable,
and easy to trace or inspect during grading and debugging.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunState:
    """Dataclass holding all dynamic state across the PLAN -> ACT -> OBSERVE -> VERIFY loop.
    
    Attributes:
        report_text: The raw text of the medical report (from text input or OCR).
        extracted_rows: Structured test results extracted from report text.
        flags: Output from `flag_abnormal` (LOW, NORMAL, HIGH, NO_RANGE).
        critical_findings: Output from `critical_check` listing panic thresholds breached.
        messages: The running conversation history sent to the LLM (OpenAI format).
        steps: Current iteration count of the agent loop (max is MAX_STEPS).
        tokens_used: Cumulative token tally across all LLM requests/responses.
        verify_retries: Number of times the verifier/critic rejected a draft and retried.
        tool_outputs: History of all executed tools and their parsed return objects.
        final: The final approved plain-language explanation delivered to the user.
        stop_reason: The reason the loop terminated ("completed", "verify_failed",
                     "budget_exceeded", "needs_review", "error").
        trace_file: Filepath of the JSONL trace capturing this run.
    """

    report_text: str = ""
    extracted_rows: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    tokens_used: int = 0
    verify_retries: int = 0
    tool_outputs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    final: str = ""
    stop_reason: str = "in_progress"
    trace_file: str = ""

    def record_tool_result(self, tool_name: str, args: Dict[str, Any], result: Any, is_error: bool) -> None:
        """Stores a structured log of tool execution in state."""
        if tool_name not in self.tool_outputs:
            self.tool_outputs[tool_name] = []
        self.tool_outputs[tool_name].append({
            "args": args,
            "result": result,
            "is_error": is_error,
            "step": self.steps,
        })
