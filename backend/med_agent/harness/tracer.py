"""Tracer for recording agent runs to JSON Lines files and console previews.

Design Principles:
1. Complete Observability: Writes every agent event (start, model, tool, verify, critic, end)
   to `traces/<timestamp>-<run_id>.jsonl`.
2. Privacy Redaction: When `TRACE_REDACT=True`, automatically redacts patient names,
   MRN/patient IDs, and sensitive personal identifiers from written traces.
3. Console Preview: Prints compact, human-readable terminal summaries so developers
   and students can watch the loop's progression in real-time.
"""

import json
import os
import re
import time
import uuid
from typing import Any, Dict, Optional


class Tracer:
    """Logs structured events to JSONL files and provides console status messages."""

    def __init__(self, trace_dir: str = "traces", redact: bool = True):
        self.trace_dir = trace_dir
        self.redact = redact
        self.run_id = uuid.uuid4().hex[:8]
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.trace_filename = f"{self.timestamp}-{self.run_id}.jsonl"
        self.filepath = os.path.join(self.trace_dir, self.trace_filename)
        
        # Ensure trace directory exists
        os.makedirs(self.trace_dir, exist_ok=True)

    def _redact_text(self, text: str) -> str:
        """Masks patient names, IDs, dates of birth, and phone numbers if redaction is enabled."""
        if not self.redact or not isinstance(text, str):
            return text

        # Redact common patient demographic labels
        redacted = re.sub(
            r"(?i)(patient\s*(?:name)?|name|mrn|patient\s*id|id|uhid)\s*[:=]\s*([^\n,;]+)",
            r"\1: [REDACTED]",
            text,
        )
        # Redact phone numbers
        redacted = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", redacted)
        # Redact dates of birth
        redacted = re.sub(r"(?i)(dob|date\s*of\s*birth)\s*[:=]\s*([^\n,;]+)", r"\1: [REDACTED]", redacted)
        return redacted

    def _clean_data(self, data: Any) -> Any:
        """Recursively cleans and redacts data structures before serialization."""
        if not self.redact:
            return data
        if isinstance(data, str):
            return self._redact_text(data)
        elif isinstance(data, dict):
            return {k: self._clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        return data

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Appends an event to the JSONL trace file and prints a console preview."""
        event = {
            "timestamp": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "type": event_type,
            "data": self._clean_data(payload),
        }

        # Write to JSONL
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[TRACER ERROR] Failed writing trace: {e}")

        # Console preview
        self._print_console_preview(event_type, payload)

    def _print_console_preview(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Outputs an aesthetic terminal status line."""
        prefix = f"[TRACE {self.run_id}]"
        try:
            if event_type == "start":
                print(f"{prefix} >> Starting run | Report chars: {payload.get('report_length', 0)}")
            elif event_type == "model":
                step = payload.get("step", "?")
                tool_calls = payload.get("tool_calls", [])
                tokens = payload.get("usage", {}).get("total_tokens", 0)
                if tool_calls:
                    call_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                    print(f"{prefix} [Step {step}] Model requested {len(tool_calls)} tool(s): {', '.join(call_names)} | Tokens: {tokens}")
                else:
                    print(f"{prefix} [Step {step}] Model drafted text response | Tokens: {tokens}")
            elif event_type == "tool":
                tool_name = payload.get("name", "")
                is_error = payload.get("is_error", False)
                status = "[ERROR]" if is_error else "[OK]"
                print(f"{prefix} Tool '{tool_name}' -> {status}")
            elif event_type == "verify":
                passed = payload.get("passed", False)
                issues = payload.get("issues", [])
                status = "[PASSED]" if passed else f"[REJECTED: {len(issues)} issues]"
                print(f"{prefix} Verifier: {status}")
            elif event_type == "critic":
                passed = payload.get("passed", False)
                issues = payload.get("issues", [])
                status = "[PASSED]" if passed else f"[REJECTED: {len(issues)} issues]"
                print(f"{prefix} LLM Critic: {status}")
            elif event_type == "end":
                reason = payload.get("stop_reason", "unknown")
                steps = payload.get("steps", 0)
                tokens = payload.get("total_tokens", 0)
                print(f"{prefix} Finished | Status: {reason} | Steps: {steps} | Total Tokens: {tokens}")
        except Exception:
            pass
