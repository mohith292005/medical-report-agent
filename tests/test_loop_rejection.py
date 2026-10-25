"""Tests demonstrating verifier rejection, self-correction retry, and trace logging."""

import json
import os
import pytest

from med_agent.harness.config import AgentConfig
from med_agent.harness.llm import OpenRouterLLM, LLMResponse, TokenUsage
from med_agent.harness.loop import run_agent, SAFE_FAIL_MESSAGE


def test_loop_rejection_and_self_correction_retry(tmp_path):
    """Deliberately simulates a bad draft with diagnosis and medication advice,
    confirms rejection in trace, and verifies self-correction on retry."""

    trace_dir = str(tmp_path / "traces")
    config = AgentConfig(trace_dir=trace_dir, max_verify_retries=2)

    call_count = 0

    def mock_agent_behavior(messages, tools, json_mode):
        nonlocal call_count
        call_count += 1

        # Check last user message to see if verifier feedback was received
        last_msg = messages[-1]["content"] if messages else ""

        # Step 1: Model calls tools (flag_abnormal and critical_check)
        if call_count == 1:
            return LLMResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_flag",
                            "type": "function",
                            "function": {
                                "name": "flag_abnormal",
                                "arguments": json.dumps({
                                    "rows": [
                                        {"test": "Hemoglobin", "value": 9.2, "unit": "g/dL", "ref_low": 13.0, "ref_high": 17.0}
                                    ]
                                }),
                            },
                        },
                        {
                            "id": "call_crit",
                            "type": "function",
                            "function": {
                                "name": "critical_check",
                                "arguments": json.dumps({
                                    "rows": [{"test": "Hemoglobin", "value": 9.2, "unit": "g/dL"}]
                                }),
                            },
                        },
                    ],
                },
                usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            )

        # Step 2: Deliberately broken draft (violates safety: diagnosis & medication)
        elif call_count == 2:
            bad_draft = (
                "You have anemia and your diagnosis is iron deficiency. "
                "You should take 325 mg ferrous sulfate daily. "
                "Hemoglobin is 9.2 g/dL."
            )
            return LLMResponse(
                message={"role": "assistant", "content": bad_draft},
                usage=TokenUsage(prompt_tokens=70, completion_tokens=25, total_tokens=95),
            )

        # Step 3: Self-corrected compliant draft after observing verifier rejection
        else:
            assert "failed deterministic safety verification" in last_msg
            good_draft = (
                "### Overview\n"
                "Your Complete Blood Count was analyzed.\n\n"
                "### Needs attention\n"
                "- Hemoglobin: 9.2 g/dL is lower than the standard reference range (13.0 - 17.0 g/dL).\n\n"
                "### Questions to ask your doctor\n"
                "- What dietary factors might explain this hemoglobin level?\n\n"
                "This is an explanation, not medical advice. Please discuss results with your doctor."
            )
            return LLMResponse(
                message={"role": "assistant", "content": good_draft},
                usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
            )

    mock_llm = OpenRouterLLM(config=config, mock_handler=mock_agent_behavior)

    result = run_agent(
        report_text="Hemoglobin: 9.2 g/dL (13.0 - 17.0)",
        config=config,
        llm=mock_llm,
    )

    # Verifications:
    assert result.stop_reason == "completed"
    assert "This is an explanation, not medical advice" in result.final
    assert "you have" not in result.final.lower()

    # Check trace file for rejection and retry events
    assert os.path.exists(result.trace_file)
    with open(result.trace_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    verify_events = [e for e in events if e["type"] == "verify"]
    assert len(verify_events) >= 2
    # First verify event must be rejected
    assert verify_events[0]["data"]["passed"] is False
    assert any("you have" in issue.lower() or "diagnosis" in issue.lower() for issue in verify_events[0]["data"]["issues"])
    # Second verify event must be passed
    assert verify_events[1]["data"]["passed"] is True


def test_exhausted_verify_retries_returns_safe_message(tmp_path):
    """Confirms that after exceeding MAX_VERIFY_RETRIES, the harness exits safely."""
    trace_dir = str(tmp_path / "traces")
    config = AgentConfig(trace_dir=trace_dir, max_verify_retries=1)

    call_count = 0

    def stubbornly_bad_agent(messages, tools, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "flag_abnormal",
                                "arguments": json.dumps({"rows": [{"test": "Glucose", "value": 150.0}]}),
                            },
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "critical_check",
                                "arguments": json.dumps({"rows": [{"test": "Glucose", "value": 150.0}]}),
                            },
                        },
                    ],
                },
                usage=TokenUsage(10, 10, 20),
            )
        # Always stubbornly emits diagnosis
        return LLMResponse(
            message={"role": "assistant", "content": "You have severe diabetes!"},
            usage=TokenUsage(10, 10, 20),
        )

    llm = OpenRouterLLM(config=config, mock_handler=stubbornly_bad_agent)
    result = run_agent(report_text="Glucose: 150 mg/dL", config=config, llm=llm)

    assert result.stop_reason == "verify_failed"
    assert result.final == SAFE_FAIL_MESSAGE
