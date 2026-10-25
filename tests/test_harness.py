"""Tests for the agent harness: config, state, registry, tracer, and llm wrapper."""

import json
import os
import pytest
from pydantic import BaseModel, Field

from med_agent.harness.config import AgentConfig
from med_agent.harness.state import RunState
from med_agent.harness.registry import ToolRegistry
from med_agent.harness.tracer import Tracer
from med_agent.harness.llm import OpenRouterLLM, LLMResponse, TokenUsage


class SampleInput(BaseModel):
    name: str
    amount: float = Field(gt=0, description="Positive amount")


def test_config_defaults():
    config = AgentConfig()
    assert config.max_steps in (8, 12)
    assert config.max_verify_retries == 2
    assert config.max_total_tokens in (40000, 60000)
    assert config.max_output_tokens == 2000
    assert config.tool_timeout_s == 20.0
    assert config.model in ("anthropic/claude-sonnet-4.5", os.getenv("OPENROUTER_MODEL", ""))


def test_run_state_lifecycle():
    state = RunState(report_text="Patient Report: Hemoglobin 11.0 g/dL")
    assert state.steps == 0
    assert state.tokens_used == 0
    assert state.stop_reason == "in_progress"

    state.record_tool_result("flag_abnormal", {"val": 11.0}, {"flag": "LOW"}, is_error=False)
    assert "flag_abnormal" in state.tool_outputs
    assert len(state.tool_outputs["flag_abnormal"]) == 1
    assert state.tool_outputs["flag_abnormal"][0]["is_error"] is False


def test_registry_registration_and_schemas():
    registry = ToolRegistry()

    @registry.tool(
        name="test_tool",
        description="A test tool for registry validation",
        input_model=SampleInput,
    )
    def my_tool(name: str, amount: float):
        return {"greeting": f"Hello {name}", "double": amount * 2}

    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "test_tool"
    assert "parameters" in schemas[0]["function"]


def test_registry_safe_execution_never_raises():
    registry = ToolRegistry()

    @registry.tool(
        name="test_tool",
        description="A test tool",
        input_model=SampleInput,
    )
    def my_tool(name: str, amount: float):
        if amount == 999.0:
            raise RuntimeError("Explosive test failure")
        return {"result": f"{name}_{amount}"}

    # 1. Successful execution
    res, is_err = registry.run("test_tool", {"name": "Alice", "amount": 10.5})
    assert is_err is False
    assert "Alice_10.5" in res

    # 2. String JSON input (OpenAI tool calls pass JSON strings)
    res, is_err = registry.run("test_tool", json.dumps({"name": "Bob", "amount": 5.0}))
    assert is_err is False
    assert "Bob_5.0" in res

    # 3. Invalid arguments (Pydantic validation error) -> catches gracefully
    res, is_err = registry.run("test_tool", {"name": "Charlie", "amount": -10.0})
    assert is_err is True
    assert "Validation Error" in res

    # 4. Unknown tool -> catches gracefully
    res, is_err = registry.run("nonexistent_tool", {})
    assert is_err is True
    assert "not registered" in res

    # 5. Tool throws internal runtime exception -> catches gracefully
    res, is_err = registry.run("test_tool", {"name": "Dave", "amount": 999.0})
    assert is_err is True
    assert "Runtime error" in res
    assert "Explosive test failure" in res


def test_tracer_and_redaction(tmp_path):
    trace_dir = str(tmp_path / "traces")
    tracer = Tracer(trace_dir=trace_dir, redact=True)

    sample_report = "Patient Name: Johnathan Doe\nPatient ID: 987654321\nHemoglobin: 13.5 g/dL"
    tracer.log_event("start", {"report_text": sample_report, "report_length": len(sample_report)})
    tracer.log_event("tool", {"name": "flag_abnormal", "is_error": False})
    tracer.log_event("end", {"stop_reason": "completed", "steps": 1})

    assert os.path.exists(tracer.filepath)
    with open(tracer.filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3
    first_event = json.loads(lines[0])
    # Check that patient name and ID were redacted
    assert "Johnathan Doe" not in first_event["data"]["report_text"]
    assert "[REDACTED]" in first_event["data"]["report_text"]


def test_llm_mock_handler():
    def mock_completion(messages, tools, json_mode):
        return LLMResponse(
            message={"role": "assistant", "content": "Simulated answer"},
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    llm = OpenRouterLLM(mock_handler=mock_completion)
    resp = llm.chat(messages=[{"role": "user", "content": "Hello"}])
    assert resp.message["content"] == "Simulated answer"
    assert resp.usage.total_tokens == 15
