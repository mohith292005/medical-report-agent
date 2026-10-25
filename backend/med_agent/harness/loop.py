"""The controlled agent execution loop.

Cycle: PLAN -> ACT -> OBSERVE -> VERIFY -> DECIDE

Hard Constraints & Safety Controls:
1. No LangChain, LlamaIndex, or third-party agent frameworks. We control the loop ourselves.
2. Safety Order Enforcement: Model CANNOT finalize an answer without executing
   `flag_abnormal` and `critical_check`. Code decides the numbers.
3. Budget Checks: Stops and exits safely if `MAX_STEPS` or `MAX_TOTAL_TOKENS` are exceeded.
4. Loop Detection: Caps identical failing tool calls at 2 retries.
5. Strict Verification: Deterministic rules + LLM critic verify every draft.
   After `MAX_VERIFY_RETRIES`, terminates safely with:
   "I could not produce a verified explanation. Please check the report with your lab or doctor."
6. Trace Logging: Every step is recorded in `traces/<timestamp>-<run_id>.jsonl`.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from med_agent.harness.config import AgentConfig, DEFAULT_CONFIG
from med_agent.harness.llm import OpenRouterLLM
from med_agent.harness.prompts import SYSTEM_PROMPT
from med_agent.harness.registry import ToolRegistry, default_registry
import med_agent.tools  # Ensures all 9 tools are automatically registered
from med_agent.harness.state import RunState
from med_agent.harness.tracer import Tracer
from med_agent.harness.verify import verify_deterministic, verify_critic
from med_agent.tools.translate import translate_summary

logger = logging.getLogger(__name__)

SAFE_FAIL_MESSAGE = "I could not produce a verified explanation. Please check the report with your lab or doctor."
BUDGET_EXCEEDED_MESSAGE = "I could not complete the report analysis within the allocated safety budget. Please consult your physician directly with your lab report."


@dataclass
class AgentResult:
    """The final output bundle returned to CLI and API consumers."""
    final: str
    stop_reason: str
    flags: List[Dict[str, Any]]
    critical_findings: List[Dict[str, Any]]
    steps: int
    tokens_used: int
    trace_file: str


def run_agent(
    report_text: str = "",
    file_path: Optional[str] = None,
    language: str = "English",
    previous_report_text: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    llm: Optional[OpenRouterLLM] = None,
    registry: Optional[ToolRegistry] = None,
    force_deterministic_pipeline: bool = False,
) -> AgentResult:
    """Executes the controlled PLAN -> ACT -> OBSERVE -> VERIFY -> DECIDE loop.
    
    Args:
        report_text: Raw medical report string (if pasted).
        file_path: Path to report file (PDF, TXT, image) if uploaded.
        language: Target explanation language (English, Hindi, Kannada, Telugu, Tamil).
        previous_report_text: Optional previous report text for longitudinal trend comparison.
        config: Runtime configuration / budget limits.
        llm: OpenRouterLLM client instance.
        registry: ToolRegistry instance (defaults to default_registry).
        force_deterministic_pipeline: When True (or when no LLM key is present),
            runs deterministic tool chain directly before drafting explanation.
            
    Returns:
        AgentResult with verified explanation, flags table, and run telemetry.
    """
    cfg = config or DEFAULT_CONFIG
    client = llm or OpenRouterLLM(config=cfg)
    reg = registry or default_registry
    tracer = Tracer(trace_dir=cfg.trace_dir, redact=cfg.trace_redact)

    # Initialize state
    state = RunState(report_text=report_text, trace_file=tracer.filepath)

    tracer.log_event("start", {
        "report_length": len(report_text),
        "file_path": file_path,
        "language": language,
        "model": cfg.model,
        "max_steps": cfg.max_steps,
        "max_tokens": cfg.max_total_tokens,
    })

    # Prepare initial messages
    user_initial_msg = ""
    if file_path:
        user_initial_msg += f"Please analyze this lab report file: {file_path}\n"
    if report_text:
        user_initial_msg += f"Report text content:\n{report_text}\n"
    if previous_report_text:
        user_initial_msg += f"\nPrevious report for trend comparison:\n{previous_report_text}\n"
    if language and language.lower() not in ("english", "en"):
        user_initial_msg += f"\nPlease explain the report in {language}.\n"

    state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_initial_msg.strip()},
    ]

    # Repeated failing tool call tracker for loop detection: (tool_name, args_repr) -> count
    failed_tool_counts: Dict[str, int] = {}

    # Check if we should run autonomous deterministic pipeline first
    # This guarantees tools are run even if LLM is offline or model jumps straight to drafting
    needs_llm_loop = (bool(client.config.api_key) or client.mock_handler is not None) and not force_deterministic_pipeline

    if not needs_llm_loop:
        # Deterministic autonomous execution path (ideal for student testing & offline grading)
        return _run_deterministic_pipeline(
            state=state,
            file_path=file_path,
            language=language,
            previous_report_text=previous_report_text,
            cfg=cfg,
            client=client,
            tracer=tracer,
        )

    # Standard LLM-driven tool calling loop
    while state.steps < cfg.max_steps:
        # 1. Budget check
        if state.tokens_used >= cfg.max_total_tokens:
            state.stop_reason = "budget_exceeded"
            state.final = BUDGET_EXCEEDED_MESSAGE
            tracer.log_event("end", {
                "stop_reason": state.stop_reason,
                "steps": state.steps,
                "total_tokens": state.tokens_used,
            })
            return _build_result(state)

        state.steps += 1

        # 2. ACT: Call LLM with tool schemas
        tools_schema = reg.schemas()
        try:
            resp = client.chat(messages=state.messages, tools=tools_schema)
        except Exception as e:
            logger.error("LLM execution error during step %d: %s", state.steps, e)
            state.stop_reason = "error"
            state.final = SAFE_FAIL_MESSAGE
            tracer.log_event("end", {"stop_reason": "error", "error": str(e)})
            return _build_result(state)

        # Update token counters
        state.tokens_used += resp.usage.total_tokens

        assistant_msg = resp.message
        tool_calls = assistant_msg.get("tool_calls")

        tracer.log_event("model", {
            "step": state.steps,
            "tool_calls": tool_calls,
            "has_content": bool(assistant_msg.get("content")),
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        })

        # Append assistant message to history
        state.messages.append(assistant_msg)

        # 3. OBSERVE: If model requested tool calls, execute them
        if tool_calls:
            for tc in tool_calls:
                call_id = tc["id"]
                tool_name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]

                # Parse JSON arguments safely
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception as parse_err:
                    err_msg = f"Error: Tool arguments must be valid JSON: {parse_err}"
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": err_msg,
                    })
                    state.record_tool_result(tool_name, {}, err_msg, is_error=True)
                    tracer.log_event("tool", {"name": tool_name, "is_error": True, "error": err_msg})
                    continue

                # Loop detection: check repeated failing calls
                call_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if failed_tool_counts.get(call_signature, 0) >= 2:
                    abort_msg = f"Loop Abort: Tool '{tool_name}' failed with identical arguments twice. Skipping."
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": abort_msg,
                    })
                    tracer.log_event("tool", {"name": tool_name, "is_error": True, "error": abort_msg})
                    continue

                # Execute tool via registry
                tool_output, is_error = reg.run(tool_name, args)

                # Store tool outputs in state for verifier and safety checks
                if not is_error:
                    try:
                        parsed_json = json.loads(tool_output)
                        if tool_name == "parse_report":
                            state.report_text = parsed_json.get("text", state.report_text)
                        elif tool_name == "extract_values":
                            state.extracted_rows = parsed_json.get("rows", [])
                            if parsed_json.get("needs_review"):
                                state.stop_reason = "needs_review"
                                state.final = SAFE_FAIL_MESSAGE
                                tracer.log_event("end", {"stop_reason": "needs_review"})
                                return _build_result(state)
                        elif tool_name == "flag_abnormal":
                            state.flags = parsed_json.get("flagged_rows", [])
                        elif tool_name == "critical_check":
                            state.critical_findings = parsed_json.get("critical_findings", [])
                    except Exception:
                        pass
                else:
                    failed_tool_counts[call_signature] = failed_tool_counts.get(call_signature, 0) + 1

                # Record in state and tracer
                state.record_tool_result(tool_name, args, tool_output, is_error=is_error)
                tracer.log_event("tool", {"name": tool_name, "args": args, "is_error": is_error})

                # Append tool response message
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output,
                })

            # Continue to next iteration of loop to let model observe tool outputs
            continue

        # 4. VERIFY: Model produced a text draft without requesting tools
        draft = assistant_msg.get("content", "").strip()

        # Hard safety enforcement: Model cannot finalize without having called flag_abnormal & critical_check
        has_flagged = "flag_abnormal" in state.tool_outputs and any(not e["is_error"] for e in state.tool_outputs["flag_abnormal"])
        has_critical = "critical_check" in state.tool_outputs and any(not e["is_error"] for e in state.tool_outputs["critical_check"])

        if not has_flagged or not has_critical:
            state.verify_retries += 1
            rejection_note = (
                "Safety Constraint Rejection: You cannot produce a final explanation without having executed "
                "'flag_abnormal' and 'critical_check'. Code decides the numbers; the LLM only explains them. "
                "Please call these tools now."
            )
            if state.verify_retries > cfg.max_verify_retries:
                state.stop_reason = "verify_failed"
                state.final = SAFE_FAIL_MESSAGE
                tracer.log_event("end", {"stop_reason": state.stop_reason})
                return _build_result(state)

            state.messages.append({"role": "user", "content": rejection_note})
            tracer.log_event("verify", {"passed": False, "issues": [rejection_note]})
            continue

        # Run Deterministic Verifier
        det_passed, det_issues = verify_deterministic(draft=draft, state=state)
        tracer.log_event("verify", {"passed": det_passed, "issues": det_issues})

        if not det_passed:
            state.verify_retries += 1
            if state.verify_retries > cfg.max_verify_retries:
                state.stop_reason = "verify_failed"
                state.final = SAFE_FAIL_MESSAGE
                tracer.log_event("end", {"stop_reason": state.stop_reason})
                return _build_result(state)

            # Feed specific rejection issues back to the model for self-correction
            issues_formatted = "\n- ".join(det_issues)
            feedback = (
                f"Your draft explanation failed deterministic safety verification for the following reasons:\n"
                f"- {issues_formatted}\n\n"
                f"Please revise your draft to strictly address every issue above."
            )
            state.messages.append({"role": "user", "content": feedback})
            continue

        # Run LLM Critic (only after deterministic verifier passes)
        critic_passed, critic_issues = verify_critic(draft=draft, state=state, llm=client)
        tracer.log_event("critic", {"passed": critic_passed, "issues": critic_issues})

        if not critic_passed:
            state.verify_retries += 1
            if state.verify_retries > cfg.max_verify_retries:
                state.stop_reason = "verify_failed"
                state.final = SAFE_FAIL_MESSAGE
                tracer.log_event("end", {"stop_reason": state.stop_reason})
                return _build_result(state)

            issues_formatted = "\n- ".join(critic_issues)
            feedback = (
                f"Your draft explanation was rejected by the Medical Safety Critic for the following reasons:\n"
                f"- {issues_formatted}\n\n"
                f"Please revise your explanation to fix these clinical and safety issues."
            )
            state.messages.append({"role": "user", "content": feedback})
            continue

        # 5. DECIDE: Both Verifier and Critic passed!
        # Translate if necessary
        final_text = draft
        if language and language.lower() not in ("english", "en"):
            trans_res = translate_summary(text=draft, language=language, llm=client)
            final_text = trans_res.get("translated_text", draft)

        state.final = final_text
        state.stop_reason = "completed"
        tracer.log_event("end", {
            "stop_reason": state.stop_reason,
            "steps": state.steps,
            "total_tokens": state.tokens_used,
        })
        return _build_result(state)

    # If loop exited without finishing, max steps reached
    state.stop_reason = "budget_exceeded"
    state.final = BUDGET_EXCEEDED_MESSAGE
    tracer.log_event("end", {
        "stop_reason": state.stop_reason,
        "steps": state.steps,
        "total_tokens": state.tokens_used,
    })
    return _build_result(state)


def _run_deterministic_pipeline(
    state: RunState,
    file_path: Optional[str],
    language: str,
    previous_report_text: Optional[str],
    cfg: AgentConfig,
    client: OpenRouterLLM,
    tracer: Tracer,
) -> AgentResult:
    """Deterministic tool chain execution for offline grading, tests, and robust fallbacks."""
    from med_agent.tools.parse import parse_report
    from med_agent.tools.extract import extract_values
    from med_agent.tools.units import normalize_units, UnitRowInput
    from med_agent.tools.flag import flag_abnormal, LabValueInput
    from med_agent.tools.critical import critical_check, CriticalRowInput
    from med_agent.tools.glossary import explain_term
    from med_agent.tools.questions import doctor_questions, FlaggedItemInput
    from med_agent.tools.trend import compare_trend, TrendRowInput

    state.steps = 1

    # 1. Parse report if file provided
    if file_path:
        p_res = parse_report(file_path)
        state.report_text = p_res.get("text", state.report_text)
        state.record_tool_result("parse_report", {"path": file_path}, p_res, is_error=bool(p_res.get("error")))
        tracer.log_event("tool", {"name": "parse_report", "is_error": bool(p_res.get("error"))})

    # 2. Extract values
    e_res = extract_values(text=state.report_text, llm=client)
    state.extracted_rows = e_res.get("rows", [])
    state.record_tool_result("extract_values", {"text": "..." }, e_res, is_error=False)
    tracer.log_event("tool", {"name": "extract_values", "is_error": False})

    if e_res.get("needs_review") or not state.extracted_rows:
        state.stop_reason = "needs_review"
        state.final = SAFE_FAIL_MESSAGE
        tracer.log_event("end", {"stop_reason": "needs_review"})
        return _build_result(state)

    # 3. Normalize units
    unit_inputs = [UnitRowInput(**r) for r in state.extracted_rows]
    u_res = normalize_units(unit_inputs)
    state.extracted_rows = u_res.get("rows", state.extracted_rows)
    state.record_tool_result("normalize_units", {}, u_res, is_error=False)
    tracer.log_event("tool", {"name": "normalize_units", "is_error": False})

    # 4. Flag abnormal
    flag_inputs = [
        LabValueInput(
            test=r["test"],
            value=r["value"],
            unit=r.get("unit"),
            ref_text=r.get("ref_text"),
            ref_low=r.get("ref_low"),
            ref_high=r.get("ref_high"),
        )
        for r in state.extracted_rows
    ]
    f_res = flag_abnormal(flag_inputs)
    state.flags = f_res.get("flagged_rows", [])
    state.record_tool_result("flag_abnormal", {}, f_res, is_error=False)
    tracer.log_event("tool", {"name": "flag_abnormal", "is_error": False})

    # 5. Critical check
    crit_inputs = [
        CriticalRowInput(test=r["test"], value=r["value"], unit=r.get("unit"))
        for r in state.extracted_rows
    ]
    c_res = critical_check(crit_inputs)
    state.critical_findings = c_res.get("critical_findings", [])
    state.record_tool_result("critical_check", {}, c_res, is_error=False)
    tracer.log_event("tool", {"name": "critical_check", "is_error": False})

    # 6. Explain terms for flagged tests
    glossary_notes: Dict[str, str] = {}
    for fl in state.flags:
        if fl.get("flag") in ("LOW", "HIGH"):
            t_name = fl.get("test", "")
            g_res = explain_term(t_name)
            state.record_tool_result("explain_term", {"term": t_name}, g_res, is_error=False)
            if g_res.get("found"):
                meaning = g_res.get("general_meaning", {}).get(fl.get("flag").lower(), "")
                glossary_notes[t_name] = f"{g_res.get('what_it_measures')} {meaning}".strip()

    # 7. Doctor questions
    dq_inputs = [
        FlaggedItemInput(test=r["test"], flag=r["flag"], value=r.get("value"), unit=r.get("unit"))
        for r in state.flags
    ]
    dq_res = doctor_questions(dq_inputs)
    questions = dq_res.get("questions", [])
    state.record_tool_result("doctor_questions", {}, dq_res, is_error=False)

    # 8. Longitudinal trend if previous provided
    trend_summary = ""
    if previous_report_text:
        prev_extract = extract_values(text=previous_report_text, llm=client)
        prev_rows = prev_extract.get("rows", [])
        if prev_rows:
            c_trends = [TrendRowInput(**r) for r in state.extracted_rows]
            p_trends = [TrendRowInput(**r) for r in prev_rows]
            tr_res = compare_trend(c_trends, p_trends)
            state.record_tool_result("compare_trend", {}, tr_res, is_error=False)

    # Assemble structured, calm explanation compliant with all safety rules
    sections = []

    # Urgent banner (if critical findings exist)
    if state.critical_findings:
        sections.append("One or more values are in a range that usually needs prompt medical attention. Please contact your doctor or a hospital today.\n")

    sections.append("### Overview")
    sections.append(f"This summary explains {len(state.flags)} laboratory test results from your report in everyday language.")

    # In range tests
    normal_tests = [f for f in state.flags if f.get("flag") == "NORMAL"]
    if normal_tests:
        sections.append("\n### In range")
        for nt in normal_tests:
            unit_str = f" {nt['unit']}" if nt.get("unit") else ""
            ref_str = f" (reference: {nt['ref_low']} - {nt['ref_high']}{unit_str})" if nt.get("ref_low") is not None else ""
            sections.append(f"- **{nt['test']}**: {nt['value']}{unit_str}{ref_str} is within the expected normal range.")

    # Needs attention tests
    abnormal_tests = [f for f in state.flags if f.get("flag") in ("LOW", "HIGH")]
    if abnormal_tests:
        sections.append("\n### Needs attention")
        for at in abnormal_tests:
            unit_str = f" {at['unit']}" if at.get("unit") else ""
            ref_str = f" (reference: {at['ref_low']} - {at['ref_high']}{unit_str})" if at.get("ref_low") is not None else ""
            flag_desc = "lower than" if at["flag"] == "LOW" else "higher than"
            explanation = glossary_notes.get(at["test"], "Discuss this reading with your doctor for context.")
            sections.append(
                f"- **{at['test']}**: {at['value']}{unit_str} is {flag_desc} the standard reference limits{ref_str}. {explanation}"
            )

    # Doctor questions
    if questions:
        sections.append("\n### Questions to ask your doctor")
        for q in questions:
            sections.append(f"- {q}")

    # Mandatory disclaimer
    sections.append("\nThis is an explanation, not medical advice. Please discuss results with your doctor.")

    draft_text = "\n".join(sections)

    # Deterministic verifier
    det_passed, det_issues = verify_deterministic(draft=draft_text, state=state)
    tracer.log_event("verify", {"passed": det_passed, "issues": det_issues})

    if not det_passed:
        state.stop_reason = "verify_failed"
        state.final = SAFE_FAIL_MESSAGE
        tracer.log_event("end", {"stop_reason": state.stop_reason})
        return _build_result(state)

    # Translation if requested
    if language and language.lower() not in ("english", "en"):
        trans_res = translate_summary(text=draft_text, language=language, llm=client)
        draft_text = trans_res.get("translated_text", draft_text)

    state.final = draft_text
    state.stop_reason = "completed"
    tracer.log_event("end", {
        "stop_reason": state.stop_reason,
        "steps": state.steps,
        "total_tokens": state.tokens_used,
    })
    return _build_result(state)


def _build_result(state: RunState) -> AgentResult:
    """Helper to convert RunState into the final AgentResult object."""
    return AgentResult(
        final=state.final,
        stop_reason=state.stop_reason,
        flags=state.flags,
        critical_findings=state.critical_findings,
        steps=state.steps,
        tokens_used=state.tokens_used,
        trace_file=state.trace_file,
    )
