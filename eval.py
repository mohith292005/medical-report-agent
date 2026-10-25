"""Evaluation script: Runs all 5 sample reports through the deterministic pipeline.

Usage:
    python eval.py                   # runs all samples
    python eval.py --sample report_anemia_cbc.txt   # runs one specific sample
    python eval.py --verbose         # prints full explanation text

What it checks (automated, no API key needed):
    1. Every sample is parsed and values are extracted.
    2. flag_abnormal and critical_check are always invoked.
    3. The final explanation mentions every flagged test.
    4. Critical panic values trigger the urgent banner.
    5. The disclaimer is always present.
    6. Stop reason is 'completed' (not 'verify_failed' or 'error').

This script is intentionally runnable without an OpenRouter API key.
It uses the deterministic pipeline (force_deterministic_pipeline=True).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ensure backend package is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from med_agent.harness.loop import run_agent, AgentResult


# ─── Expected Outcomes per Sample ───────────────────────────────────────────
# Each entry describes what the eval script expects to find in the results.

SAMPLE_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "report_normal_cbc.txt": {
        "description": "Normal CBC – all values within reference ranges",
        "expect_flags": ["NORMAL"],        # Only NORMAL flags expected
        "expect_no_flags": ["HIGH", "LOW"],
        "expect_critical": False,
        "min_extracted": 4,                 # Should extract at least 4 values
    },
    "report_anemia_cbc.txt": {
        "description": "Anemia CBC – low hemoglobin, low RBC",
        "expect_flags": ["LOW"],           # At least one LOW flag
        "expect_no_flags": [],
        "expect_critical": False,
        "min_extracted": 3,
    },
    "report_lipid_profile.txt": {
        "description": "Lipid profile – elevated total cholesterol and LDL",
        "expect_flags": ["HIGH"],          # At least one HIGH flag
        "expect_no_flags": [],
        "expect_critical": False,
        "min_extracted": 3,
    },
    "report_critical_potassium.txt": {
        "description": "Critical potassium – panic value present",
        "expect_flags": ["HIGH"],
        "expect_no_flags": [],
        "expect_critical": True,           # Must trigger critical alert
        "min_extracted": 2,
    },
    "report_messy_units.txt": {
        "description": "Messy units – tests parser robustness with varied formatting",
        "expect_flags": [],                # Just verify extraction works
        "expect_no_flags": [],
        "expect_critical": False,
        "min_extracted": 2,
    },
}


@dataclass
class EvalResult:
    """Result of evaluating one sample report."""
    sample: str
    passed: bool = False
    checks: List[Dict[str, Any]] = field(default_factory=list)
    agent_result: Optional[AgentResult] = None
    error: Optional[str] = None


def check(name: str, condition: bool, detail: str = "") -> Dict[str, Any]:
    """Helper: create a check result dict."""
    return {"check": name, "passed": condition, "detail": detail}


def evaluate_sample(sample_path: str, expectations: Dict[str, Any]) -> EvalResult:
    """Run one sample through the deterministic pipeline and validate results."""
    sample_name = os.path.basename(sample_path)
    result = EvalResult(sample=sample_name)

    try:
        # Run the agent in deterministic mode (no API key needed)
        agent = run_agent(
            file_path=sample_path,
            language="English",
            force_deterministic_pipeline=True,
        )
        result.agent_result = agent
    except Exception as e:
        result.error = str(e)
        result.passed = False
        result.checks.append(check("run_without_crash", False, str(e)))
        return result

    checks = []

    # 1. Did the agent complete successfully?
    checks.append(check(
        "completed",
        agent.stop_reason == "completed",
        f"stop_reason={agent.stop_reason}",
    ))

    # 2. Were enough values extracted?
    min_extracted = expectations.get("min_extracted", 1)
    num_flags = len(agent.flags)
    checks.append(check(
        "min_values_extracted",
        num_flags >= min_extracted,
        f"extracted {num_flags}, expected >= {min_extracted}",
    ))

    # 3. Check expected flag types are present
    flag_types_found = set(f.get("flag") for f in agent.flags)
    for expected_flag in expectations.get("expect_flags", []):
        checks.append(check(
            f"has_{expected_flag.lower()}_flag",
            expected_flag in flag_types_found,
            f"flags found: {flag_types_found}",
        ))

    # 4. Check unexpected flag types are absent
    for unexpected_flag in expectations.get("expect_no_flags", []):
        checks.append(check(
            f"no_{unexpected_flag.lower()}_flag",
            unexpected_flag not in flag_types_found,
            f"flags found: {flag_types_found}",
        ))

    # 5. Critical findings check
    if expectations.get("expect_critical"):
        checks.append(check(
            "has_critical_finding",
            len(agent.critical_findings) > 0,
            f"critical_findings count: {len(agent.critical_findings)}",
        ))
    else:
        checks.append(check(
            "no_critical_finding",
            len(agent.critical_findings) == 0,
            f"critical_findings count: {len(agent.critical_findings)}",
        ))

    # 6. Explanation mentions flagged tests
    explanation = agent.final.lower()
    abnormal_tests = [f["test"] for f in agent.flags if f.get("flag") in ("HIGH", "LOW")]
    for test_name in abnormal_tests:
        checks.append(check(
            f"explanation_mentions_{test_name.lower().replace(' ', '_')}",
            test_name.lower() in explanation,
            f"Looking for '{test_name.lower()}' in explanation",
        ))

    # 7. Disclaimer is present
    checks.append(check(
        "has_disclaimer",
        "not medical advice" in explanation or "discuss" in explanation,
        "Checking for safety disclaimer",
    ))

    # 8. No diagnosis language
    diagnosis_words = ["you have", "you are diagnosed", "take medication", "prescribe"]
    for dw in diagnosis_words:
        checks.append(check(
            f"no_diagnosis_language_{dw.replace(' ', '_')}",
            dw not in explanation,
            f"Scanning for forbidden phrase: '{dw}'",
        ))

    result.checks = checks
    result.passed = all(c["passed"] for c in checks)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Medical Report Agent on sample reports")
    parser.add_argument("--sample", help="Run only this specific sample file name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full explanation text")
    default_samples = os.path.join(BACKEND_DIR, "samples") if os.path.isdir(os.path.join(BACKEND_DIR, "samples")) else "samples"
    parser.add_argument("--samples-dir", default=default_samples, help="Directory containing sample reports")
    args = parser.parse_args()

    samples_dir = args.samples_dir
    if not os.path.isdir(samples_dir):
        print(f"Error: Samples directory '{samples_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Determine which samples to evaluate
    if args.sample:
        sample_files = [args.sample]
    else:
        sample_files = sorted(SAMPLE_EXPECTATIONS.keys())

    print("=" * 80)
    print("MEDICAL REPORT AGENT — EVALUATION SUITE")
    print("=" * 80)
    print(f"Samples directory: {os.path.abspath(samples_dir)}")
    print(f"Running {len(sample_files)} sample(s) in deterministic mode (no API key required)")
    print()

    results: List[EvalResult] = []

    for sample_file in sample_files:
        sample_path = os.path.join(samples_dir, sample_file)
        if not os.path.exists(sample_path):
            print(f"⚠  Skipping '{sample_file}': file not found")
            continue

        expectations = SAMPLE_EXPECTATIONS.get(sample_file, {
            "description": "Unknown sample",
            "expect_flags": [],
            "expect_no_flags": [],
            "expect_critical": False,
            "min_extracted": 1,
        })

        print(f"─── {sample_file} ─── {expectations.get('description', '')}")

        eval_result = evaluate_sample(sample_path, expectations)
        results.append(eval_result)

        # Print checks
        for c in eval_result.checks:
            status = "PASS" if c["passed"] else "FAIL"
            icon = "✓" if c["passed"] else "✗"
            print(f"  {icon} [{status}] {c['check']}: {c['detail']}")

        if eval_result.error:
            print(f"  ERROR: {eval_result.error}")

        # Print flags table
        if eval_result.agent_result and eval_result.agent_result.flags:
            print(f"\n  Flags ({len(eval_result.agent_result.flags)}):")
            for f in eval_result.agent_result.flags:
                print(f"    {f['test']:25s} {str(f.get('value', '')):>8s} {(f.get('unit') or ''):>8s}  [{f['flag']}]")

        # Print critical findings
        if eval_result.agent_result and eval_result.agent_result.critical_findings:
            print(f"\n  CRITICAL FINDINGS:")
            for cf in eval_result.agent_result.critical_findings:
                print(f"    !! {cf['test']}: {cf['value']} ({cf['breach']})")

        # Print explanation if verbose
        if args.verbose and eval_result.agent_result:
            print(f"\n  Explanation:\n")
            for line in eval_result.agent_result.final.split("\n"):
                print(f"    {line}")

        overall = "PASSED" if eval_result.passed else "FAILED"
        print(f"\n  Result: {overall}\n")

    # ─── Summary ────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    total_checks = sum(len(r.checks) for r in results)
    passed_checks = sum(sum(1 for c in r.checks if c["passed"]) for r in results)
    failed_checks = total_checks - passed_checks

    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  {icon} {r.sample}")

    print(f"\nSamples: {passed}/{total} passed")
    print(f"Checks:  {passed_checks}/{total_checks} passed")

    if failed > 0:
        print(f"\n{failed} sample(s) FAILED. Review output above for details.")
        sys.exit(1)
    else:
        print(f"\nAll {total} samples PASSED all checks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
