"""Command-line interface (CLI) for the Medical Report Explanation Agent.

Usage:
    python main.py samples/report_normal_cbc.txt
    python main.py samples/report_anemia_cbc.txt --language Hindi
    python main.py samples/report_critical_potassium.txt
"""

import argparse
import os
import sys

# Ensure backend package is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from med_agent.harness.loop import run_agent


def main():
    parser = argparse.ArgumentParser(
        description="Medical Report Explanation Agent - Safe, Plain-Language Lab Report Explanations"
    )
    parser.add_argument(
        "report",
        help="Path to medical report file (.txt, .pdf, .png, .jpg)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="English",
        help="Target language: English, Hindi, Kannada, Telugu, Tamil (default: English)",
    )
    parser.add_argument(
        "--previous",
        "-p",
        default=None,
        help="Optional path to a previous report file for longitudinal trend analysis",
    )
    parser.add_argument(
        "--deterministic",
        "-d",
        action="store_true",
        help="Force deterministic offline pipeline (useful for offline grading/testing)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.report):
        alt_path = os.path.join("backend", args.report)
        if os.path.exists(alt_path):
            args.report = alt_path
        else:
            print(f"Error: Report file not found at '{args.report}'", file=sys.stderr)
            sys.exit(1)

    previous_text = None
    if args.previous and os.path.exists(args.previous):
        with open(args.previous, "r", encoding="utf-8", errors="replace") as pf:
            previous_text = pf.read()

    print("=" * 70)
    print("MEDICAL REPORT EXPLANATION AGENT")
    print(f"Analyzing: {args.report}")
    print(f"Language:  {args.language}")
    print("=" * 70)

    result = run_agent(
        file_path=args.report,
        language=args.language,
        previous_report_text=previous_text,
        force_deterministic_pipeline=args.deterministic,
    )

    print("\n" + "=" * 70)
    print("📊 STRUCTURED TEST EVALUATION (Code-Decided Flags)")
    print("=" * 70)
    if result.flags:
        print(f"{'Test':<28} {'Value':<10} {'Unit':<10} {'Flag':<10} {'Reference Limits'}")
        print("-" * 70)
        for f in result.flags:
            flag_str = f["flag"]
            if flag_str == "HIGH":
                flag_display = "[HIGH]"
            elif flag_str == "LOW":
                flag_display = "[LOW]"
            elif flag_str == "NORMAL":
                flag_display = "[NORMAL]"
            else:
                flag_display = "[NO_RANGE]"

            ref_range = f"{f.get('ref_low', '')} - {f.get('ref_high', '')}"
            print(f"{f['test'][:27]:<28} {str(f['value']):<10} {(f.get('unit') or ''):<10} {flag_display:<12} {ref_range}")
    else:
        print("No quantitative test flags generated.")

    if result.critical_findings:
        print("\n" + "!" * 70)
        print("CRITICAL PANIC FINDINGS DETECTED:")
        for cf in result.critical_findings:
            print(f"   * {cf['test']}: {cf['value']} {cf.get('unit', '')} ({cf['breach']}) -> {cf.get('urgency_message', '')}")
        print("!" * 70)

    print("\n" + "=" * 70)
    print("PATIENT EXPLANATION")
    print("=" * 70)
    print(result.final)

    print("\n" + "-" * 70)
    print("📈 EXECUTION TELEMETRY:")
    print(f"   • Status:       {result.stop_reason}")
    print(f"   • Steps Taken:  {result.steps}")
    print(f"   • Tokens Used:  {result.tokens_used}")
    print(f"   • Trace File:   {result.trace_file}")
    print("-" * 70)


if __name__ == "__main__":
    main()
