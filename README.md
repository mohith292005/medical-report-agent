# Medical Report Explanation Agent

A student-friendly, production-hardened AI agent that reads laboratory reports
(PDF, image, or pasted text), extracts test values, flags abnormal ones with
**deterministic code**, checks for panic-level critical values, and explains
everything in plain language. It **never diagnoses** and **never recommends
medication**.

> **Core Principle**: Code decides the numbers, the LLM only explains them.
> The LLM never decides whether a value is HIGH or LOW — that comes exclusively
> from the `flag_abnormal` tool.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [The Agent Loop](#the-agent-loop)
- [Project Structure](#project-structure)
- [The 9 Tools](#the-9-tools)
- [Safety Controls](#safety-controls)
- [API Server](#api-server)
- [Running Tests](#running-tests)
- [Evaluation Script](#evaluation-script)
- [Configuration](#configuration)
- [Adding a New Tool](#adding-a-new-tool)
- [Technology Stack](#technology-stack)

---

## Quick Start

```bash
# 1. Clone and enter the project
cd medical-report-agent

# 2. Create a virtual environment and install dependencies
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# 3. (Optional) Add your OpenRouter API key for LLM explanations
cp .env.example .env
# Edit .env → set OPENROUTER_API_KEY=your_key

# 4. Run the CLI on a sample report (works offline in deterministic mode)
python main.py samples/report_anemia_cbc.txt --deterministic

# 5. Run the eval suite to verify everything works
python eval.py

# 6. Run tests
python -m pytest tests/ -v

# 7. Start the API server
python server.py
# → http://localhost:8000/docs  (interactive Swagger UI)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (CLI / API)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐  │
│  │   PLAN    │───→│   ACT    │───→│ OBSERVE │───→│  VERIFY  │  │
│  │  (LLM)   │    │ (Tools)  │    │ (State) │    │ (Code +  │  │
│  │          │    │          │    │         │    │  Critic) │  │
│  └───────────┘    └──────────┘    └─────────┘    └──────────┘  │
│       ↑                                              │         │
│       └──────────────── RETRY ◄──────────────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Safety Controls (always on)                 │   │
│  │  • Budget (max steps, max tokens)                       │   │
│  │  • Loop detection (max 2 retries per failing call)      │   │
│  │  • Mandatory tool enforcement (flag + critical)         │   │
│  │  • Deterministic verifier + LLM critic                  │   │
│  │  • Safe fail message after MAX_VERIFY_RETRIES           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Trace Logger                          │   │
│  │  Every step → traces/<timestamp>-<run_id>.jsonl         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Agent Loop

The loop in [`med_agent/harness/loop.py`](med_agent/harness/loop.py) follows
the cycle **PLAN → ACT → OBSERVE → VERIFY → DECIDE**:

1. **PLAN**: The LLM receives the report text, system prompt, and tool schemas.
2. **ACT**: The LLM calls tools (parse, extract, flag, etc.) via OpenAI-style
   tool calling.
3. **OBSERVE**: Tool outputs are stored in `RunState` and appended to the
   conversation history.
4. **VERIFY**: When the LLM produces a draft explanation:
   - **Safety Gate**: Rejects if `flag_abnormal` and `critical_check` haven't
     been called yet.
   - **Deterministic Verifier**: Checks the draft mentions every flagged test,
     has the disclaimer, has the urgent banner (if critical values exist), and
     contains no diagnosis/prescription language.
   - **LLM Critic**: A second LLM pass reviews the draft for safety violations.
5. **DECIDE**: If both verifiers pass, the explanation is finalized (optionally
   translated) and returned.

If verification fails, the loop feeds specific rejection reasons back to the
LLM for self-correction. After `MAX_VERIFY_RETRIES` (default 3), the agent
terminates safely with:
> "I could not produce a verified explanation. Please check the report with
> your lab or doctor."

---

## Project Structure

```
medical-report-agent/
├── main.py                  # CLI entry point (delegates to backend)
├── server.py                # Root FastAPI entry point (delegates to backend.server)
├── eval.py                  # Evaluation script (runs all 5 sample reports)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (OpenRouter key, model)
├── .env.example             # Template for environment variables
├── pytest.ini               # Pytest configuration
├── Dockerfile               # Production multi-stage Dockerfile
├── .dockerignore            # Docker build ignore rules
│
├── frontend/                # Modern Medical Web Application
│   ├── index.html           # Single-page 3D medical dashboard & chatbot UI
│   ├── css/
│   │   └── styles.css       # Clean glassmorphism styling & animations
│   ├── js/
│   │   └── app.js           # Three.js 3D helix, file drag-drop, chat client
│   └── images/
│       └── robot_assistant.jpg # 3D Medical Robot Mascot
│
├── backend/                 # Production Backend Server & Agent Core
│   ├── __init__.py
│   ├── server.py            # FastAPI app (/api/analyze, /api/chat, /api/health)
│   ├── data/                # Clinical JSON datasets
│   │   ├── glossary.json         # ~27 lab test definitions
│   │   ├── reference_ranges.json # Fallback adult reference ranges
│   │   └── critical_values.json  # Panic threshold definitions
│   ├── samples/             # Sample reports (Anemia, Potassium, Lipid, Normal)
│   └── med_agent/           # Agent loop & tools package
│       ├── harness/         # Config, state, registry, tracer, verifier, loop
│       └── tools/           # 9 registered deterministic & hybrid tools
│
├── tests/                   # Complete test suite (30 tests, all passing)
│   ├── test_harness.py      # Config, state, registry, tracer, LLM mock
│   ├── test_flag.py         # Deterministic flagging logic
│   ├── test_phase2.py       # Parse, extract, unit normalization
│   ├── test_phase3.py       # Critical checks, glossary, trend, questions
│   ├── test_loop_rejection.py # Agent safety enforcement loop
│   └── test_server.py       # FastAPI endpoints, health, chat, and upload
│
└── traces/                  # Runtime trace logs (auto-created JSONL)
```

---

## The 9 Tools

| # | Tool | Type | What It Does |
|---|------|------|-------------|
| 1 | `parse_report` | Deterministic | Reads PDF (PyMuPDF), image (pytesseract), or TXT files into raw text |
| 2 | `extract_values` | Hybrid | Regex extracts test name, value, unit, reference range from text. Falls back to LLM JSON-mode for complex formats |
| 3 | `normalize_units` | Deterministic | Converts mmol/L → mg/dL, µmol/L → mg/dL, g/L → g/dL with medical conversion factors |
| 4 | `flag_abnormal` | **Deterministic** | Compares each value against report ranges or fallback ranges. Returns HIGH, LOW, NORMAL, or NO_RANGE |
| 5 | `critical_check` | **Deterministic** | Checks values against panic thresholds (e.g., potassium > 6.5). Returns urgency messages |
| 6 | `explain_term` | Deterministic | Looks up test name in `glossary.json` for trusted plain-language definitions |
| 7 | `compare_trend` | Deterministic | Compares current vs. previous report values and describes directional changes |
| 8 | `doctor_questions` | Deterministic | Generates template-based questions for the patient to ask their physician |
| 9 | `translate_summary` | LLM | Translates the final explanation to Hindi, Kannada, Telugu, or Tamil while preserving numbers and units |

---

## Safety Controls

### 1. Mandatory Tool Enforcement
The model **cannot** produce a final answer without having successfully called
both `flag_abnormal` and `critical_check`. If it tries to skip them, the loop
rejects the draft and forces the model to call them.

### 2. Deterministic Verifier
Before any draft reaches the user, [`verify.py`](med_agent/harness/verify.py)
checks:
- Every flagged test name appears in the explanation
- The disclaimer is present
- The urgent banner is present (if critical values exist)
- No diagnosis language ("you have", "you are diagnosed with")
- No treatment/dosage language ("take medication", "mg")

### 3. LLM Critic
A second LLM pass (using the Critic prompt) independently reviews the draft
for safety violations, tone issues, missing banners, or forbidden language.

### 4. Budget Controls
- **MAX_STEPS** (default 15): Caps total loop iterations
- **MAX_TOTAL_TOKENS** (default 32,000): Caps total LLM token usage
- **MAX_VERIFY_RETRIES** (default 3): Caps verification retry attempts

### 5. Loop Detection
If the same tool is called with identical arguments and fails twice, the loop
stops retrying that specific call.

### 6. Safe Fail
If all retries are exhausted, the agent returns:
> "I could not produce a verified explanation. Please check the report with
> your lab or doctor."

It never returns an unverified explanation.

---

## Web UI & Production API Server

### 1. Launching the Full-Stack Application

Start the production FastAPI server:

```bash
python server.py
# → Web UI:      http://localhost:8000
# → Swagger UI:  http://localhost:8000/docs
# → Health:      http://localhost:8000/api/health
```

### 2. Modern Web Dashboard (`/`)

The application serves a single-page medical dashboard styled with modern glassmorphism, animated background orbs, and color-coded status badges:
- **Dual Input Modes**: Drag-and-drop file upload (`.txt`, `.pdf`, `.png`, `.jpg`, `.jpeg`) or instant text paste.
- **Quick-Load Presets**: 1-click sample loaders (Normal CBC, Anemia CBC, Critical Potassium, Lipid Profile).
- **Target Language**: English, Hindi, Kannada, Telugu, or Tamil.
- **Visual Intelligence**:
  - Panic-level critical finding alert banner with pulsing indicator.
  - Animated stat counters for tests parsed, abnormal flags, and critical alerts.
  - Color-coded flags breakdown (HIGH, LOW, NORMAL).
  - Rich Markdown explanation rendering with copy & print capabilities.
- **Trace Transparency**: Displays run ID, step count, tokens consumed, and local trace file path.

### 3. REST API Endpoints

#### `POST /api/analyze` (or `POST /analyze`)
Accepts `multipart/form-data` with either `file` or `text`, and optional `language` and `deterministic`:

```bash
# Upload a PDF / image / TXT report
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@samples/report_critical_potassium.txt" \
  -F "language=English"

# Direct text analysis
curl -X POST http://localhost:8000/api/analyze \
  -F "text=Potassium: 6.8 mEq/L (ref: 3.5-5.0)" \
  -F "language=English"
```

Response JSON:
```json
{
  "explanation": "### Overview\nThis report shows ...",
  "flags": [
    {
      "test": "potassium",
      "value": 6.8,
      "unit": "mEq/L",
      "flag": "HIGH",
      "reference_low": 3.5,
      "reference_high": 5.0
    }
  ],
  "critical_findings": [
    {
      "test": "potassium",
      "value": 6.8,
      "unit": "mEq/L",
      "threshold": 6.5,
      "direction": "HIGH",
      "message": "CRITICAL: potassium is severely elevated..."
    }
  ],
  "stop_reason": "completed",
  "steps": 4,
  "tokens_used": 1240,
  "trace_file": "traces/20261024_021355-fed88a01.jsonl"
}
```

#### `GET /api/health`
Liveness probe returning server status:
```bash
curl http://localhost:8000/api/health
# → {"status": "ok", "app": "medical-report-agent", "version": "1.0.0"}
```

---

## Docker Deployment

Build and run the containerized application with multi-stage build and OCR support:

```bash
# 1. Build the Docker image
docker build -t med-report-agent .

# 2. Run the container on port 8000
docker run -p 8000:8000 --env-file .env med-report-agent

# → Visit http://localhost:8000
```

---

## Running Tests

The test suite contains 27 comprehensive unit and integration tests covering the agent harness, tool execution, safety rejection loops, and FastAPI endpoints:

```bash
# Run all 27 tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_server.py -v
python -m pytest tests/test_loop_rejection.py -v
python -m pytest tests/test_flag.py -v
```

All tests run offline with zero external API dependencies.

---

## Evaluation Script

```bash
# Run all 5 sample reports (deterministic, no API key)
python eval.py

# Run verbose (prints full explanations)
python eval.py --verbose

# Run one specific sample
python eval.py --sample report_critical_potassium.txt
```

The eval script validates:
- Values are extracted from every sample
- Expected flag types (HIGH, LOW, NORMAL) are present
- Critical panic values trigger alerts when expected
- The explanation mentions every flagged test
- The safety disclaimer is always present
- No forbidden diagnosis/prescription language appears

---

## Configuration

Copy `.env.example` to `.env` and set your values:

```env
# Required for LLM explanations (get one at https://openrouter.ai/keys)
OPENROUTER_API_KEY=your_key_here

# Model (must support tool calling)
OPENROUTER_MODEL=openai/gpt-4o-mini

# Safety budgets
MAX_STEPS=15
MAX_TOTAL_TOKENS=32000
MAX_VERIFY_RETRIES=3
```

Without an API key, the agent runs in **deterministic mode** — it still
extracts, flags, and explains using code-generated text, but without LLM
natural-language polish.

---

## Adding a New Tool

1. Create a new file in `med_agent/tools/`, e.g., `my_tool.py`
2. Define a Pydantic input model and the tool function:

```python
from pydantic import BaseModel
from med_agent.harness.registry import default_registry

class MyToolInput(BaseModel):
    """Input schema — becomes the tool's JSON schema for the LLM."""
    param: str

@default_registry.tool(
    name="my_tool",
    description="What this tool does (shown to the LLM).",
    input_model=MyToolInput,
)
def my_tool(param: str) -> dict:
    """Deterministic logic here."""
    return {"result": f"Processed: {param}"}
```

3. Import it in `med_agent/tools/__init__.py`:
```python
from med_agent.tools.my_tool import my_tool
```

4. The tool is now automatically available to the LLM via `registry.schemas()`.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| LLM Access | OpenAI SDK → OpenRouter |
| PDF Parsing | PyMuPDF |
| OCR (images) | pytesseract + pdf2image |
| Testing | pytest |
| Agent Framework | **None** — custom loop written from scratch |

---

## License

This project is for educational purposes. The reference ranges, glossary
definitions, and critical value thresholds are illustrative examples and
**must not** be used for real clinical decisions.

---

*Built as a learning exercise in agent harness design. The code is heavily
commented to help students understand every component.*
