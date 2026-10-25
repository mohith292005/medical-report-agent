"""Production FastAPI server for the Medical Report Explanation Agent.

Provides:
  - Serves modern medical dashboard UI from /frontend
  - POST /api/analyze: Full lab report extraction, deterministic flagging, critical checking
  - POST /api/chat: Conversational medical explanation and lifestyle guidance
  - GET  /api/health: Liveness and status probe
"""

import os
import shutil
import tempfile
import logging
from typing import Optional, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from med_agent.harness.loop import run_agent

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Report Explanation Agent",
    description=(
        "Production medical AI assistant for laboratory report understanding. "
        "Deterministic numeric flags, critical panic thresholds, and plain-language explanation. "
        "Never diagnoses and never recommends medication."
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ─── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ─────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    """Structured response returned by POST /api/analyze."""
    explanation: str = Field(..., description="Plain-language summary of the lab report.")
    flags: list = Field(default_factory=list, description="Code-decided flags for test values.")
    critical_findings: list = Field(default_factory=list, description="Critical/panic alerts.")
    stop_reason: str = Field(..., description="Terminal status of the agent run.")
    steps: int = Field(..., description="Number of agent loop iterations used.")
    tokens_used: int = Field(..., description="Total LLM tokens consumed.")
    trace_file: str = Field(..., description="Path to the JSONL trace log.")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query about report or medical guidance.")
    report_text: Optional[str] = Field(default=None, description="Raw report text.")
    flags: Optional[list] = Field(default_factory=list, description="Extracted biomarker flags.")
    critical_findings: Optional[list] = Field(default_factory=list, description="Critical findings.")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Conversation history.")
    language: Optional[str] = Field(default="English", description="Target language.")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant response.")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up prompt suggestions.")


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health", tags=["ops"])
@app.get("/health", tags=["ops"], include_in_schema=False)
def health():
    """Simple liveness probe."""
    return {"status": "ok", "app": "medical-report-agent", "version": "2.0.0"}


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Analyze a medical lab report",
)
@app.post("/analyze", response_model=AnalyzeResponse, include_in_schema=False)
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("English"),
    previous_text: Optional[str] = Form(None),
):
    """
    Upload a report file (PDF, TXT, PNG, JPG) via 'file' or paste raw text
    via 'text'. Returns structured flags, critical findings, and explanation.
    """
    if not file and not text:
        raise HTTPException(
            status_code=422,
            detail="Please provide either a 'file' upload or 'text' content.",
        )

    file_path = None
    report_text = text or ""

    try:
        if file:
            suffix = os.path.splitext(file.filename or "report.txt")[1]
            tmp_dir = tempfile.mkdtemp(prefix="medagent_")
            file_path = os.path.join(tmp_dir, f"upload{suffix}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info("Uploaded file: %s (%d bytes)", file.filename, len(content))

        result = run_agent(
            report_text=report_text,
            file_path=file_path,
            language=language,
            previous_report_text=previous_text,
        )

        return AnalyzeResponse(
            explanation=result.final,
            flags=result.flags,
            critical_findings=result.critical_findings,
            stop_reason=result.stop_reason,
            steps=result.steps,
            tokens_used=result.tokens_used,
            trace_file=result.trace_file,
        )

    except Exception as e:
        logger.error("Analysis error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if file_path and os.path.exists(file_path):
            shutil.rmtree(os.path.dirname(file_path), ignore_errors=True)


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Chat with the Medical Report Explanation Assistant",
)
def chat_with_assistant(req: ChatRequest):
    """
    Answers questions about the uploaded report, explains biomarker findings,
    offers evidence-based lifestyle/diet suggestions, and suggests doctor questions.
    Never diagnoses and never prescribes medication.
    """
    msg_lower = req.message.lower().strip()
    flags = req.flags or []
    critical = req.critical_findings or []

    # Check for OpenRouter API Key
    from med_agent.harness.config import AgentConfig
    cfg = AgentConfig()
    api_key_valid = (
        bool(cfg.api_key)
        and "your_openrouter_api_key_here" not in cfg.api_key
        and len(cfg.api_key.strip()) > 10
    )

    if api_key_valid:
        try:
            from med_agent.harness.llm import OpenRouterLLM
            llm = OpenRouterLLM(config=cfg)

            system_prompt = (
                "You are MedExplain Assistant, a warm, clear, and empathetic medical AI companion.\n"
                "The user is asking questions about their laboratory blood test report.\n\n"
                "ACTIVE LAB REPORT DATA:\n"
                f"- Flags: {flags}\n"
                f"- Critical Alerts: {critical}\n"
                f"- Language: {req.language}\n\n"
                "MANDATORY SAFETY GUARDRAILS:\n"
                "1. NEVER DIAGNOSE. Do not say 'You have diabetes' or 'You suffer from anemia'. Use phrasing like 'This level is higher than typical reference ranges' or 'This pattern is often associated with...'\n"
                "2. NEVER PRESCRIBE. Do not recommend pharmaceutical drugs, medications, or specific milligram dosages.\n"
                "3. PROVIDE PRACTICAL LIFESTYLE/DIET SUGGESTIONS: When asked about suggestions or lifestyle, offer wholesome, safe habits (hydration, dietary sources like fiber, leafy greens, lean protein, balanced meals, sleep, stress reduction).\n"
                "4. EMPOWER THE PATIENT: Give them intelligent, clear questions to ask their doctor.\n"
                "5. CRITICAL VALUES: If critical findings exist, prioritize urging them to contact their healthcare provider promptly.\n"
                "6. Format response in clean, beautiful Markdown with bullet points where appropriate.\n"
                "7. Always conclude with a brief reminder that this is educational guidance, not a medical prescription."
            )

            messages = [{"role": "system", "content": system_prompt}]
            for h in (req.history or [])[-6:]:
                messages.append({"role": h.role, "content": h.content})
            messages.append({"role": "user", "content": req.message})

            llm_resp = llm.chat(messages=messages, temperature=0.3)
            reply_text = llm_resp.message.get("content", "")
            if reply_text:
                suggestions = [
                    "What's going on with my report?",
                    "What dietary changes might help my levels?",
                    "What questions should I ask my doctor?",
                ]
                return ChatResponse(reply=reply_text, suggestions=suggestions)
        except Exception as e:
            logger.warning("LLM chat call failed, falling back to deterministic: %s", e)

    # ─── Deterministic / Offline Fallback Engine ──────────────────────────────
    from med_agent.tools.glossary import explain_term
    from med_agent.tools.questions import doctor_questions, FlaggedItemInput

    abnormal_items = [f for f in flags if isinstance(f, dict) and f.get("flag") in ("HIGH", "LOW")]
    critical_items = [c for c in critical if isinstance(c, dict)]

    # Intent 1: Critical or Panic inquiry
    if any(k in msg_lower for k in ("critical", "emergency", "danger", "panic", "urgent")):
        if critical_items:
            items_str = "\n".join(
                f"- **{c.get('test', '').title()}**: {c.get('value')} {c.get('unit', '')} ({c.get('direction', '')} threshold: {c.get('threshold')})\n  *{c.get('message', '')}*"
                for c in critical_items
            )
            reply = (
                "### ⚠️ Critical Lab Finding Notice\n\n"
                "Your lab report contains one or more values that crossed immediate medical attention thresholds:\n\n"
                f"{items_str}\n\n"
                "**What you should do right now:**\n"
                "- Please contact your healthcare provider, clinic, or urgent care helpline without delay to discuss these findings.\n"
                "- Bring a physical or digital copy of this report with you.\n\n"
                "*Disclaimer: This analysis is for educational purposes and is not a clinical diagnosis or treatment.*"
            )
        else:
            reply = (
                "### ✅ No Critical Panic Values Detected\n\n"
                "Good news! None of the tests evaluated in this report crossed critical or panic-level thresholds.\n\n"
                f"We reviewed **{len(flags)} test parameters** in total. "
                + (f"There are {len(abnormal_items)} values outside standard baseline reference ranges, but none meet acute emergency criteria."
                   if abnormal_items else "All analyzed values are sitting comfortably within expected normal reference intervals.")
                + "\n\nFeel free to ask for a breakdown of any specific biomarker or lifestyle guidance!"
            )
        suggestions = ["What is going on overall?", "What should I ask my doctor?", "Show lifestyle tips"]
        return ChatResponse(reply=reply, suggestions=suggestions)

    # Intent 2: "What's going on" / Overview / Summary
    if any(k in msg_lower for k in ("what's going on", "whats going on", "overview", "summary", "explain my report", "what does this mean", "how is my health")):
        if not flags:
            reply = (
                "### 👋 Welcome to MedExplain!\n\n"
                "I'm ready to explain your lab report in plain, easy-to-understand language.\n\n"
                "**To get started:**\n"
                "1. Drag and drop your report PDF, photo, or paste your lab text in the analyzer section above.\n"
                "2. Click **Analyze Report**.\n"
                "3. Once analyzed, I will break down every biomarker, highlight abnormal values, and provide tailored questions for your doctor!"
            )
            suggestions = ["Load sample report", "How does this work?", "What tests can you read?"]
            return ChatResponse(reply=reply, suggestions=suggestions)

        abn_text = ""
        if abnormal_items:
            abn_list = []
            for item in abnormal_items:
                t_name = item.get("test", "").title()
                val = item.get("value", "")
                unit = item.get("unit", "")
                fl = item.get("flag", "")
                r_low = item.get("reference_low", "")
                r_high = item.get("reference_high", "")
                ref_str = f"(Standard Ref: {r_low} - {r_high} {unit})" if r_low is not None else ""
                abn_list.append(f"- **{t_name}**: `{val} {unit}` — **{fl}** {ref_str}")
            abn_text = "\n".join(abn_list)
        else:
            abn_text = "- All evaluated test numbers are currently within standard reference ranges!"

        crit_banner = ""
        if critical_items:
            crit_banner = (
                "> 🚨 **Important Notice**: One or more results (such as "
                + ", ".join(c.get("test", "").title() for c in critical_items)
                + ") reached critical thresholds. We recommend contacting your physician promptly.\n\n"
            )

        reply = (
            f"### 📋 Report Summary Breakdown\n\n"
            f"{crit_banner}"
            f"Here is an overview of what was evaluated in your report:\n\n"
            f"**Key Findings:**\n"
            f"- **Total Tests Evaluated**: {len(flags)}\n"
            f"- **Abnormal Biomarkers ({len(abnormal_items)})**:\n{abn_text}\n\n"
            "**What this means:**\n"
            "Lab values can fluctuate due to hydration, diet, exercise, stress, or temporary physiological states. "
            "Biomarkers outside reference ranges are common indicator lights that guide your physician's clinical assessment.\n\n"
            "*Reminder: Only your licensed physician can interpret these numbers in the context of your complete medical history.*"
        )
        suggestions = ["Give me lifestyle & dietary suggestions", "What questions should I ask my doctor?", "Check critical values"]
        return ChatResponse(reply=reply, suggestions=suggestions)

    # Intent 3: Suggestions, Diet, Lifestyle, Food, Exercise
    if any(k in msg_lower for k in ("suggest", "suggestion", "diet", "food", "eat", "lifestyle", "habit", "exercise", "what should i do", "what can i do")):
        lifestyle_points = []

        if any("potassium" in str(f.get("test", "")).lower() and f.get("flag") == "HIGH" for f in flags):
            lifestyle_points.append(
                "• **Potassium Balance**: Since potassium is elevated, temporarily moderate high-potassium foods (such as bananas, avocados, potatoes, spinach, tomatoes, and salt substitutes containing potassium chloride) until you speak with your doctor."
            )
        elif any("potassium" in str(f.get("test", "")).lower() and f.get("flag") == "LOW" for f in flags):
            lifestyle_points.append(
                "• **Potassium Support**: Consider natural dietary sources like sweet potatoes, coconut water, bananas, and lentils, and discuss electrolyte balance with your doctor."
            )

        if any(any(k in str(f.get("test", "")).lower() for k in ("hemoglobin", "rbc", "ferritin", "hematocrit")) and f.get("flag") == "LOW" for f in flags):
            lifestyle_points.append(
                "• **Iron & Oxygen Support**: To support red blood cell vitality, incorporate iron-rich foods (lentils, chickpeas, spinach, pumpkin seeds, lean meats) paired with vitamin C (citrus, bell peppers) to boost absorption. Avoid drinking coffee or tea during iron-rich meals."
            )

        if any(any(k in str(f.get("test", "")).lower() for k in ("glucose", "hba1c", "blood sugar")) and f.get("flag") == "HIGH" for f in flags):
            lifestyle_points.append(
                "• **Metabolic Health & Glucose Balance**: Focus on high-fiber whole foods, steady hydration, lean proteins, and complex carbohydrates. Light 10-15 minute walks after meals can naturally support cellular glucose clearance."
            )

        if any(any(k in str(f.get("test", "")).lower() for k in ("cholesterol", "ldl", "triglyceride")) and f.get("flag") == "HIGH" for f in flags):
            lifestyle_points.append(
                "• **Cardiovascular Wellness**: Emphasize heart-healthy omega-3s and unsaturated fats (olive oil, flaxseeds, walnuts), soluble fiber (oats, legumes), and aim for 150 minutes of moderate aerobic activity weekly."
            )

        if any("creatinine" in str(f.get("test", "")).lower() and f.get("flag") == "HIGH" for f in flags):
            lifestyle_points.append(
                "• **Kidney Hydration Support**: Maintain consistent, clean hydration throughout the day unless your doctor has placed you on fluid restriction. Limit excessive non-steroidal anti-inflammatory drugs (NSAIDs) like ibuprofen without clinical guidance."
            )

        if not lifestyle_points:
            lifestyle_points = [
                "• **Consistent Hydration**: Aim for 2 to 2.5 liters of water daily to maintain electrolyte equilibrium and cellular function.",
                "• **Balanced Nutrition**: Prioritize unprocessed whole foods, colorful vegetables, adequate dietary fiber, and quality protein sources.",
                "• **Daily Movement**: Engage in regular light-to-moderate physical activity (brisk walking, swimming, cycling) suited to your comfort level.",
                "• **Restorative Sleep**: Prioritize 7-8 hours of quality sleep to support natural endocrine and immune regulation.",
            ]

        pts_str = "\n".join(lifestyle_points)
        reply = (
            "### 🥗 Evidence-Based Lifestyle & Dietary Suggestions\n\n"
            f"Based on the biomarkers analyzed in your lab report, here are safe, non-clinical wellness considerations:\n\n"
            f"{pts_str}\n\n"
            "**Next Step:**\n"
            "Share these lifestyle adjustments with your doctor or registered dietitian so they can tailor them to your unique personal health profile.\n\n"
            "*Note: These tips are educational wellness suggestions and are never a replacement for prescribed medical treatment.*"
        )
        suggestions = ["What questions should I ask my doctor?", "What's going on with my report?", "Explain my abnormal values"]
        return ChatResponse(reply=reply, suggestions=suggestions)

    # Intent 4: Doctor questions
    if any(k in msg_lower for k in ("doctor", "physician", "questions to ask", "what to ask", "appointment", "clinic")):
        flag_objs = [
            FlaggedItemInput(
                test=f.get("test", "test"),
                flag=f.get("flag", "NORMAL"),
                value=f.get("value"),
                unit=f.get("unit"),
            )
            for f in flags if isinstance(f, dict)
        ]
        q_result = doctor_questions(flags=flag_objs)
        q_list = q_result.get("questions", [])
        q_formatted = "\n".join(f"{i+1}. **{q}**" for i, q in enumerate(q_list))

        reply = (
            "### 🩺 Questions Prepared for Your Doctor's Visit\n\n"
            "Here are clear, collaborative questions tailored to your lab results to take to your next appointment:\n\n"
            f"{q_formatted}\n\n"
            "**Patient Tip:** You can print or save these questions to ensure you get thorough answers during your consultation."
        )
        suggestions = ["What's going on overall?", "Show lifestyle suggestions", "Explain abnormal biomarkers"]
        return ChatResponse(reply=reply, suggestions=suggestions)

    # Intent 5: Specific test query
    matched_test = None
    for f in flags:
        t_name = str(f.get("test", "")).lower()
        if t_name and t_name in msg_lower:
            matched_test = f
            break

    if matched_test:
        t_name = matched_test.get("test", "").title()
        val = matched_test.get("value", "")
        unit = matched_test.get("unit", "")
        fl = matched_test.get("flag", "")
        r_low = matched_test.get("reference_low", "N/A")
        r_high = matched_test.get("reference_high", "N/A")

        glossary_info = explain_term(test_name=t_name.lower())
        desc = glossary_info.get("what_it_measures", "A standard clinical biomarker evaluated in blood tests.")
        meaning_dict = glossary_info.get("general_meaning", {})
        flag_key = fl.lower() if fl.lower() in meaning_dict else "normal"
        meaning = meaning_dict.get(flag_key, f"This value was evaluated as {fl} relative to standard laboratory thresholds.")

        reply = (
            f"### 🔬 Understanding {t_name}\n\n"
            f"- **Your Result**: `{val} {unit}` (**{fl}**)\n"
            f"- **Standard Reference Range**: `{r_low} – {r_high} {unit}`\n\n"
            f"**What it measures:**\n{desc}\n\n"
            f"**What this {fl} level indicates:**\n{meaning}\n\n"
            f"**Recommended discussion point for your physician:**\n"
            f"\"Could recent diet, hydration, physical activity, or temporary conditions have influenced my {t_name} reading?\""
        )
        suggestions = ["What other values are abnormal?", "Show lifestyle tips", "Questions for my doctor"]
        return ChatResponse(reply=reply, suggestions=suggestions)

    # Default fallback
    reply = (
        "### 💬 MedExplain Companion\n\n"
        "I'm here to help you understand your laboratory results clearly and safely.\n\n"
        "You can ask me:\n"
        "- **\"What's going on with my report?\"** — for an overview of your findings.\n"
        "- **\"What dietary or lifestyle suggestions do you have?\"** — for evidence-based wellness tips.\n"
        "- **\"What questions should I ask my doctor?\"** — for customized physician questions.\n"
        "- Or ask about any specific test, like **\"Why is my potassium high?\"**\n\n"
        "*Remember: I do not provide medical diagnoses or prescribe medications. Always consult your healthcare provider.*"
    )
    suggestions = ["What's going on?", "Lifestyle & diet suggestions", "Doctor questions", "Check critical alerts"]
    return ChatResponse(reply=reply, suggestions=suggestions)


# ─── Serve Frontend ─────────────────────────────────────────────────────────

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Mount frontend directory for styles, scripts, and images
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the clean, modern medical frontend."""
    frontend_index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index, media_type="text/html")
    return JSONResponse({"status": "running", "message": "MedExplain API running."})


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  MedExplain AI — Production Backend Server")
    print("  Frontend UI:  http://localhost:8000")
    print("  Swagger Docs: http://localhost:8000/api/docs")
    print("  Health Probe: http://localhost:8000/api/health")
    print("=" * 60 + "\n")
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
