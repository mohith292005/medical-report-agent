# ─── Build Stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for PyMuPDF and pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Runtime Stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY med_agent/ ./med_agent/
COPY data/ ./data/
COPY static/ ./static/
COPY server.py .
COPY main.py .
COPY eval.py .
COPY samples/ ./samples/

# Create trace directory
RUN mkdir -p traces uploads

# Environment defaults (override at runtime)
ENV OPENROUTER_API_KEY=""
ENV OPENROUTER_MODEL="openai/gpt-4o-mini"
ENV MAX_STEPS="15"
ENV MAX_TOTAL_TOKENS="32000"
ENV MAX_VERIFY_RETRIES="3"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run the production server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
