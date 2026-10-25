"""Tool for parsing lab reports from PDF, images, or text files.

Extraction Strategy:
1. Plain Text (.txt): Directly reads file with utf-8 encoding.
2. PDF Documents (.pdf): First attempts fast, precise native text extraction
   via PyMuPDF (fitz). If extracted character count is under ~50 (typical for scanned
   documents), it falls back to OCR via pdf2image and pytesseract.
3. Images (.png, .jpg, .jpeg, etc.): Runs OCR via pytesseract and Pillow.
4. Graceful Error Handling: If OCR binaries (Tesseract or Poppler) are missing,
   returns clear, actionable setup guidance instead of crashing.
"""

import os
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    pytesseract = None
    Image = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from med_agent.harness.registry import default_registry


class ParseReportInput(BaseModel):
    """Input payload for parse_report tool."""
    path: str = Field(description="Local filesystem path to PDF, image, or text report file")


def _ocr_image_file(image_path: str) -> Tuple[str, Optional[str]]:
    """Helper to run pytesseract on an image file path."""
    if not Image or not pytesseract:
        return "", "Missing Python dependencies 'Pillow' or 'pytesseract'. Please install them."
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip(), None
    except Exception as e:
        err_msg = str(e)
        if "tesseract is not installed" in err_msg.lower() or "not found" in err_msg.lower():
            return "", (
                "Tesseract OCR is not installed or not in your system PATH. "
                "On Windows, download installer from: https://github.com/UB-Mannheim/tesseract/wiki "
                "and ensure tesseract.exe is in your PATH."
            )
        return "", f"OCR extraction failed: {err_msg}"


def _ocr_pdf_file(pdf_path: str) -> Tuple[str, Optional[str]]:
    """Helper to convert PDF pages to images and run pytesseract."""
    if not convert_from_path:
        return "", "Missing dependency 'pdf2image'. Please install with pip install pdf2image."
    try:
        pages = convert_from_path(pdf_path, first_page=1, last_page=5)
        texts = []
        for p in pages:
            txt = pytesseract.image_to_string(p)
            texts.append(txt)
        return "\n\n".join(texts).strip(), None
    except Exception as e:
        err_msg = str(e)
        if "poppler" in err_msg.lower():
            return "", (
                "Poppler is required for PDF OCR conversion. "
                "On Windows, install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/ "
                "and add the bin/ directory to your PATH."
            )
        if "tesseract" in err_msg.lower():
            return "", "Tesseract OCR engine is not installed or not in PATH."
        return "", f"PDF OCR failed: {err_msg}"


@default_registry.tool(
    name="parse_report",
    description="Extracts text content from a report file (PDF, PNG, JPG, or TXT). Falls back to OCR for scanned documents.",
    input_model=ParseReportInput,
)
def parse_report(path: str) -> Dict[str, Any]:
    """Extracts text from the provided file path."""
    if not os.path.exists(path):
        return {
            "error": f"File does not exist: {path}",
            "text": "",
            "method": "failed",
        }

    ext = os.path.splitext(path)[1].lower()

    # 1. Plain text files
    if ext in (".txt", ".text", ".log", ".csv"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {
                "text": content,
                "method": "plain_text",
                "char_count": len(content),
                "page_count": 1,
            }
        except Exception as e:
            return {"error": f"Error reading text file: {e}", "text": "", "method": "failed"}

    # 2. PDF files
    if ext == ".pdf":
        if not fitz:
            return {"error": "PyMuPDF (fitz) is not installed.", "text": "", "method": "failed"}
        try:
            doc = fitz.open(path)
            extracted_pages = []
            for page in doc:
                extracted_pages.append(page.get_text())
            doc.close()

            combined_text = "\n\n".join(extracted_pages).strip()
            
            # If extracted text is under ~50 characters, assume scanned/raster PDF and try OCR
            if len(combined_text) >= 50:
                return {
                    "text": combined_text,
                    "method": "text",
                    "char_count": len(combined_text),
                    "page_count": len(extracted_pages),
                }

            # Fallback to OCR
            ocr_text, ocr_err = _ocr_pdf_file(path)
            if ocr_err:
                return {
                    "error": f"Scanned PDF contains sparse text ({len(combined_text)} chars). OCR attempted but failed: {ocr_err}",
                    "text": combined_text,
                    "method": "text_fallback_sparse",
                }
            return {
                "text": ocr_text,
                "method": "ocr",
                "char_count": len(ocr_text),
                "page_count": len(extracted_pages),
            }
        except Exception as e:
            return {"error": f"PDF reading failed: {e}", "text": "", "method": "failed"}

    # 3. Image files
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        ocr_text, ocr_err = _ocr_image_file(path)
        if ocr_err:
            return {"error": ocr_err, "text": "", "method": "failed"}
        return {
            "text": ocr_text,
            "method": "ocr",
            "char_count": len(ocr_text),
            "page_count": 1,
        }

    return {
        "error": f"Unsupported file extension '{ext}'. Supported: .pdf, .txt, .png, .jpg, .jpeg",
        "text": "",
        "method": "failed",
    }
