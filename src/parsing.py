from __future__ import annotations

import io
from dataclasses import dataclass

import pdfplumber


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    text: str


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            parts: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)
            return "\n\n".join(parts).strip()
    except Exception:
        # Best-effort extraction; caller will show a parse error if empty.
        return ""


def extract_text_from_txt_bytes(txt_bytes: bytes) -> str:
    if not txt_bytes:
        return ""
    # Try utf-8 first, then fall back.
    try:
        return txt_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return txt_bytes.decode("latin-1", errors="ignore").strip()


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    return extract_text_from_txt_bytes(file_bytes)

