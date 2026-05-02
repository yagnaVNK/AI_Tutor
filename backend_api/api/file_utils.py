"""Lightweight text extraction helpers for uploaded files.

We extract on upload and store the result on the row so the LLM can be
fed file content directly without re-reading the binary on every chat.
For large/long-term scaling, swap this for a chunk + embed pipeline
backed by pgvector.
"""
from __future__ import annotations

import io
import logging
from typing import IO

logger = logging.getLogger(__name__)

MAX_EXTRACTED_CHARS = 200_000


def extract_text(file_obj: IO[bytes], filename: str, mime_type: str) -> str:
    name_lower = (filename or "").lower()
    try:
        if name_lower.endswith(".pdf") or mime_type == "application/pdf":
            return _extract_pdf(file_obj)
        if name_lower.endswith(".docx") or mime_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            return _extract_docx(file_obj)
        if name_lower.endswith((".txt", ".md", ".csv", ".json", ".log")) or (
            mime_type and mime_type.startswith("text/")
        ):
            return _extract_text(file_obj)
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort
        logger.warning("Failed to extract text from %s: %s", filename, exc)
    return ""


def _extract_pdf(file_obj: IO[bytes]) -> str:
    from pypdf import PdfReader

    file_obj.seek(0)
    reader = PdfReader(file_obj)
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
        if sum(len(p) for p in parts) > MAX_EXTRACTED_CHARS:
            break
    return "\n".join(parts)[:MAX_EXTRACTED_CHARS]


def _extract_docx(file_obj: IO[bytes]) -> str:
    from docx import Document

    file_obj.seek(0)
    doc = Document(io.BytesIO(file_obj.read()))
    return "\n".join(p.text for p in doc.paragraphs)[:MAX_EXTRACTED_CHARS]


def _extract_text(file_obj: IO[bytes]) -> str:
    file_obj.seek(0)
    raw = file_obj.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)[:MAX_EXTRACTED_CHARS]
        except UnicodeDecodeError:
            continue
    return ""
