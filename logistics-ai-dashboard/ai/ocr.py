"""
ai/ocr.py
OCR service abstraction. Until an OCR-capable NIM model is wired via the
registry (OCR capability), this falls back to the existing digital-text
extractor so the document pipeline keeps working for text PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from ai.router import AI
from ai.types import Capability

_log = config.get_logger(__name__)


@dataclass
class OCRResult:
    text: str
    ok: bool
    engine: str


def extract(file_bytes: bytes, filename: str) -> OCRResult:
    """Extract text from a document. Prefers a wired OCR model; falls back to
    digital-text extraction (pypdf/txt)."""
    if AI.status().get("ocr", False):
        # Placeholder for a wired OCR NIM: providers currently expose chat/embed;
        # image OCR would add a dedicated provider method. Until then, fall through.
        _log.info("OCR capability configured but image OCR path not yet wired.")
    from modules import doc_intel
    text, msg = doc_intel.extract_text(file_bytes, filename)
    if text is None:
        return OCRResult(text="", ok=False, engine=f"digital-text ({msg})")
    return OCRResult(text=text, ok=True, engine="digital-text")
