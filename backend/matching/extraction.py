"""Extract plain text from an uploaded resume file, so matching/views.py's
upload path can feed the result into the exact same
matching/services.py::get_matches() flow the paste-text path already uses -
no matching logic lives here, only text acquisition.
"""
from pathlib import Path

import docx
import pdfplumber

MIN_EXTRACTED_LENGTH = 10  # matches MatchRequestSerializer's resume_text min_length

_SUPPORTED_TYPES_MESSAGE = (
    "Unsupported file type. Please upload a .pdf, .docx, or .txt file, "
    "or paste your resume text instead."
)


class TextExtractionError(Exception):
    """User-facing extraction failure - the message is safe to show as-is
    (see matching/views.py, which returns str(exc) directly to the client)."""


def _extract_pdf(uploaded_file) -> str:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise TextExtractionError(
            "Couldn't extract text from this PDF - it may be corrupted. "
            "Try pasting the text instead."
        ) from exc
    return "\n".join(pages)


def _extract_docx(uploaded_file) -> str:
    try:
        document = docx.Document(uploaded_file)
    except Exception as exc:
        raise TextExtractionError(
            "Couldn't extract text from this DOCX file - it may be corrupted. "
            "Try pasting the text instead."
        ) from exc
    return "\n".join(p.text for p in document.paragraphs)


def _extract_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="replace")


def extract_text(uploaded_file) -> str:
    """`uploaded_file` is a Django UploadedFile (from request.FILES). Returns
    stripped extracted text, or raises TextExtractionError with a
    user-facing message - unsupported type, corrupted file, or no
    extractable text (e.g. a scanned/image-only PDF)."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(uploaded_file)
    elif suffix == ".docx":
        text = _extract_docx(uploaded_file)
    elif suffix == ".txt":
        text = _extract_txt(uploaded_file)
    else:
        raise TextExtractionError(_SUPPORTED_TYPES_MESSAGE)

    text = text.strip()
    if len(text) < MIN_EXTRACTED_LENGTH:
        raise TextExtractionError(
            "No readable text found in this file - it may be a scanned/image-only "
            "document. Try pasting the text instead."
        )
    return text
