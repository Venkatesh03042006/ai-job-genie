"""Text normalization shared by the loader, chunker, and pairing logic."""
import re

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and blank lines; keep single newlines
    (bullet chunking relies on them)."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    return text.strip()


def normalize_category(category: str) -> str:
    """Lowercase and strip punctuation so e.g. 'Data Science' and
    'data-science' are treated as the same category when pairing."""
    if not isinstance(category, str):
        return ""
    category = category.lower().strip()
    category = _NON_ALNUM_RE.sub(" ", category)
    return re.sub(r"\s+", " ", category).strip()
