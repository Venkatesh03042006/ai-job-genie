"""Category definitions shared by the postings-cache builder and the loader.

Scope: the 14 (of the original 28) resume categories that the LinkedIn Job
Postings dataset (`data/raw/postings.csv`) has genuine (100+) real-posting
title coverage for - see the Phase 1 postings inspection. The other 14
categories (VAT Executive, ICC/SEVP-DMD, iOS Engineer, ...) are South-Asian
corporate-template titles or otherwise underrepresented in a US-centric
LinkedIn dump and are out of scope until a supplementary source is found.

Patterns are matched against a posting's `title` (case-insensitive) in the
order below - first match wins - so a posting is assigned to exactly one
category even if its title happens to contain more than one keyword. Order
runs most-specific pattern first to minimize a generic pattern (e.g. "project
coordinator") stealing a posting a more specific pattern would also match.
"""
import re
from typing import List, Optional, Tuple

# (category, regex) in priority order - see module docstring.
CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    ("Database Administrator (DBA)", r"database administrator|\bdba\b"),
    ("Full Stack Developer (Python,React js)", r"full[\s-]?stack (?:developer|engineer)"),
    ("DevOps Engineer", r"devops"),
    ("Network Support Engineer", r"network support|network engineer"),
    (
        "System Administrator (Operation & Maintenance of Server, Storage & Service Desk System)",
        r"system administrator",
    ),
    ("Data Engineer", r"\bdata engineer\b"),
    ("Data Science Engineer", r"data scien"),
    ("Senior Software Engineer", r"senior software engineer|sr\.?\s*software engineer"),
    ("Civil Engineer", r"civil engineer"),
    ("Mechanical Engineer", r"mechanical engineer"),
    ("Project Coordinator (Civil)", r"project coordinator"),
    ("Manager- Human Resource Management (HRM)", r"hr manager|human resources? manager|human resource management"),
    ("Marketing Officer", r"marketing officer|marketing coordinator|marketing specialist"),
    ("Business Development Executive", r"business development"),
]

TARGET_CATEGORIES: List[str] = [cat for cat, _ in CATEGORY_KEYWORDS]

_COMPILED = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in CATEGORY_KEYWORDS]


def assign_category(title: str) -> Optional[str]:
    """Return the first (highest-priority) target category whose pattern
    matches `title`, or None if no pattern matches."""
    if not isinstance(title, str) or not title:
        return None
    for cat, pattern in _COMPILED:
        if pattern.search(title):
            return cat
    return None


def count_ambiguous_matches(title: str) -> int:
    """How many distinct target categories' patterns match `title` - used
    only for diagnostics (reporting how often the first-match priority
    order actually had to break a tie)."""
    if not isinstance(title, str) or not title:
        return 0
    return sum(1 for _, pattern in _COMPILED if pattern.search(title))
