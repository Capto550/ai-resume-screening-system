from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "nlp",
    "machine learning",
    "scikit-learn",
    "pandas",
    "numpy",
    "streamlit",
]


@dataclass(frozen=True)
class ExtractedHighlights:
    skills: list[str]
    years_experience: int | None
    education: str | None


_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*(?:years|yrs)\b", re.IGNORECASE)
_EDU_RE = re.compile(r"\b(b\.?s\.?|bachelor|m\.?s\.?|master|ph\.?d\.?|doctorate)\b", re.IGNORECASE)


def _find_years_experience(text: str) -> int | None:
    matches = [int(m.group(1)) for m in _YEARS_RE.finditer(text or "")]
    if not matches:
        return None
    # Heuristic: report max mentioned.
    return max(matches)


def _find_education(text: str) -> str | None:
    m = _EDU_RE.search(text or "")
    if not m:
        return None
    token = m.group(0).lower().replace(".", "")
    if token in {"bs", "bachelor"}:
        return "Bachelor"
    if token in {"ms", "master"}:
        return "Master"
    if token in {"phd", "doctorate"}:
        return "PhD"
    return m.group(0)


def extract_highlights(text: str, *, skills: list[str] | None = None) -> ExtractedHighlights:
    t = (text or "").lower()
    skill_list = skills or DEFAULT_SKILLS
    found: list[str] = []
    for s in skill_list:
        if s.lower() in t:
            found.append(s)
    found_sorted = sorted(set(found), key=lambda x: x.lower())
    return ExtractedHighlights(
        skills=found_sorted,
        years_experience=_find_years_experience(text),
        education=_find_education(text),
    )

