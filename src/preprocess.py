from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]")  # keep basic skill tokens: c++, c#, node.js


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _NON_WORD_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()

