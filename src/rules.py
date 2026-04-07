"""
src/rules.py — Rule-based sentence boundary component.

Used as one tier in the ensemble. Provides:
  - Hard overrides: blank lines are ALWAYS boundaries
  - Soft signals: punctuation + capitalisation patterns
  - Suppression: known abbreviations should NOT trigger a boundary
"""

import re
import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Abbreviation lists (these should NOT trigger a boundary after their period)
# ---------------------------------------------------------------------------

ABBREVS_EN = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "al", "inc", "ltd", "corp", "co", "dept",
    "u.s", "u.k", "u.n", "ph.d", "m.d", "b.c", "a.d",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
    "fig", "eq", "no", "approx", "est",
}

ABBREVS_IT = {
    "art", "artt", "prof", "dott", "dott.ssa", "ing", "avv", "sig",
    "sig.ra", "p", "pag", "cap", "fig", "tab", "cfr", "ecc",
    "gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago",
    "set", "ott", "nov", "dic",
    "s.p.a", "s.r.l", "s.n.c", "s.a.s",
}

_ABBREV_PATTERN_EN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(ABBREVS_EN, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)
_ABBREV_PATTERN_IT = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(ABBREVS_IT, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)

# Matches end-of-sentence punctuation followed by whitespace and an uppercase letter
_PUNCT_UPPER = re.compile(r"[.!?]\s+[A-Z\u00C0-\u024F]")
# Matches a blank line (paragraph boundary)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def rule_based_probs(text: str, lang: str = "en") -> np.ndarray:
    """
    Compute per-character boundary probabilities using hand-crafted rules.

    Returns:
        np.ndarray of shape (len(text),) with values roughly in [-0.5, 1.0].
        Positive values = evidence FOR a boundary.
        Negative values = evidence AGAINST (abbreviation suppression).
    """
    scores = np.zeros(len(text))

    # --- Positive signal: sentence-ending punctuation + capital letter -------
    for m in _PUNCT_UPPER.finditer(text):
        # m.start() is the punctuation character
        scores[m.start()] += 0.7

    # --- Hard positive: blank lines are ALWAYS boundaries --------------------
    for m in _BLANK_LINE.finditer(text):
        # Find the last non-whitespace character before the blank line
        pos = m.start()
        while pos > 0 and text[pos - 1] in " \t\n":
            pos -= 1
        if pos > 0:
            scores[pos - 1] = 1.0  # override

    # --- Suppression: abbreviations before period ----------------------------
    abbrev_pat = _ABBREV_PATTERN_EN if lang == "en" else _ABBREV_PATTERN_IT
    for m in abbrev_pat.finditer(text):
        # The period is at m.end() - 1
        period_pos = m.end() - 1
        scores[period_pos] = min(scores[period_pos], -0.5)

    return scores


def get_hard_boundaries(text: str) -> List[int]:
    """
    Return character positions that are GUARANTEED boundaries (blank lines).
    These override model predictions.
    """
    boundaries = []
    for m in _BLANK_LINE.finditer(text):
        pos = m.start()
        while pos > 0 and text[pos - 1] in " \t\n":
            pos -= 1
        if pos > 0:
            boundaries.append(pos - 1)
    return sorted(set(boundaries))
