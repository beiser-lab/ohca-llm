"""
Preprocessing utilities for the OHCA LLM pipeline.
"""

import re
import pandas as pd
from typing import Optional
from .config import (
    CARDIAC_KEYWORDS,
    PMH_ONLY_PATTERNS,
    CURRENT_ARREST_OVERRIDE_PATTERNS,
    TRAUMA_PATTERNS,
    TRANSFER_PATTERNS,
    MIN_NOTE_WORDS,
    LABEL_NOT_OHCA,
    LABEL_TRAUMATIC,
    LABEL_TRANSFER,
    LABEL_SKIP,
)


def clean_text(text: str) -> str:
    """
    Normalize a clinical note for downstream processing.
    Removes de-identification placeholders (DATE, AGE, NAME, IDNUM),
    collapses excess whitespace, and strips leading/trailing space.
    """
    if not isinstance(text, str):
        return ""

    # Specific placeholders first so they don't get swallowed by the generic pattern
    text = re.sub(r"\*\*IDNUM\*\*", "[ID]", text)
    text = re.sub(r"\*\*DATE<[^>]*>\*\*", "[DATE]", text)
    text = re.sub(r"\*\*AGE<[^>]*>\*\*", "[AGE]", text)
    text = re.sub(r"\*\*\w+<[^>]*>\*\*", "[REDACTED]", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def keyword_gate(text: str) -> bool:
    """
    Return True if the note contains at least one cardiac-arrest keyword.
    Only keyword-positive notes are forwarded to the LLM (~6% of all ED notes).
    """
    lower = text.lower()
    return any(kw in lower for kw in CARDIAC_KEYWORDS)


def pmh_shortcut(text: str) -> bool:
    """
    Return True if the note describes only a *historical* cardiac arrest.
    Fires when PMH-arrest language is found AND no current-arrest override
    is present AND no active presentation language ("presents", "p/w") is nearby.
    """
    lower = text.lower()

    # Check for current-arrest overrides first
    for pattern in CURRENT_ARREST_OVERRIDE_PATTERNS:
        if re.search(pattern, lower):
            return False

    # Check for PMH arrest language
    for pattern in PMH_ONLY_PATTERNS:
        if re.search(pattern, lower):
            # Make sure "presents" or "p/w" is not nearby cardiac arrest language
            # (indicating a current event, not just old PMH)
            nearby_current = re.search(
                r'\b(p/?w|presenting|brought\s+in|arrived|on\s+arrival|'
                r'transferred\s+for|ems\s+(?:brought|called|arrived))\b',
                lower
            )
            if not nearby_current:
                return True

    return False


def trauma_shortcut(text: str) -> bool:
    """Return True if the arrest appears traumatic in origin."""
    lower = text.lower()
    return any(re.search(p, lower) for p in TRAUMA_PATTERNS)


def transfer_shortcut(text: str) -> bool:
    """Return True if the patient was transferred from another facility."""
    lower = text.lower()
    return any(re.search(p, lower) for p in TRANSFER_PATTERNS)


def preprocess_notes(
    df: pd.DataFrame,
    text_col: str = "note_text",
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply all pre-filters to a DataFrame of clinical notes.

    Adds columns: clean_text, word_count, keyword_positive,
                  pre_filter_label, needs_llm
    """
    df = df.copy()

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    df["clean_text"] = df[text_col].apply(clean_text)
    df["word_count"]  = df["clean_text"].apply(lambda t: len(t.split()))
    df["keyword_positive"] = df["clean_text"].apply(keyword_gate)

    def _pre_filter(row) -> Optional[str]:
        text = row["clean_text"]
        if row["word_count"] < MIN_NOTE_WORDS:
            return LABEL_SKIP
        if not row["keyword_positive"]:
            return LABEL_NOT_OHCA
        if trauma_shortcut(text):
            return LABEL_TRAUMATIC
        if transfer_shortcut(text):
            return LABEL_TRANSFER
        if pmh_shortcut(text):
            return LABEL_NOT_OHCA
        return None  # Needs LLM

    df["pre_filter_label"] = df.apply(_pre_filter, axis=1)
    df["needs_llm"] = df["pre_filter_label"].isna()

    n_total  = len(df)
    n_llm    = df["needs_llm"].sum()
    n_kw_pos = df["keyword_positive"].sum()
    pct      = 100 * n_llm / n_total if n_total else 0

    print(f"[preprocess_notes] {n_total:,} total notes")
    print(f"  Keyword-positive : {n_kw_pos:,}  ({100*n_kw_pos/n_total:.1f}%)")
    print(f"  Needs LLM        : {n_llm:,}  ({pct:.1f}% of total)")
    print(f"  Pre-filtered out : {n_total - n_llm:,}")

    return df
