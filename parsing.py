"""Parsing helpers for binary YES/NO LLM responses."""

from __future__ import annotations

import re
from typing import Tuple


THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<think(?:ing)?>.*$", re.I | re.S)
ANSWER_ANCHOR_RE = re.compile(
    r"answer\s*(?:\([^)]*\))?\s*[:\-]?\s*(YES|NO)\b", re.I
)


def parse_yes_no(response: str) -> Tuple[bool, str]:
    """Return a robust YES/NO verdict and concise rationale.

    The parser is designed for local reasoning models that may emit thinking
    text before the final answer. It strips explicit thinking blocks and then
    resolves the final answer by priority:

    1. Last explicit "Answer: YES/NO" anchor.
    2. Last line containing only YES or NO.
    3. Last standalone YES/NO token.

    Ambiguous output defaults to NO, matching the conservative behavior of the
    original classifier.
    """

    raw = response or ""
    text = THINK_BLOCK_RE.sub(" ", raw)
    stripped = THINK_OPEN_RE.sub(" ", text).strip()
    if re.search(r"\b(YES|NO)\b", stripped, re.I):
        text = stripped
    else:
        text = text.strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    is_yes = None

    anchors = ANSWER_ANCHOR_RE.findall(text)
    if anchors:
        is_yes = anchors[-1].upper() == "YES"

    if is_yes is None:
        verdict_lines = [
            line for line in lines
            if re.fullmatch(r"(YES|NO)[.:!) ]*", line.upper())
        ]
        if verdict_lines:
            is_yes = verdict_lines[-1].upper().startswith("YES")

    if is_yes is None:
        tokens = re.findall(r"\b(YES|NO)\b", text.upper())
        if tokens:
            is_yes = tokens[-1] == "YES"

    rationale = ""
    for line in lines:
        if not re.fullmatch(r"(YES|NO)[.:!) ]*", line.upper()):
            rationale = line
    if not rationale:
        rationale = lines[-1] if lines else text[:200]

    return (False if is_yes is None else is_yes), rationale[:400]
