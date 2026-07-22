"""
LLM classifier — 4-step zero-shot chain-of-thought reasoning via Ollama.

Each note that passes the pre-filters is evaluated through four sequential
binary questions.  A note is labeled OHCA only if it passes all four steps.

Step 1 — Is there a *current* cardiac arrest? (not historical)
Step 2 — Did the arrest start *outside* the hospital?
Step 3 — Is the arrest *non-traumatic*? (LLM fallback for ambiguous cases)
Step 4 — Is the patient *not* a transfer from another facility?

The LLM is called once per step; early exit on any "No" answer.
"""

import re
import time
import json
import logging
import requests
from typing import Dict, Optional, Tuple

from .config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    TEMPERATURE,
    SEED,
    MAX_TOKENS,
    LABEL_OHCA,
    LABEL_NOT_OHCA,
    LABEL_TRAUMATIC,
    LABEL_TRANSFER,
)

logger = logging.getLogger(__name__)


# ── Prompt templates ───────────────────────────────────────────────────────────

STEP1_PROMPT = """You are a physician reviewing an emergency department note.

Your task: Determine whether this note describes a patient who is CURRENTLY experiencing a cardiac arrest (or arrived in cardiac arrest) at this ED visit.

Do NOT count:
- A cardiac arrest that happened in the past (history of, s/p, prior)
- A cardiac arrest at another facility that is now resolved
- A patient who came in for something else and has "cardiac arrest" in their past medical history

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

STEP2_PROMPT = """You are a physician reviewing an emergency department note.

The patient in this note had a CURRENT cardiac arrest. Your task: Determine whether the cardiac arrest started OUTSIDE the hospital (i.e., the patient collapsed outside, at home, at a scene, or in the field — and EMS/bystanders responded).

Do NOT count:
- Arrests that started INSIDE a hospital (floor arrest, ICU arrest, code blue after admission)
- Arrests that started in a procedure room, OR, or another in-hospital setting

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

STEP3_PROMPT = """You are a physician reviewing an emergency department note.

The patient had an out-of-hospital cardiac arrest. Your task: Determine whether the arrest was NON-TRAUMATIC (i.e., from a medical cause such as heart disease, arrhythmia, or unknown cause).

Answer NO if:
- The arrest was caused by a gunshot wound, stabbing, or penetrating injury
- The arrest was caused by blunt trauma (MVC, fall, assault)
- The arrest was caused by drowning, hanging, or electrocution

Answer YES if:
- The arrest appears to be from a cardiac, respiratory, or other medical cause
- The cause is unclear but no traumatic mechanism is mentioned

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

STEP4_PROMPT = """You are a physician reviewing an emergency department note.

The patient had a non-traumatic out-of-hospital cardiac arrest. Your task: Determine whether this patient was TRANSFERRED to this emergency department from another hospital or acute-care facility (i.e., they were seen somewhere else first and then moved here).

Answer YES if:
- The patient was transferred from an outside hospital, clinic, or acute-care facility
- The note mentions "transferred from OSH", "outside hospital transfer", or similar
- The patient went to another ED first and is now being transferred here

Answer NO if:
- EMS brought the patient directly to this ED
- The patient collapsed and was brought here as the first ED
- There is no mention of prior hospital contact

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

# Each step: (key, prompt, display_name, fail_label, invert)
#   Normal step (invert=False): a YES answer PASSES, a NO answer FAILS -> fail_label.
#   Inverted step (invert=True): a YES answer FAILS -> fail_label, a NO answer PASSES.
# Inversion lets a criterion be phrased positively even when a "yes" means EXCLUDE
# (e.g. step 4 asks "was this a transfer?" — yes = transfer = exclude). Positive
# phrasing avoids the double-negative that made models flip the polarity of the answer.
STEPS = [
    ("step1_current_arrest", STEP1_PROMPT,
     "Current cardiac arrest",
     LABEL_NOT_OHCA,   # label if step fails
     False),
    ("step2_outside_hospital", STEP2_PROMPT,
     "Outside-hospital origin",
     LABEL_NOT_OHCA,
     False),
    ("step3_non_traumatic", STEP3_PROMPT,
     "Non-traumatic cause",
     LABEL_TRAUMATIC,
     False),
    ("step4_is_transfer", STEP4_PROMPT,
     "Transfer from another facility",
     LABEL_TRANSFER,
     True),            # inverted: YES = transfer = fail
]


# ── Ollama client ──────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, retries: int = 3, backoff: float = 2.0) -> str:
    """
    Call the local Ollama API and return the model's raw text response.

    Parameters
    ----------
    prompt   : Full formatted prompt string.
    retries  : Number of retry attempts on connection errors.
    backoff  : Seconds to wait between retries (doubles each attempt).

    Returns
    -------
    str
        Raw model output text.

    Raises
    ------
    RuntimeError
        If Ollama is unreachable after all retries.
    """
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": TEMPERATURE,
            "seed":        SEED,
            "num_predict": MAX_TOKENS,
        },
    }

    last_error = None
    wait = backoff
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Ollama connection error (attempt {attempt}/{retries}). "
                           f"Is `ollama serve` running? Retrying in {wait:.0f}s…")
            time.sleep(wait)
            wait *= 2
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(f"Ollama request error (attempt {attempt}/{retries}): {e}")
            time.sleep(wait)
            wait *= 2

    raise RuntimeError(
        f"Could not reach Ollama at {OLLAMA_URL} after {retries} attempts. "
        f"Make sure Ollama is running: `ollama serve` and `ollama pull {OLLAMA_MODEL}`.\n"
        f"Last error: {last_error}"
    )


def _parse_yes_no(response: str) -> Tuple[bool, str]:
    """
    Extract YES/NO answer and rationale from the model response.

    Returns
    -------
    (is_yes, rationale)
        is_yes   : True if the model said YES.
        rationale: One-sentence explanation extracted from the response.
    """
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

    answer_line = lines[0] if lines else ""
    rationale   = lines[1] if len(lines) > 1 else response[:200]

    # Normalize: look for YES/NO anywhere in the first line
    upper = answer_line.upper()
    if "YES" in upper:
        return True, rationale
    elif "NO" in upper:
        return False, rationale
    else:
        # Fallback: search entire response
        if re.search(r"\bYES\b", response, re.IGNORECASE):
            return True, rationale
        return False, rationale  # Default to No if ambiguous


# ── Single-note classifier ─────────────────────────────────────────────────────

def classify_note(
    note_text: str,
    verbose: bool = False,
) -> Dict:
    """
    Run the 4-step LLM reasoning chain on a single note.

    Parameters
    ----------
    note_text : str
        Cleaned ED note text (should have already passed the keyword gate).
    verbose   : bool
        If True, print step-by-step decisions to stdout.

    Returns
    -------
    dict with keys:
        llm_label      : str  — "Yes", "No", "Traumatic", or "Transfer"
        llm_confidence : float — naive confidence based on steps passed (0.25–1.0)
        llm_rationale  : str  — concatenated rationale from each step
        steps_passed   : int  — number of steps the note passed (0–4)
        step_responses : dict — raw yes/no and rationale per step
    """
    result = {
        "llm_label":      LABEL_NOT_OHCA,
        "llm_confidence": 0.0,
        "llm_rationale":  "",
        "steps_passed":   0,
        "step_responses": {},
    }

    rationale_parts = []

    for step_key, prompt_template, step_name, fail_label, invert in STEPS:
        prompt = prompt_template.format(note=note_text[:3000])  # truncate for token budget

        try:
            raw_response = _call_ollama(prompt)
        except RuntimeError as e:
            logger.error(f"LLM call failed at {step_key}: {e}")
            result["llm_label"]      = "Error"
            result["llm_rationale"]  = str(e)
            return result

        is_yes, rationale = _parse_yes_no(raw_response)

        # An inverted step fails on YES (e.g. "is this a transfer?" yes = exclude);
        # a normal step fails on NO. step_passed collapses both cases.
        step_passed = (not is_yes) if invert else is_yes

        result["step_responses"][step_key] = {
            "answer":    "YES" if is_yes else "NO",
            "rationale": rationale,
            "raw":       raw_response[:500],
        }
        rationale_parts.append(f"[{step_name}] {rationale}")

        if verbose:
            print(f"  {step_name}: {'✓ YES' if is_yes else '✗ NO'} "
                  f"({'passed' if step_passed else 'FAILED'})")
            print(f"    → {rationale}")

        if not step_passed:
            # Step failed — assign the failure label and stop
            result["llm_label"]      = fail_label
            result["llm_confidence"] = result["steps_passed"] / len(STEPS)
            result["llm_rationale"]  = "; ".join(rationale_parts)
            return result

        result["steps_passed"] += 1

    # All 4 steps passed → OHCA confirmed
    result["llm_label"]      = LABEL_OHCA
    result["llm_confidence"] = 1.0
    result["llm_rationale"]  = "; ".join(rationale_parts)

    return result


# ── Batch classifier ───────────────────────────────────────────────────────────

def batch_classify(
    notes: list,
    verbose: bool = False,
    progress: bool = True,
) -> list:
    """
    Run classify_note on a list of note strings.

    Parameters
    ----------
    notes    : list of str
        Cleaned ED note texts.
    verbose  : bool
        Pass through to classify_note.
    progress : bool
        Show a simple progress counter.

    Returns
    -------
    list of dict
        One result dict per note (same format as classify_note).
    """
    results = []
    total = len(notes)

    for i, note in enumerate(notes):
        if progress and (i % 50 == 0 or i == total - 1):
            print(f"  [{i+1}/{total}] Processing…", flush=True)

        res = classify_note(note, verbose=verbose)
        results.append(res)

    return results
