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

Your task: Determine whether this encounter involves an ACUTE cardiac arrest event — one that occurred during this presentation or immediately before it (e.g., brought in by EMS after arresting).

Count as YES regardless of the OUTCOME of the arrest. All of these are YES:
- The patient is still in cardiac arrest / actively receiving CPR
- The patient achieved ROSC (return of spontaneous circulation) and now has a pulse
- The patient was resuscitated and admitted (e.g., to the ICU or cath lab)
- The patient died / was pronounced dead / "time of death" was called during this encounter
A resolved or fatal arrest is STILL an acute arrest event for this purpose. Do not answer NO
just because the arrest is no longer ongoing, the patient has a pulse, or the patient has died.

Answer NO only if:
- The arrest is purely HISTORICAL (happened on a prior admission or in the remote past;
  "history of cardiac arrest", "s/p arrest [months/years ago]")
- "Cardiac arrest" appears only in the past medical history and no acute arrest occurred this encounter
- No cardiac arrest actually occurred (the term is negated or hypothetical)

Note: a "time of death" written as a 4-digit clock time (e.g. "time of death 2010" = 20:10)
is a TIME, not a calendar year — it does not make the arrest historical.

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

STEP2_PROMPT = """You are a physician reviewing a clinical note (it may be an ED note, a cath lab / procedure note, or an admission note).

The patient in this note had an acute cardiac arrest. Your task: Determine whether the patient's FIRST (index) cardiac arrest began OUTSIDE the hospital — before they arrived at or were inside this hospital.

Focus on WHERE THE FIRST ARREST HAPPENED, not on where the patient came from or why they first
presented, and not on whether they later had additional arrests.

Answer YES if the FIRST arrest began outside the hospital:
- Collapsed at home, at a scene, in the field, in public, at a nursing home / SNF, etc.
- Arrested before OR during EMS transport (in the ambulance); bystander/EMS CPR before arrival
- "Arrived in arrest", "found down", "brought in by EMS after arrest", "s/p ROSC" from a field/EMS arrest
- IMPORTANT: If the patient arrested outside (field or ambulance), achieved ROSC, arrived with a
  pulse, and then RE-ARRESTED in the ED, the answer is still YES — the index arrest was
  out-of-hospital. Re-arrest in the ED after an out-of-hospital arrest does NOT make it in-hospital.
- (A brief, single line stating a field/EMS/out-of-hospital arrest is sufficient even in a
  cath lab or admission note that omits the full prehospital narrative.)

Answer NO only if the patient's FIRST arrest began inside this hospital, with NO prior arrest outside:
- The patient presented (walked in, or brought by EMS) for something ELSE — chest pain, STEMI,
  altered mental status, sepsis, shortness of breath — had a pulse throughout arrival, and then
  arrested for the FIRST time while in the ED or after admission. Answer NO.
- Floor arrest, ICU arrest, code blue after admission, arrest in a procedure room / OR, with no
  preceding out-of-hospital arrest.
- Key test: NO only if the ONLY arrest(s) occurred after the patient was already inside the
  hospital. If ANY arrest occurred in the field or ambulance before arrival, answer YES.

Answer with ONLY one of these two words: YES or NO.
Then on a new line, write one sentence explaining your reasoning: state WHERE the FIRST arrest occurred.

---
CLINICAL NOTE:
{note}
---

Answer (YES or NO):"""

STEP3_PROMPT = """You are a physician reviewing a clinical note.

The patient had a cardiac arrest. Your task: Determine whether the arrest was caused by TRAUMA (a physical injury mechanism).

Answer YES (traumatic) only if the arrest was caused by an injury mechanism such as:
- Gunshot wound, stabbing, or penetrating injury
- Blunt trauma (motor vehicle crash, fall from height, assault)
- Drowning, hanging/strangulation, or electrocution

Answer NO (non-traumatic / medical) if:
- The arrest was from a medical cause — cardiac, arrhythmia (VF/VT/PEA/asystole), respiratory,
  metabolic, STEMI, heart failure, ESRD/electrolyte, overdose, or any illness
- The cause is unclear or unknown, but NO traumatic injury mechanism is described
- The note only mentions incidental/minor injury (e.g., a fall AT THE TIME of collapse from the
  arrest itself, CPR-related rib fracture) without trauma being the CAUSE of the arrest

Focus on the CAUSE of the arrest. A medical arrest is NO even if the patient collapsed, fell,
or was found down — collapsing is not a traumatic mechanism unless injury caused the arrest.

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
    ("step3_is_traumatic", STEP3_PROMPT,
     "Traumatic cause",
     LABEL_TRAUMATIC,
     True),            # inverted: YES = traumatic = fail
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
