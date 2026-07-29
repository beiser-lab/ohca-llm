"""
Shared prompt definitions for the OHCA four-step reasoning chain.

Each note that passes the pre-filters is evaluated through four sequential
binary questions.  A note is labeled OHCA only if it passes all four steps.

Step 1 — Is there a *current* cardiac arrest? (not historical)
Step 2 — Did the arrest start *outside* the hospital?
Step 3 — Is the arrest *non-traumatic*? (LLM fallback for ambiguous cases)
Step 4 — Is the patient *not* a transfer from another facility?

The production classifier in ``qwen_classifier.py`` calls these prompts through
an OpenAI-compatible endpoint; this module intentionally contains no model
runtime client.
"""

from .config import (
    LABEL_OHCA,
    LABEL_NOT_OHCA,
    LABEL_TRAUMATIC,
    LABEL_TRANSFER,
)

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

Do NOT infer cardiac arrest from severe respiratory failure, hypoxemia, altered mental status,
unresponsiveness, BiPAP, intubation, ICU admission, EMS transport, or nursing home/SNF/assisted
living origin unless the note explicitly documents arrest, pulselessness, CPR, defibrillation,
ROSC after arrest, code/cardiac arrest, or a pulseless arrest rhythm.

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


# ── Single-call combined prompt ────────────────────────────────────────────────
# Asks all four questions in ONE model call and returns strict JSON. Same four
# criteria and same semantics as the 4-step chain (step3/step4 phrased positively so
# YES = exclude, matching the invert flags). ~4x fewer calls than the sequential chain;
# used for batch scoring. The sequential chain (classify_note) is kept for interpretability
# and as the reference the single-call output is validated against.
COMBINED_PROMPT = """You are a physician reviewing a clinical note (ED, cath lab, or admission note). The note concerns a possible out-of-hospital cardiac arrest (OHCA). Answer FOUR questions about it.

Q1 ACUTE ARREST: Did an acute cardiac arrest occur during THIS encounter or immediately before it (brought in by EMS after arresting)? Count YES regardless of outcome — still arresting, achieved ROSC, admitted, OR died/pronounced this encounter all count YES. Answer NO only if the arrest is purely HISTORICAL (a prior admission / remote past) or no arrest actually occurred. Do NOT infer cardiac arrest from severe respiratory failure, hypoxemia, altered mental status, unresponsiveness, BiPAP, intubation, ICU admission, EMS transport, or nursing home/SNF/assisted living origin unless arrest, pulselessness, CPR, defibrillation, ROSC after arrest, code/cardiac arrest, or a pulseless arrest rhythm is documented. ("Time of death 2010" = the clock time 20:10, NOT the year — not historical.)

Q2 OUTSIDE-HOSPITAL: Did the patient's FIRST (index) arrest begin OUTSIDE this hospital — at home, a scene, in public, a nursing home/SNF, or before/during EMS transport (in the ambulance)? Answer YES if the first arrest was out-of-hospital, EVEN IF the patient achieved ROSC, arrived with a pulse, and then RE-ARRESTED in the ED. Answer NO only if the patient's FIRST arrest happened after they were already inside this hospital (e.g., presented with a pulse for something else — STEMI, sepsis, SOB — and then arrested in the ED for the first time).

Q3 TRAUMATIC: Was the arrest CAUSED by trauma — gunshot, stabbing, penetrating injury, blunt trauma (MVC, fall, assault), drowning, hanging, or electrocution? Answer YES only if a traumatic injury mechanism caused the arrest. Answer NO for medical causes (cardiac, arrhythmia, VF/VT/PEA/asystole, respiratory, metabolic, STEMI, overdose, unknown). A collapse/fall FROM the arrest, or CPR-related injury, is NOT traumatic.

Q4 TRANSFER: Was the patient TRANSFERRED to this hospital from ANOTHER hospital or acute-care facility (seen elsewhere first, then moved here)? Answer YES only for inter-facility transfer from a hospital/ED/acute-care facility. Answer NO if EMS brought them directly from the field/home/scene/nursing home (a nursing home is NOT an acute-care transfer).

Respond with ONLY a JSON object, no other text:
{{"q1_acute_arrest": "YES"|"NO", "q2_outside_hospital": "YES"|"NO", "q3_traumatic": "YES"|"NO", "q4_transfer": "YES"|"NO", "rationale": "one sentence citing the key evidence"}}

---
CLINICAL NOTE:
{note}
---

JSON:"""

