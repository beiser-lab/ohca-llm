"""Synthetic, PHI-free validation cases for the sequential Qwen classifier.

These notes are fabricated — they contain no protected health information and
are safe to keep in the repository. They exist so the validation harness and
its metrics can be exercised in CI with a deterministic fake client, and so the
Step 1 arrest-existence guardrail is tested directly against the exact
categories that produced false positives in the single-call classifier:
respiratory failure, intubation, altered mental status, and
nursing-home/SNF/ALF origin with no documented arrest.

For real validation, do NOT rely on these. Point the harness at a reviewed
CSV of actual cases inside the HIPAA-compliant environment (see
``docs/validation.md`` / ``ohca_llm.validation``). Each record carries a
``category`` so hard-negative subgroups can be reported separately.

Schema per record:
    id            : str  — stable identifier
    note_text     : str  — synthetic clinical note
    manual_label  : int  — reviewer ground truth (1 = OHCA, 0 = not OHCA)
    category      : str  — subgroup for breakdown (e.g. "hard_negative_resp")
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


# ── Hard negatives: critical illness WITHOUT documented arrest ────────────────
# Every one of these must classify as NOT OHCA. They are the population that
# produced single-call false positives.
HARD_NEGATIVES: List[Dict] = [
    {
        "id": "hn_resp_01",
        "manual_label": 0,
        "category": "hard_negative_resp",
        "note_text": (
            "72M brought by EMS from home with acute hypoxemic respiratory "
            "failure and severe COPD exacerbation. Sats 78% on arrival, placed "
            "on BiPAP then intubated in the ED for worsening work of breathing. "
            "Hemodynamically stable throughout, maintained a pulse the entire "
            "time. No CPR, no chest compressions, no pulselessness, no "
            "defibrillation, and no cardiac arrest at any point. Admitted to "
            "the MICU on the ventilator."
        ),
    },
    {
        "id": "hn_intubation_02",
        "manual_label": 0,
        "category": "hard_negative_intubation",
        "note_text": (
            "58F with status asthmaticus, intubated in the ED for impending "
            "respiratory failure. RSI performed, tube confirmed, sedated and "
            "ventilated. Blood pressure and pulse stable throughout the "
            "intubation. No arrest, no ROSC, no code called."
        ),
    },
    {
        "id": "hn_ams_03",
        "manual_label": 0,
        "category": "hard_negative_ams",
        "note_text": (
            "80M found with altered mental status and unresponsive to voice at "
            "home, brought in by EMS. GCS 8, protected airway with intubation. "
            "Workup for AMS underway — possible sepsis vs metabolic. Patient had "
            "a palpable pulse and measurable blood pressure on EMS arrival and "
            "throughout transport. No cardiac arrest, no CPR administered."
        ),
    },
    {
        "id": "hn_snf_04",
        "manual_label": 0,
        "category": "hard_negative_snf",
        "note_text": (
            "91F sent from her skilled nursing facility for lethargy and "
            "hypoxia. On BiPAP for presumed aspiration pneumonia, then admitted "
            "to the ICU. Nursing home reported no arrest and no CPR. She "
            "maintained a pulse throughout. No pulselessness or defibrillation "
            "documented."
        ),
    },
    {
        "id": "hn_alf_icu_05",
        "manual_label": 0,
        "category": "hard_negative_snf",
        "note_text": (
            "84M from assisted living with EMS transport for severe sepsis and "
            "respiratory distress, admitted to the ICU and intubated. "
            "Persistently hypotensive requiring pressors but never lost pulses. "
            "No code, no chest compressions, no cardiac arrest during this "
            "encounter."
        ),
    },
    {
        "id": "hn_history_only_06",
        "manual_label": 0,
        "category": "hard_negative_historical",
        "note_text": (
            "63M presents with chest pain. PMH significant for CAD, prior MI, "
            "and history of cardiac arrest two years ago status post ICD "
            "placement. Currently hemodynamically stable, no acute arrest this "
            "visit. Troponin pending."
        ),
    },
]


# ── Positives: genuine OHCA that must pass all four steps ─────────────────────
TRUE_POSITIVES: List[Dict] = [
    {
        "id": "tp_field_rosc_01",
        "manual_label": 1,
        "category": "positive_field_arrest",
        "note_text": (
            "55M witnessed collapse at home. Bystander CPR started immediately, "
            "EMS found him in ventricular fibrillation, defibrillated x2 with "
            "ROSC achieved in the field. Brought to this ED intubated with a "
            "pulse. Non-traumatic, presumed cardiac etiology. Taken to cath "
            "lab. Not a transfer — EMS brought him directly here."
        ),
    },
    {
        "id": "tp_ems_ongoing_02",
        "manual_label": 1,
        "category": "positive_field_arrest",
        "note_text": (
            "67F found down and pulseless at home by family. 911 called, EMS "
            "performed ACLS for approximately 15 minutes with ongoing chest "
            "compressions on arrival to the ED for a non-traumatic PEA arrest. "
            "Directly transported by EMS from the scene, no outside hospital "
            "involved."
        ),
    },
    {
        "id": "tp_snf_arrest_03",
        "manual_label": 1,
        "category": "positive_snf_arrest",
        "note_text": (
            "88M at his nursing home had a witnessed cardiac arrest. Staff "
            "started CPR, EMS continued ACLS, asystole then ROSC after "
            "epinephrine. Brought directly to this ED by EMS. Medical arrest, "
            "no trauma. Not transferred from another hospital."
        ),
    },
]


def load_synthetic_validation_set() -> pd.DataFrame:
    """Return the fabricated validation set as a DataFrame.

    Columns: id, note_text, manual_label, category.
    """

    records = HARD_NEGATIVES + TRUE_POSITIVES
    return pd.DataFrame.from_records(
        records, columns=["id", "note_text", "manual_label", "category"]
    )
