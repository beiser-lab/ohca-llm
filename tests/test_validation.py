"""Tests for the validation harness.

These use a scripted fake client so no real model or PHI is involved. The fake
answers each step from simple cues in the note, which is enough to verify the
harness plumbing and that the fixture is internally consistent (hard negatives
stop at Step 1; positives pass all four steps).
"""

import re

import pandas as pd

from ohca_llm.validation import run_validation, summarize, format_report
from ohca_llm.validation_fixtures import load_synthetic_validation_set


class ScriptedClient:
    """Answers each OHCA step from keyword cues in the prompt's note text.

    This mimics a well-behaved model that honors the Step 1 guardrail: it says
    YES to an acute arrest only when explicit arrest evidence (CPR, ROSC,
    pulseless, defibrillation, cardiac arrest, code) is present, and never
    infers arrest from respiratory failure / intubation / AMS / SNF origin.
    """

    ARREST_EVIDENCE = re.compile(
        r"\b(cpr|chest compressions?|rosc|pulseless|defibrillat|"
        r"cardiac arrest|acls|asystole|ventricular fibrillation|pea arrest|"
        r"code)\b",
        re.I,
    )
    HISTORICAL = re.compile(r"history of cardiac arrest|arrest .* years? ago", re.I)
    OUTSIDE = re.compile(
        r"\b(field|scene|at home|bystander|ems (found|performed|continued)|"
        r"nursing home|collapse)\b",
        re.I,
    )
    TRAUMA = re.compile(
        r"\b(gunshot|gsw|stab|penetrating|blunt trauma|mvc|drowning|hanging)\b",
        re.I,
    )
    TRANSFER = re.compile(r"\b(outside hospital|osh|transferred from)\b", re.I)

    def __init__(self):
        self.calls = 0

    def _has_unnegated_arrest_evidence(self, note: str) -> bool:
        for m in self.ARREST_EVIDENCE.finditer(note):
            start = max(0, m.start() - 12)
            window = note[start:m.start()].lower()
            if re.search(r"\b(no|not|without|denies|denied|negative for)\b\s*$", window):
                continue
            return True
        return False

    def _note(self, prompt: str) -> str:
        # The note is interpolated between the CLINICAL NOTE marker and the
        # trailing delimiter in every step prompt.
        m = re.search(r"CLINICAL NOTE:\s*(.*?)\s*---", prompt, re.S)
        return m.group(1) if m else prompt

    def complete(self, prompt, *, temperature, seed, max_tokens):
        self.calls += 1
        note = self._note(prompt)
        upper_prompt = prompt

        # Identify which step this is from the question framing.
        if "ACUTE cardiac arrest event" in upper_prompt:
            # Model the Step 1 guardrail: only count arrest evidence that is
            # NOT explicitly negated ("no CPR", "no cardiac arrest"), and treat
            # purely historical mentions as NO.
            has_evidence = self._has_unnegated_arrest_evidence(note)
            only_historical = bool(self.HISTORICAL.search(note)) and not re.search(
                r"witnessed cardiac arrest|arrest this|found down and pulseless", note, re.I
            )
            ans = "YES" if (has_evidence and not only_historical) else "NO"
            return f"{ans}\nEvidence gate for acute arrest."
        if "began OUTSIDE" in upper_prompt:
            return ("YES\nFirst arrest outside." if self.OUTSIDE.search(note)
                    else "NO\nInside hospital.")
        if "caused by TRAUMA" in upper_prompt:
            return ("YES\nTraumatic." if self.TRAUMA.search(note)
                    else "NO\nMedical cause.")
        if "TRANSFERRED to this" in upper_prompt:
            is_transfer = False
            for m in self.TRANSFER.finditer(note):
                start = max(0, m.start() - 12)
                window = note[start:m.start()].lower()
                if re.search(r"\b(no|not|without|never)\b[\w\s]*$", window):
                    continue
                is_transfer = True
                break
            return "YES\nTransfer." if is_transfer else "NO\nDirect EMS."
        return "NO\nDefault."


def test_synthetic_fixture_loads_and_is_labeled() -> None:
    df = load_synthetic_validation_set()
    assert set(["id", "note_text", "manual_label", "category"]).issubset(df.columns)
    assert df["manual_label"].isin([0, 1]).all()
    # Must contain the hard-negative categories the guardrail targets.
    cats = set(df["category"])
    assert {"hard_negative_resp", "hard_negative_intubation",
            "hard_negative_ams", "hard_negative_snf"}.issubset(cats)


def test_hard_negatives_produce_no_false_positives() -> None:
    df = load_synthetic_validation_set()
    client = ScriptedClient()

    scored = run_validation(df, client)
    summary = summarize(scored)

    # The whole point of the harness: critical-illness hard negatives must not
    # be called OHCA.
    assert summary["overall"]["FP"] == 0, format_report(summary)
    # And the genuine field/SNF arrests should be caught.
    assert summary["overall"]["TP"] >= 3


def test_step1_gate_stops_hard_negatives_early() -> None:
    df = load_synthetic_validation_set()
    hard_neg = df[df["category"].str.startswith("hard_negative")]
    client = ScriptedClient()

    scored = run_validation(hard_neg, client)

    # Every hard negative should be labeled No and, for the respiratory/AMS/SNF
    # cases, fail at Step 1 (steps_passed == 0).
    assert (scored["predicted_ohca"] == 0).all()
    non_historical = scored[scored["category"] != "hard_negative_historical"]
    assert (non_historical["steps_passed"] == 0).all()


def test_keyword_negative_hard_negatives_are_gated_out() -> None:
    """The --score-all-nonblank false-positive population must be ruled out by
    the deterministic gate without ever reaching the LLM."""
    from ohca_llm.preprocessor import (
        clean_text, keyword_gate, short_note_has_ohca_signal,
    )
    from ohca_llm.validation_fixtures import load_keyword_negative_set

    kn = load_keyword_negative_set()
    assert len(kn) >= 7
    for _, row in kn.iterrows():
        clean = clean_text(row["note_text"])
        forwarded = keyword_gate(clean) or short_note_has_ohca_signal(clean)
        assert not forwarded, (
            f"{row['id']} was forwarded to the LLM; the gate should rule it out"
        )


def test_gate_audit_reports_zero_true_ohca_dropped_on_positives() -> None:
    """Every synthetic true OHCA carries arrest vocabulary, so the gate must
    forward all of them (recall 1.0, none dropped)."""
    from ohca_llm.validation import audit_keyword_gate
    from ohca_llm.validation_fixtures import load_synthetic_validation_set

    audit = audit_keyword_gate(load_synthetic_validation_set())
    assert audit["true_ohca_dropped"] == 0
    assert audit["gate_recall_on_ohca"] == 1.0


def test_gate_audit_flags_a_dropped_true_ohca() -> None:
    """A keyword-free note labeled OHCA must show up as wrongly dropped, so the
    audit actually catches gate recall failures."""
    import pandas as pd
    from ohca_llm.validation import audit_keyword_gate

    df = pd.DataFrame(
        {
            "note_text": [
                "Patient with witnessed cardiac arrest, CPR and ROSC.",  # keyworded OHCA
                "Elderly patient found somnolent, admitted for workup.",  # no keyword, but labeled OHCA
            ],
            "manual_label": [1, 1],
        }
    )
    audit = audit_keyword_gate(df)
    assert audit["true_ohca_dropped"] == 1
    assert audit["gate_recall_on_ohca"] == 0.5


def test_report_flags_categories_with_false_positives() -> None:
    # Verify the report annotates categories that contain a false positive.
    scored = pd.DataFrame(
        {
            "manual_label": [0, 1],
            "predicted_ohca": [1, 1],
            "category": ["hard_negative_resp", "positive_field_arrest"],
        }
    )
    summary = summarize(scored)
    report = format_report(summary)
    assert "FP present" in report
    assert summary["by_category"]["hard_negative_resp"]["false_positives"] == 1
