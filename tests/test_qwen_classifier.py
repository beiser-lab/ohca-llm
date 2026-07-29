import pandas as pd


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt, *, temperature, seed, max_tokens):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_short_note_with_strong_ohca_signal_needs_llm() -> None:
    from ohca_llm.preprocessor import preprocess_notes

    df = pd.DataFrame(
        {
            "note_text": [
                "EMS CPR for cardiac arrest, ROSC in field.",
                "",
                "AED battery checked.",
            ]
        }
    )

    out = preprocess_notes(df, text_col="note_text")

    assert out.loc[0, "word_count"] < 100
    assert out.loc[0, "short_note_ohca_signal"]
    assert pd.isna(out.loc[0, "pre_filter_label"])
    assert out.loc[0, "needs_llm"]
    assert out.loc[1, "pre_filter_label"] == "Skipped"
    assert out.loc[2, "pre_filter_label"] == "Skipped"


def test_qwen_step1_guardrail_mentions_respiratory_failure_negatives() -> None:
    from ohca_llm.classifier import STEP1_PROMPT

    prompt = STEP1_PROMPT.lower()

    assert "do not infer cardiac arrest" in prompt
    assert "respiratory failure" in prompt
    assert "intubation" in prompt
    assert "nursing home" in prompt


def test_parse_yes_no_uses_final_answer_after_thinking() -> None:
    from ohca_llm.parsing import parse_yes_no

    is_yes, rationale = parse_yes_no(
        "<think>The note says no past history, but the final answer is explicit.</think>\n"
        "Answer: YES\n"
        "The note documents CPR and ROSC before arrival."
    )

    assert is_yes is True
    assert "ROSC" in rationale


def test_qwen_classifier_stops_at_step1_for_respiratory_failure_only() -> None:
    from ohca_llm.qwen_classifier import classify_note_qwen

    note = (
        "Patient brought by EMS from assisted living with altered mental status, "
        "severe hypoxemia, respiratory failure, BiPAP, and intubation in the ED. "
        "No CPR, pulselessness, defibrillation, cardiac arrest, code, or ROSC is documented."
    )
    client = FakeClient([
        "NO\nSevere respiratory failure and intubation are documented, but no arrest evidence is documented."
    ])

    result = classify_note_qwen(note, client)

    assert result["llm_label"] == "No"
    assert result["steps_passed"] == 0
    assert len(client.prompts) == 1
