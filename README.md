# OHCA LLM Pipeline

Python package for detecting out-of-hospital cardiac arrest (OHCA) from
emergency department clinical notes using a keyword gate, shortcut filters, and
zero-shot LLM classification. The production path is a sequential four-step
binary reasoning chain served through an OpenAI-compatible endpoint (Qwen via
vLLM).

## Features

- Cleans emergency department note text and de-identification placeholders.
- Applies cardiac-arrest keyword screening as an audit flag before LLM inference.
- Short-circuits historical arrest, traumatic arrest, transfer, and
  non-informative short-note cases.
- Sends short notes to the LLM when they contain strong current-OHCA cues such
  as cardiac arrest, CPR, ROSC, field/EMS arrest, or pulseless rhythms.
- Runs a four-step binary reasoning chain for likely OHCA notes; a note is
  labeled OHCA only if it passes all four steps.
- Step 1 refuses to infer arrest from respiratory failure, hypoxemia, altered
  mental status, intubation, BiPAP, ICU admission, EMS transport, or
  nursing-home/SNF/ALF origin unless arrest evidence is documented.
- Returns annotated `pandas` DataFrames with labels and binary OHCA predictions.

## Requirements

- Python 3.9 or newer
- An OpenAI-compatible LLM server reachable over HTTP. In production this is
  vLLM serving a Qwen instruct model (default `Qwen2.5-7B-Instruct` at
  `http://127.0.0.1:8000/v1`).

Install runtime dependencies:

```bash
pip install -e .
```

Serve the model (example, vLLM):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen2.5-7B-Instruct \
    --port 8000
```

## Usage

### CLI (batch scoring)

```bash
ohca-llm score-csv \
    --input ed_notes.csv \
    --out predictions.csv \
    --text-col "note text" \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen2.5-7B-Instruct \
    --score-all-nonblank
```

The scorer never drops rows, supports checkpoint/resume (re-run with the same
`--out` to continue), and writes per-step detail for audit. `--single-call` is
deliberately not offered: the single-call JSON classifier is exploratory only
because it produced false positives on critical-illness / intubation notes.

### Python (single note)

```python
from ohca_llm import classify_note_qwen
from ohca_llm.openai_client import OpenAIChatClient

client = OpenAIChatClient()  # defaults to Qwen via vLLM
result = classify_note_qwen(note_text, client)
print(result["llm_label"], result["llm_step_detail"])
```

## Output

Scoring appends fields including:

- `clean_text`
- `word_count`
- `keyword_positive`
- `short_note_ohca_signal`
- `llm_label`
- `llm_confidence`
- `llm_rationale`
- `llm_step_detail`
- `steps_passed`
- `residential_rescue`
- `llm_ran` / `llm_gated_reason`
- `llm_predicted_ohca`

## Legacy / exploratory components

The Ollama four-step chain (`classify_note`) and the single-call JSON prompt in
`ohca_llm.classifier` predate the Qwen/vLLM path and are retained for reference
and interpretability only. They are not part of the production scorer and are
not reachable from the CLI.

## Validation

Performance depends on model and dataset. Establish sensitivity/specificity per
deployment against a reviewed validation set rather than assuming figures from
prior runs. Include hard negatives — respiratory failure, intubation, altered
mental status, and nursing-home/SNF/ALF cases with no documented arrest — so the
Step 1 guardrail is exercised directly.

## Data Handling

Clinical input and output data are intentionally ignored by git. Keep
identifiable or restricted clinical data outside the repository.
