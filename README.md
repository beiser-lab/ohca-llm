# OHCA LLM Pipeline

Python package for detecting out-of-hospital cardiac arrest (OHCA) from emergency department clinical notes using a keyword gate, shortcut filters, and zero-shot LLM classification through a local Ollama model.

## Features

- Cleans emergency department note text and de-identification placeholders.
- Applies cardiac-arrest keyword screening before LLM inference.
- Short-circuits historical arrest, traumatic arrest, transfer, and short-note cases.
- Runs a four-step binary reasoning pipeline for likely OHCA notes.
- Returns annotated `pandas` DataFrames with final labels and binary OHCA predictions.

## Requirements

- Python 3.9 or newer
- Ollama running locally
- The `llama3.2` model pulled into Ollama, unless you pass another model name

Install runtime dependencies:

```bash
pip install -e .
```

Prepare Ollama:

```bash
ollama serve
ollama pull llama3.2
```

## Usage

```python
from ohca_llm import OHCALLMPipeline

pipe = OHCALLMPipeline(model="llama3.2")
results = pipe.run(
    "ed_notes.csv",
    text_col="note_text",
    id_col="note_id",
    output_path="predictions.csv",
)
```

For an in-memory DataFrame:

```python
results = pipe.run_df(df, text_col="note_text", id_col="note_id")
```

For one note:

```python
prediction = pipe.predict_one(note_text)
```

## Output

The pipeline appends fields including:

- `clean_text`
- `word_count`
- `keyword_positive`
- `pre_filter_label`
- `needs_llm`
- `llm_label`
- `llm_confidence`
- `llm_rationale`
- `final_label`
- `predicted_ohca`

## Data Handling

Clinical input and output data are intentionally ignored by git. Keep identifiable or restricted clinical data outside the repository.
