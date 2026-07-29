"""
OHCA LLM Pipeline
=================
Zero-shot chain-of-thought pipeline for detecting Out-of-Hospital
Cardiac Arrest (OHCA) from emergency department clinical notes.

Production path
---------------
Sequential four-step binary reasoning chain served through an
OpenAI-compatible endpoint (Qwen via vLLM). Entry points:
    - ``classify_note_qwen``  — score a single note (production classifier)
    - ``ohca-llm score-csv``  — batch-score a CSV (CLI)
A note is labeled OHCA only if it passes all four steps; the Step 1
arrest-existence gate is the primary accuracy guardrail and explicitly
refuses to infer arrest from respiratory failure, hypoxemia, AMS,
intubation, BiPAP, ICU admission, EMS transport, or SNF/ALF origin.

Legacy / exploratory (not production)
-------------------------------------
The Ollama four-step chain (``classify_note``) and the single-call JSON
prompt in ``classifier`` predate the Qwen path and are retained only for
reference and interpretability. The single-call classifier is treated as
exploratory owing to false positives on critical-illness / intubation
notes and is intentionally not reachable from the CLI.

Performance figures are dataset- and model-specific and should be
established per deployment against a reviewed validation set rather than
assumed from prior runs.
"""

from .pipeline import OHCALLMPipeline
from .preprocessor import preprocess_notes, keyword_gate
from .classifier import classify_note, batch_classify
from .qwen_classifier import classify_note_qwen
from .validation import run_validation, summarize, format_report
from .validation import audit_keyword_gate, format_gate_audit
from .validation_fixtures import (
    load_synthetic_validation_set,
    load_keyword_negative_set,
)

__version__ = "6.0.0"
__all__ = [
    "OHCALLMPipeline",
    "preprocess_notes",
    "keyword_gate",
    "classify_note",
    "batch_classify",
    "classify_note_qwen",
    "run_validation",
    "summarize",
    "format_report",
    "audit_keyword_gate",
    "format_gate_audit",
    "load_synthetic_validation_set",
    "load_keyword_negative_set",
]
