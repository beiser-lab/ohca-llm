"""
OHCA LLM Pipeline
=================
Zero-shot chain-of-thought pipeline for detecting Out-of-Hospital
Cardiac Arrest (OHCA) from emergency department clinical notes.

Model: Llama 3.2 (3B) via Ollama
Approach: 4-step binary reasoning chain (no training required)
Performance on UChicago C19 LDS ED notes (n=26,755; 49 confirmed OHCA):
    Sensitivity : 0.939
    Specificity : 1.000
    PPV         : 1.000
    F1          : 0.968
    False positives: 0
"""

from .pipeline import OHCALLMPipeline
from .preprocessor import preprocess_notes, keyword_gate
from .classifier import classify_note, batch_classify
from .qwen_classifier import classify_note_qwen

__version__ = "6.0.0"
__all__ = [
    "OHCALLMPipeline",
    "preprocess_notes",
    "keyword_gate",
    "classify_note",
    "batch_classify",
    "classify_note_qwen",
]
