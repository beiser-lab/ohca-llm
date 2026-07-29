"""
Configuration and constants for the OHCA LLM pipeline.
"""

# ── Inference settings (production: Qwen via vLLM, OpenAI-compatible) ──────────
# These constants drive the production sequential classifier
# (`qwen_classifier.classify_note_qwen`) through `OpenAIChatClient`. They are
# runtime-agnostic: the same values apply to any OpenAI-compatible server
# (vLLM, llama.cpp, etc.). Keep them deterministic — this phenotype is scored
# once and audited, so reproducibility matters more than sampling diversity.
QWEN_MODEL     = "Qwen2.5-7B-Instruct"   # Served model name (matches CLI default)
QWEN_BASE_URL  = "http://127.0.0.1:8000/v1"  # vLLM OpenAI-compatible endpoint
TEMPERATURE    = 0.0               # Deterministic inference (greedy)
SEED           = 42                # Passed to the server when it honors seeds
MAX_TOKENS     = 400               # Per-step response cap (binary answer + 1 line)

# ── Legacy Ollama settings (exploratory only — NOT the production path) ───────
# The Ollama four-step chain in `classifier.py` (`classify_note`) predates the
# Qwen/vLLM path and is retained only for reference and interpretability. The
# single-call JSON prompt in that module is exploratory and intentionally not
# reachable from the CLI. Do not wire these into the production scorer.
OLLAMA_MODEL   = "llama3.2"        # Model tag to pull/run
OLLAMA_URL     = "http://localhost:11434/api/generate"

# ── Keyword gate ─────────────────────────────────────────────────────────────
# Notes must contain ≥1 keyword to reach the LLM
CARDIAC_KEYWORDS = [
    "cardiac arrest",
    "cardiopulmonary arrest",
    "cpr",
    "pulseless",
    "rosc",
    "return of spontaneous circulation",
    "defibrillat",
    "asystole",
    "vfib",
    "v-fib",
    "ventricular fibrillation",
    "ventricular tachycardia",
    "v-tach",
    "vtach",
    "found unresponsive",
    "found down",
    "collapsed",
    "pea",
    "pulseless electrical activity",
    "chest compressions",
    "resuscitat",
    "acls",
    "bls",
    "aed",
    "automated external defibrillat",
    "code blue",
]

# Short notes can still be definitive OHCA evidence. These regexes allow a
# sub-threshold note to reach the LLM when it contains a strong current-arrest
# cue; note length is kept as an audit flag, not a terminal exclusion.
SHORT_NOTE_STRONG_OHCA_PATTERNS = [
    r"\bcardiac\s+arrest\b",
    r"\b(?:s/?p|status\s+post)\s+(?:cardiac\s+)?arrest\b",
    r"\b(?:arrived|brought\s+in|presents?|presenting)\s+(?:in|after|s/?p)?\s*(?:cardiac\s+)?arrest\b",
    r"\b(?:found|found\s+down|found\s+unresponsive)\b.{0,80}\b(?:cpr|arrest|pulseless|rosc)\b",
    r"\b(?:cpr|compressions?|acls|bls)\b.{0,80}\b(?:ems|field|scene|prehospital|pre-?hospital|arrival|arrest)\b",
    r"\b(?:ems|field|scene|prehospital|pre-?hospital)\b.{0,80}\b(?:cpr|compressions?|rosc|arrest|pulseless)\b",
    r"\b(?:rosc|return\s+of\s+spontaneous\s+circulation)\b",
    r"\b(?:pea|asystole|vfib|v-?fib|ventricular\s+fibrillation|v-?tach|vtach|ventricular\s+tachycardia)\b.{0,80}\b(?:arrest|cpr|rosc|pulseless)\b",
    r"\bno\s+(?:pulse|pulses?|rosc)\s+on\s+(?:arrival|presentation)\b",
]

# ── PMH shortcut ─────────────────────────────────────────────────────────────
# If note contains ONLY historical arrest language (no current indicators),
# skip LLM and label as Not OHCA immediately.
PMH_ONLY_PATTERNS = [
    r"s/p\s+cardiac\s+arrest",
    r"h/o\s+cardiac\s+arrest",
    r"history\s+of\s+cardiac\s+arrest",
    r"c/b\s+cardiac\s+arrest",
    r"complicated\s+by\s+cardiac\s+arrest",
    r"prior\s+cardiac\s+arrest",
    r"previous\s+cardiac\s+arrest",
    r"past\s+cardiac\s+arrest",
    r"pmh\w*.{0,50}cardiac\s+arrest",
    r"icd\s+.{0,50}cardiac\s+arrest",
]

# Current-arrest patterns that override the PMH shortcut
CURRENT_ARREST_OVERRIDE_PATTERNS = [
    r"presents?\s+(?:to\s+(?:ed|er|emergency)\s+)?via\s+ems\s+for\s+(?:cardiac\s+)?arrest",
    r"status\s+post\s+cardiac\s+arrest\s+(?:with|now|currently|presenting|p/w)",
    r"p/w\s+(?:s/p\s+)?cardiac\s+arrest",
    r"presents?\s+(?:to\s+(?:ed|er))?\s+for\s+cardiac\s+arrest",
    r"no\s+(?:rosc|pulse|pulses?)\s+on\s+(?:arrival|presentation)",
    r"acls\s+(?:performed|initiated|started|done)\s+for\s+(?:approximately\s+)?\d+",
    r"brought\s+(?:in\s+)?(?:by|via)\s+ems\s+(?:in\s+)?(?:cardiac\s+)?arrest",
]

# ── Outside-hospital patterns (Step 2 boost) ─────────────────────────────────
OUTSIDE_ARREST_PATTERNS = [
    r"found\s+(?:down|unresponsive|pulseless)\s+(?:at\s+)?(?:home|outside|in\s+the\s+field|at\s+scene)",
    r"bystander\s+cpr",
    r"ems\s+(?:initiated|started|performed|found|arrived)",
    r"911\s+called",
    r"paramedic",
    r"ambulance",
    r"scene",
    r"field\s+(?:resuscitation|cpr|rosc)",
    r"rosc\s+(?:in|achieved\s+in)\s+(?:field|scene|pre-?hospital)",
    r"pre-?hospital\s+(?:arrest|rosc|cpr)",
    r"presents?\s+(?:to\s+(?:ed|er|emergency)\s+)?via\s+ems",
    r"via\s+ems\s+for\s+(?:cardiac\s+)?arrest",
    r"brought\s+(?:to\s+)?(?:ed|er)\s+(?:via|by)\s+ems",
]

# ── Trauma exclusion patterns (auto-No without LLM call) ─────────────────────
TRAUMA_PATTERNS = [
    r"gunshot",
    r"gsw",
    r"stab(?:bing)?",
    r"penetrating\s+trauma",
    r"blunt\s+trauma",
    r"mvc",
    r"motor\s+vehicle\s+(?:accident|collision|crash)",
    r"traumatic\s+arrest",
    r"traumatic\s+cardiac\s+arrest",
    r"drowning",
    r"hanging",
    r"electrocution",
]

# ── Transfer exclusion patterns (auto-No without LLM call) ───────────────────
TRANSFER_PATTERNS = [
    r"transfer(?:red)?\s+from\s+(?:outside|another|external|outside\s+hospital)",
    r"transferred\s+from\s+(?:an?\s+)?(?:osh|outside\s+hospital)",
    r"outside\s+hospital\s+transfer",
    r"inter-?facility\s+transfer",
    r"(?:osh|outside\s+hospital)\s+(?:transfer|cath|cath\s+lab)",
]

# ── Output labels ─────────────────────────────────────────────────────────────
LABEL_OHCA      = "Yes"
LABEL_NOT_OHCA  = "No"
LABEL_TRAUMATIC = "Traumatic"
LABEL_TRANSFER  = "Transfer"
LABEL_SKIP      = "Skipped"      # Note too short / no text

# ── Processing ────────────────────────────────────────────────────────────────
MIN_NOTE_WORDS = 100             # Short notes without strong OHCA signal are skipped
