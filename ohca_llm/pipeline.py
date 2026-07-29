"""
OHCALLMPipeline — high-level interface for running the full pipeline.

Usage
-----
from ohca_llm import OHCALLMPipeline

pipe = OHCALLMPipeline()
results = pipe.run("path/to/ed_notes.csv", text_col="note_text")
results.to_csv("predictions.csv", index=False)
"""

import time
import logging
import pandas as pd
from typing import Optional

from .preprocessor import preprocess_notes
from .classifier import classify_note
from .config import LABEL_NOT_OHCA, LABEL_OHCA, LABEL_SKIP

logger = logging.getLogger(__name__)


class OHCALLMPipeline:
    """
    End-to-end OHCA detection pipeline.

    Workflow
    --------
    1.  Load and clean notes.
    2.  Apply keyword gate       → drops ~94% of notes immediately.
    3.  Apply shortcut filters   → drops historical, traumatic, transfer notes.
    4.  Run 4-step LLM reasoning → on remaining notes only.
    5.  Combine labels and return annotated DataFrame.

    Parameters
    ----------
    model : str
        Ollama model tag to use (default: "llama3.2").
    verbose : bool
        Print step-by-step LLM reasoning for each note.
    """

    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        from .config import OLLAMA_MODEL
        import ohca_llm.config as cfg
        if model != OLLAMA_MODEL:
            cfg.OLLAMA_MODEL = model  # override global
        self.model   = model
        self.verbose = verbose

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        text_col:   str            = "note_text",
        id_col:     Optional[str]  = None,
        output_path: Optional[str] = None,
        sep:        str            = ",",
        encoding:   str            = "utf-8",
    ) -> pd.DataFrame:
        """
        Run the full pipeline on a CSV/TSV file of ED notes.

        Parameters
        ----------
        input_path  : Path to input file (CSV or pipe-delimited TXT).
        text_col    : Column name containing note text.
        id_col      : Optional patient/note ID column (kept in output).
        output_path : If provided, save results to this CSV path.
        sep         : Delimiter for the input file (default: ",").
        encoding    : File encoding (default: "utf-8").

        Returns
        -------
        pd.DataFrame
            Original columns + pipeline output columns:
              clean_text, keyword_positive, pre_filter_label, needs_llm,
              llm_label, llm_confidence, llm_rationale, final_label,
              predicted_ohca (0/1 binary)
        """
        print(f"[OHCALLMPipeline] Loading {input_path} …")
        df = pd.read_csv(input_path, sep=sep, encoding=encoding, low_memory=False)
        print(f"  Loaded {len(df):,} rows")

        return self.run_df(df, text_col=text_col, id_col=id_col, output_path=output_path)

    def run_df(
        self,
        df:          pd.DataFrame,
        text_col:    str           = "note_text",
        id_col:      Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline on an existing DataFrame.

        Same as run() but accepts a DataFrame directly instead of a file path.
        """
        t0 = time.time()

        # ── Step 1: pre-filters ────────────────────────────────────────────────
        print("\n[Step 1/2] Pre-filtering notes …")
        df = preprocess_notes(df, text_col=text_col, id_col=id_col)

        llm_mask  = df["needs_llm"]
        llm_notes = df.loc[llm_mask, "clean_text"].tolist()
        n_llm     = len(llm_notes)

        # ── Step 2: LLM classification ─────────────────────────────────────────
        print(f"\n[Step 2/2] Running LLM on {n_llm:,} keyword-positive notes …")
        print(f"  Model : {self.model}")
        print(f"  ETA   : ~{n_llm * 20 / 60:.0f} min  (≈20s per note)\n")

        llm_results = []
        for i, note in enumerate(llm_notes):
            if i % 100 == 0 or i == n_llm - 1:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{n_llm}]  elapsed={elapsed/60:.1f}m", flush=True)

            res = classify_note(note, verbose=self.verbose)
            llm_results.append(res)

        # ── Merge LLM results back ─────────────────────────────────────────────
        llm_df = pd.DataFrame(llm_results, index=df.index[llm_mask])
        for col in ["llm_label", "llm_confidence", "llm_rationale", "steps_passed"]:
            df[col] = None
            df.loc[llm_mask, col] = llm_df[col]

        # ── Final label: combine pre-filter and LLM ────────────────────────────
        df["final_label"] = df.apply(
            lambda row: row["pre_filter_label"]
            if not row["needs_llm"]
            else row["llm_label"],
            axis=1,
        )

        df["predicted_ohca"] = (df["final_label"] == LABEL_OHCA).astype(int)

        # ── Summary ────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        n_ohca  = df["predicted_ohca"].sum()
        print(f"\n{'='*60}")
        print(f"  Pipeline complete in {elapsed/60:.1f} min")
        print(f"  Total notes      : {len(df):,}")
        print(f"  Predicted OHCA   : {n_ohca:,}")
        print(f"  Predicted Not    : {len(df) - n_ohca:,}")
        print(f"{'='*60}\n")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  Saved → {output_path}")

        return df

    # ── Convenience: single note ───────────────────────────────────────────────

    def predict_one(self, note_text: str) -> dict:
        """
        Classify a single raw note string.

        Returns the full result dict from classify_note, plus a
        'predicted_ohca' key (0 or 1).

        Parameters
        ----------
        note_text : str
            Raw note text (cleaning is applied automatically).

        Returns
        -------
        dict
        """
        from .preprocessor import clean_text, keyword_gate, short_note_has_ohca_signal, trauma_shortcut
        from .preprocessor import transfer_shortcut, pmh_shortcut
        from .config import MIN_NOTE_WORDS

        text = clean_text(note_text)
        short_note_signal = short_note_has_ohca_signal(text)

        if len(text.split()) == 0:
            return {"final_label": LABEL_SKIP, "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "Blank note"}

        if len(text.split()) < MIN_NOTE_WORDS and not short_note_signal:
            return {"final_label": LABEL_SKIP, "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "Short note without strong OHCA signal"}

        if not keyword_gate(text):
            return {"final_label": LABEL_NOT_OHCA, "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "No cardiac arrest keywords found"}

        if trauma_shortcut(text):
            return {"final_label": "Traumatic", "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "Traumatic arrest pattern detected"}

        if transfer_shortcut(text):
            return {"final_label": "Transfer", "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "Transfer patient pattern detected"}

        if pmh_shortcut(text):
            return {"final_label": LABEL_NOT_OHCA, "predicted_ohca": 0,
                    "llm_label": None, "llm_rationale": "Historical arrest only (PMH shortcut)"}

        result = classify_note(text, verbose=self.verbose)
        result["final_label"]    = result["llm_label"]
        result["predicted_ohca"] = 1 if result["llm_label"] == LABEL_OHCA else 0
        return result
