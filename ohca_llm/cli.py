"""Command line entrypoints for OHCA LLM scoring."""

from __future__ import annotations

if __package__ in (None, ""):
    import importlib.util
    import sys
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ohca_llm",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ohca_llm"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from ohca_llm.cli import main as _main

    if __name__ == "__main__":
        _main()
    raise SystemExit

import argparse
import os
import time
from collections import Counter

import pandas as pd

from .config import (
    LABEL_NOT_OHCA,
    LABEL_SKIP,
    MIN_NOTE_WORDS,
    QWEN_BASE_URL,
    QWEN_MODEL,
)
from .openai_client import OpenAIChatClient
from .preprocessor import (
    clean_text,
    keyword_gate,
    pmh_shortcut,
    short_note_has_ohca_signal,
    transfer_shortcut,
    trauma_shortcut,
)
from .qwen_classifier import classify_note_qwen


DEFAULT_TEXT_COL = "note text"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ohca-llm",
        description="Score clinical notes for OHCA using a sequential local LLM chain.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_csv = subparsers.add_parser(
        "score-csv",
        help="Score a CSV file with the sequential Qwen/OpenAI-compatible classifier.",
    )
    score_csv.add_argument("--input", "--union", dest="input", required=True)
    score_csv.add_argument("--out", required=True)
    score_csv.add_argument("--text-col", default=DEFAULT_TEXT_COL)
    score_csv.add_argument("--base-url", default=QWEN_BASE_URL)
    score_csv.add_argument("--model", default=QWEN_MODEL)
    score_csv.add_argument("--note-char-cap", type=int, default=3000)
    score_csv.add_argument("--max-tokens", type=int, default=400)
    score_csv.add_argument("--limit", type=int, default=None)
    score_csv.add_argument(
        "--score-all-nonblank",
        action="store_true",
        help="Send every nonblank note to the LLM; keep keyword_positive as audit.",
    )
    score_csv.add_argument("--debug-raw", action="store_true")
    score_csv.add_argument("--no-seed", action="store_true")
    score_csv.add_argument(
        "--thinking",
        action="store_true",
        help="Re-enable model thinking output. Default is disabled.",
    )
    score_csv.add_argument("--req-pause", type=float, default=0.0)
    score_csv.add_argument(
        "--no-residential-rescue",
        action="store_true",
        help="Do not rescue nursing-home/home origins from transfer labels.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Run the sequential Qwen classifier over a labeled validation set "
             "and report metrics with a hard-negative breakdown.",
    )
    validate.add_argument(
        "--input",
        help="CSV of reviewed cases with a ground-truth label column. "
             "Omit with --synthetic to use the PHI-free fixture.",
    )
    validate.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in PHI-free synthetic validation set instead of --input.",
    )
    validate.add_argument("--out", default=None,
                          help="Optional path to write per-case scored results.")
    validate.add_argument("--text-col", default="note_text")
    validate.add_argument("--label-col", default="manual_label")
    validate.add_argument("--category-col", default="category")
    validate.add_argument("--id-col", default="id")
    validate.add_argument("--base-url", default=QWEN_BASE_URL)
    validate.add_argument("--model", default=QWEN_MODEL)
    validate.add_argument("--note-char-cap", type=int, default=3000)
    validate.add_argument("--max-tokens", type=int, default=400)
    validate.add_argument("--no-seed", action="store_true")
    validate.add_argument("--thinking", action="store_true")
    validate.add_argument(
        "--no-residential-rescue",
        action="store_true",
        help="Do not rescue nursing-home/home origins from transfer labels.",
    )

    args = parser.parse_args()

    if args.command == "score-csv":
        score_csv_command(args)
    elif args.command == "validate":
        validate_command(args)


def score_csv_command(args: argparse.Namespace) -> None:
    """Score a CSV without dropping rows, with checkpoint/resume support."""

    if not os.path.exists(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    n_rows = len(df)
    if args.text_col not in df.columns:
        candidates = [
            col for col in df.columns
            if "note" in col.lower() and "text" in col.lower()
        ]
        if not candidates:
            raise SystemExit(
                f"Text column '{args.text_col}' not found. Columns: {list(df.columns)}"
            )
        args.text_col = candidates[0]
        print(f"[note] using detected text column: '{args.text_col}'")

    df = annotate_audit_flags(df, args.text_col)

    done_idx = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if "llm_ran" in prev.columns and len(prev) == n_rows:
            df = annotate_audit_flags(prev, args.text_col)
            done_idx = set(df.index[df["llm_ran"].astype("object") == True])  # noqa: E712
            print(f"[resume] {len(done_idx)} rows already scored in {args.out}; continuing.")

    for col in (
        "llm_label",
        "llm_rationale",
        "llm_confidence",
        "steps_passed",
        "llm_ran",
        "llm_gated_reason",
        "llm_step_detail",
        "residential_rescue",
    ):
        if col not in df.columns:
            df[col] = None

    client = OpenAIChatClient(
        base_url=args.base_url,
        model=args.model,
        send_seed=not args.no_seed,
        req_pause=args.req_pause,
        disable_thinking=not args.thinking,
    )

    n_remaining = int((~df.index.isin(done_idx)).sum())
    n_to_do = min(n_remaining, args.limit) if args.limit is not None else n_remaining
    mode = f"SMOKE TEST (limit={args.limit})" if args.limit is not None else "full run"
    print(
        f"Rows: {n_rows} | remaining: {n_remaining} | to score now: {n_to_do} "
        f"[{mode}] | model: {args.model} | thinking: "
        f"{'ON' if args.thinking else 'OFF'} | note cap: {args.note_char_cap} chars"
    )

    scored = 0
    n_errors = 0
    start = time.time()
    for idx in df.index:
        if idx in done_idx:
            continue
        if args.limit is not None and scored >= args.limit:
            break

        text = df.at[idx, "clean_text"]
        if df.at[idx, "word_count"] == 0:
            _mark_gated(df, idx, LABEL_SKIP, "blank note")
        elif (
            not args.score_all_nonblank
            and df.at[idx, "word_count"] < MIN_NOTE_WORDS
            and not df.at[idx, "short_note_ohca_signal"]
        ):
            _mark_gated(df, idx, LABEL_SKIP, "short note without strong OHCA cue")
        elif not args.score_all_nonblank and not df.at[idx, "keyword_positive"]:
            _mark_gated(df, idx, LABEL_NOT_OHCA, "no cardiac-arrest keyword")
        else:
            result = classify_note_qwen(
                text,
                client,
                note_char_cap=args.note_char_cap,
                max_tokens=args.max_tokens,
                rescue_residential_origin=not args.no_residential_rescue,
                debug_raw=args.debug_raw,
                row_id=idx,
            )
            errored = result["llm_label"] == "Error"
            df.at[idx, "llm_label"] = result["llm_label"]
            df.at[idx, "llm_rationale"] = result["llm_rationale"]
            df.at[idx, "llm_confidence"] = result["llm_confidence"]
            df.at[idx, "steps_passed"] = result["steps_passed"]
            df.at[idx, "llm_ran"] = not errored
            df.at[idx, "llm_gated_reason"] = (
                "endpoint error (retry on resume)" if errored else ""
            )
            df.at[idx, "llm_step_detail"] = result.get("llm_step_detail", "")
            df.at[idx, "residential_rescue"] = result.get("residential_rescue", "")
            if errored:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  [error] row {idx}: {result['llm_rationale'][:160]}")
                if n_errors == 20:
                    print("  [error] 20+ endpoint errors; stopping early.", flush=True)
                    df.to_csv(args.out, index=False)
                    break

        scored += 1
        if scored % 50 == 0 or scored == n_to_do:
            elapsed_min = (time.time() - start) / 60
            rate = scored / max(elapsed_min, 1e-9)
            eta = (n_to_do - scored) / max(rate, 1e-9)
            print(f"  [{scored}/{n_to_do}] {elapsed_min:.1f}m elapsed, ~{eta:.1f}m left", flush=True)
            df.to_csv(args.out, index=False)

    ran = df["llm_ran"].astype("object") == True  # noqa: E712
    scored_mask = ran & (df["llm_label"].astype(str) != "Error")
    df["llm_predicted_ohca"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[scored_mask, "llm_predicted_ohca"] = (
        df.loc[scored_mask, "llm_label"] == "Yes"
    ).astype("Int64")

    assert len(df) == n_rows, "row count changed; scorer must never drop rows"
    df.to_csv(args.out, index=False)

    print("\nLabel distribution (advisory, scored rows only):")
    for label, count in sorted(Counter(df.loc[scored_mask, "llm_label"].astype(str)).items()):
        print(f"  {label:12s}: {count:6d}")
    if n_errors:
        print(f"  {'(errored)':12s}: {n_errors:6d}")
    n_unscored = int((~scored_mask).sum())
    if n_unscored:
        print(f"  {'(unscored)':12s}: {n_unscored:6d}")
    print(f"\nWrote {args.out}")


def validate_command(args: argparse.Namespace) -> None:
    """Validate the sequential Qwen classifier against a labeled set."""

    from .validation import run_validation, summarize, format_report

    if args.synthetic:
        from .validation_fixtures import load_synthetic_validation_set

        df = load_synthetic_validation_set()
        print(f"[validate] using synthetic PHI-free set ({len(df)} cases)")
    else:
        if not args.input:
            raise SystemExit("Provide --input CSV or pass --synthetic.")
        if not os.path.exists(args.input):
            raise SystemExit(f"Input CSV not found: {args.input}")
        df = pd.read_csv(args.input)
        print(f"[validate] loaded {len(df)} cases from {args.input}")

    if args.label_col not in df.columns:
        raise SystemExit(
            f"Ground-truth column '{args.label_col}' not found. "
            f"Columns: {list(df.columns)}"
        )

    client = OpenAIChatClient(
        base_url=args.base_url,
        model=args.model,
        send_seed=not args.no_seed,
        disable_thinking=not args.thinking,
    )

    scored = run_validation(
        df,
        client,
        text_col=args.text_col,
        label_col=args.label_col,
        category_col=args.category_col,
        id_col=args.id_col,
        note_char_cap=args.note_char_cap,
        max_tokens=args.max_tokens,
        rescue_residential_origin=not args.no_residential_rescue,
    )

    summary = summarize(
        scored,
        label_col=args.label_col,
        category_col=args.category_col,
    )
    print("\n" + format_report(summary))

    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"\nWrote per-case results to {args.out}")


def annotate_audit_flags(frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
    frame = frame.copy()
    clean = frame[text_col].apply(clean_text)
    frame["clean_text"] = clean
    frame["word_count"] = clean.apply(lambda text: len(text.split()))
    frame["keyword_positive"] = clean.apply(keyword_gate)
    frame["short_note_ohca_signal"] = clean.apply(short_note_has_ohca_signal)
    frame["pre_trauma_flag"] = clean.apply(trauma_shortcut)
    frame["pre_transfer_flag"] = clean.apply(transfer_shortcut)
    frame["pre_pmh_flag"] = clean.apply(pmh_shortcut)
    return frame


def _mark_gated(df: pd.DataFrame, idx: object, label: str, reason: str) -> None:
    df.at[idx, "llm_label"] = label
    df.at[idx, "llm_ran"] = False
    df.at[idx, "llm_gated_reason"] = reason


if __name__ == "__main__":
    main()
