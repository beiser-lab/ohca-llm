"""Validation harness for the sequential Qwen OHCA classifier.

Runs the production classifier (``classify_note_qwen``) over a labeled set and
reports sensitivity/specificity/PPV/NPV/F1 plus a per-category breakdown so
false positives in specific hard-negative subgroups (respiratory failure,
intubation, altered mental status, nursing-home/SNF/ALF) are visible rather
than buried in an aggregate specificity number.

Intended use (in the HIPAA-compliant environment): assemble a reviewed CSV of
real cases with a ground-truth column and a category column, then run

    ohca-llm validate --input reviewed.csv --base-url ... --model ...

before any production rerun. A PHI-free synthetic set is available via
``--synthetic`` (or ``load_synthetic_validation_set``) to smoke-test the harness
and exercise the Step 1 guardrail without touching patient data.

This module takes a client object (duck-typed ``.complete(...)``), so it can be
driven by a real ``OpenAIChatClient`` or a fake client in tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .qwen_classifier import classify_note_qwen


REQUIRED_COLUMNS = ("note_text", "manual_label")


def run_validation(
    df: pd.DataFrame,
    client,
    *,
    text_col: str = "note_text",
    label_col: str = "manual_label",
    category_col: Optional[str] = "category",
    id_col: Optional[str] = "id",
    note_char_cap: int = 3000,
    max_tokens: int = 400,
    rescue_residential_origin: bool = True,
) -> pd.DataFrame:
    """Score every row with the sequential Qwen classifier.

    Returns a copy of ``df`` with the classifier's output columns appended,
    including a binary ``predicted_ohca`` (1 if ``llm_label == "Yes"``, else 0)
    and a ``correct`` flag against ``label_col``. Rows are never dropped.
    """

    if text_col not in df.columns:
        raise KeyError(f"text column {text_col!r} not found; have {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not found; have {list(df.columns)}")

    out = df.copy().reset_index(drop=True)
    records: List[Dict] = []
    for _, row in out.iterrows():
        result = classify_note_qwen(
            row[text_col],
            client,
            note_char_cap=note_char_cap,
            max_tokens=max_tokens,
            rescue_residential_origin=rescue_residential_origin,
            row_id=row[id_col] if id_col and id_col in out.columns else None,
        )
        records.append(result)

    res = pd.DataFrame.from_records(records)
    for col in ("llm_label", "llm_confidence", "llm_rationale",
                "steps_passed", "llm_step_detail", "residential_rescue"):
        out[col] = res[col].values

    out["predicted_ohca"] = (out["llm_label"] == "Yes").astype(int)
    out["correct"] = out["predicted_ohca"].eq(out[label_col].astype(int))
    return out


def summarize(
    scored: pd.DataFrame,
    *,
    label_col: str = "manual_label",
    pred_col: str = "predicted_ohca",
    category_col: Optional[str] = "category",
) -> Dict:
    """Compute overall metrics and per-category error breakdown.

    Returns a dict with an ``overall`` metrics block and, when a category
    column is present, a ``by_category`` mapping. Hard-negative categories are
    surfaced with their false-positive counts because a single FP there is the
    exact failure mode this harness is meant to catch.
    """

    y_true = scored[label_col].astype(int)
    y_pred = scored[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def _safe(n, d):
        return round(n / d, 4) if d else None

    overall = {
        "n": len(scored),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "PPV": _safe(tp, tp + fp),
        "NPV": _safe(tn, tn + fn),
        "F1": _safe(2 * tp, 2 * tp + fp + fn),
    }

    result: Dict = {"overall": overall}

    if category_col and category_col in scored.columns:
        by_cat = {}
        for cat, grp in scored.groupby(category_col):
            gt = grp[label_col].astype(int)
            gp = grp[pred_col].astype(int)
            by_cat[str(cat)] = {
                "n": len(grp),
                "n_true_ohca": int(gt.sum()),
                "false_positives": int(((gt == 0) & (gp == 1)).sum()),
                "false_negatives": int(((gt == 1) & (gp == 0)).sum()),
                "accuracy": _safe(int((gt == gp).sum()), len(grp)),
            }
        result["by_category"] = by_cat

    return result


def format_report(summary: Dict) -> str:
    """Render a plain-text validation report from ``summarize`` output."""

    o = summary["overall"]
    lines = [
        "── OHCA Sequential-Qwen Validation ──────────────────────",
        f"  Cases          : {o['n']}",
        f"  TP={o['TP']}  FP={o['FP']}  TN={o['TN']}  FN={o['FN']}",
        f"  Sensitivity    : {o['sensitivity']}",
        f"  Specificity    : {o['specificity']}",
        f"  PPV            : {o['PPV']}",
        f"  NPV            : {o['NPV']}",
        f"  F1             : {o['F1']}",
    ]
    if "by_category" in summary:
        lines.append("")
        lines.append("  By category (FP = false positive, FN = false negative):")
        for cat in sorted(summary["by_category"]):
            c = summary["by_category"][cat]
            flag = "  <-- FP present" if c["false_positives"] else ""
            lines.append(
                f"    {cat:28s} n={c['n']:3d}  FP={c['false_positives']}  "
                f"FN={c['false_negatives']}  acc={c['accuracy']}{flag}"
            )
    lines.append("─────────────────────────────────────────────────────────")
    return "\n".join(lines)
