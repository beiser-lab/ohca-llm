"""
Evaluation utilities for the OHCA LLM pipeline.

Functions
---------
evaluate(df, true_col, pred_col)  : Compute sensitivity, specificity, PPV, NPV, F1.
confusion_matrix_text(df, ...)    : Print a text confusion matrix.
false_negatives(df, ...)          : Return the false-negative notes for review.
false_positives(df, ...)          : Return the false-positive notes for review.
"""

import pandas as pd
import numpy as np
from typing import Optional


def evaluate(
    df: pd.DataFrame,
    true_col: str = "manual_label",
    pred_col: str = "predicted_ohca",
    label_col: str = "final_label",
) -> dict:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    df        : DataFrame with ground-truth and prediction columns.
    true_col  : Column with manual labels (1 = OHCA, 0 = Not OHCA).
    pred_col  : Column with binary predictions (1 = OHCA, 0 = Not OHCA).
    label_col : Column with string labels (used for breakdown by label type).

    Returns
    -------
    dict with keys: TP, FP, TN, FN, sensitivity, specificity, PPV, NPV, F1, AUC
    """
    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv         = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv         = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    f1          = (2 * sensitivity * ppv / (sensitivity + ppv)
                   if (sensitivity + ppv) > 0 else 0.0)

    metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "PPV":         round(ppv, 4),
        "NPV":         round(npv, 4),
        "F1":          round(f1, 4),
        "n_total":     len(df),
        "n_ohca":      int(y_true.sum()),
        "n_pred_ohca": int(y_pred.sum()),
    }

    print("\n── OHCA Pipeline Evaluation ─────────────────────────────")
    print(f"  Notes evaluated : {metrics['n_total']:,}")
    print(f"  True OHCA       : {metrics['n_ohca']}")
    print(f"  Predicted OHCA  : {metrics['n_pred_ohca']}")
    print(f"\n  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"\n  Sensitivity : {sensitivity:.3f}  (recall)")
    print(f"  Specificity : {specificity:.3f}")
    print(f"  PPV         : {ppv:.3f}  (precision)")
    print(f"  NPV         : {npv:.3f}")
    print(f"  F1          : {f1:.3f}")
    print("─────────────────────────────────────────────────────────\n")

    return metrics


def false_negatives(
    df: pd.DataFrame,
    true_col: str = "manual_label",
    pred_col: str = "predicted_ohca",
    text_col: str = "clean_text",
    rationale_col: str = "llm_rationale",
) -> pd.DataFrame:
    """Return rows where true=OHCA but predicted=Not OHCA (missed cases)."""
    mask = (df[true_col].astype(int) == 1) & (df[pred_col].astype(int) == 0)
    cols = [c for c in [text_col, rationale_col, "pre_filter_label", "step_responses"]
            if c in df.columns]
    return df.loc[mask, cols].copy()


def false_positives(
    df: pd.DataFrame,
    true_col: str = "manual_label",
    pred_col: str = "predicted_ohca",
    text_col: str = "clean_text",
    rationale_col: str = "llm_rationale",
) -> pd.DataFrame:
    """Return rows where true=Not OHCA but predicted=OHCA (false alarms)."""
    mask = (df[true_col].astype(int) == 0) & (df[pred_col].astype(int) == 1)
    cols = [c for c in [text_col, rationale_col, "pre_filter_label", "step_responses"]
            if c in df.columns]
    return df.loc[mask, cols].copy()
