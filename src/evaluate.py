"""
src/evaluate.py — Evaluation metrics for sentence boundary detection.

Primary metric: boundary-level F1
  - Precision: what fraction of predicted boundaries are correct?
  - Recall: what fraction of gold boundaries were found?
  - F1: harmonic mean

Secondary metric: sentence exact match
  - What fraction of predicted sentences exactly match a gold sentence?
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .data import parse_sent_split_file, detect_lang

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def boundary_f1(
    pred: Set[int],
    gold: Set[int],
    tolerance: int = 0,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for boundary detection.

    Args:
        pred:      Set of predicted boundary character positions.
        gold:      Set of gold boundary character positions.
        tolerance: How many characters off still counts as correct (default 0 = exact).

    Returns:
        Dict with keys: precision, recall, f1, tp, fp, fn.
    """
    if tolerance == 0:
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
    else:
        gold_list = sorted(gold)
        tp = sum(
            1 for p in pred
            if any(abs(p - g) <= tolerance for g in gold_list)
        )
        fp = len(pred) - tp
        fn = sum(
            1 for g in gold_list
            if not any(abs(p - g) <= tolerance for p in pred)
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        float(tp),
        "fp":        float(fp),
        "fn":        float(fn),
    }


def sentence_exact_match(
    pred_boundaries: List[int],
    gold_boundaries: List[int],
    text: str,
) -> float:
    """
    Compute the fraction of predicted sentences that exactly match a gold sentence.

    A "sentence" is defined as the text between two consecutive boundary positions.
    """
    def to_sentences(boundaries: List[int], text: str) -> List[str]:
        sents = []
        prev = 0
        for b in sorted(boundaries):
            sents.append(text[prev : b + 1])
            prev = b + 1
        if prev < len(text):
            sents.append(text[prev:])
        return sents

    pred_sents = to_sentences(pred_boundaries, text)
    gold_set   = set(to_sentences(gold_boundaries, text))

    if not pred_sents:
        return 0.0

    matches = sum(1 for s in pred_sents if s in gold_set)
    return matches / len(pred_sents)


# ---------------------------------------------------------------------------
# File-level evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions_on_file(
    predicted_boundaries: List[int],
    gold_filepath: Path,
) -> Dict[str, float]:
    """
    Evaluate predicted boundaries against a gold .sent_split file.

    Args:
        predicted_boundaries: List of predicted boundary char positions.
        gold_filepath:        Path to the gold .sent_split file.

    Returns:
        Dict with precision, recall, f1, exact_match, and dataset name.
    """
    text, gold_boundaries = parse_sent_split_file(gold_filepath)

    metrics = boundary_f1(set(predicted_boundaries), set(gold_boundaries))
    metrics["exact_match"] = sentence_exact_match(
        predicted_boundaries, gold_boundaries, text
    )
    metrics["dataset"] = gold_filepath.parent.name
    metrics["file"]    = gold_filepath.name
    metrics["n_gold"]  = float(len(gold_boundaries))
    metrics["n_pred"]  = float(len(predicted_boundaries))

    return metrics


# ---------------------------------------------------------------------------
# Multi-dataset evaluation report
# ---------------------------------------------------------------------------


def format_report(results: List[Dict[str, float]], title: str = "Evaluation") -> str:
    """
    Format a list of per-file evaluation results into a readable table.
    """
    header = (
        f"\n{'=' * 78}\n"
        f"  {title}\n"
        f"{'=' * 78}\n"
        f"  {'Dataset':<35} {'P':>6} {'R':>6} {'F1':>6} {'ExM':>6} {'#Gold':>7}\n"
        f"  {'-' * 70}\n"
    )
    rows = []
    for r in results:
        name = f"{r.get('dataset','?')}/{r.get('file','?')}"[:35]
        rows.append(
            f"  {name:<35} "
            f"{r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} "
            f"{r.get('exact_match', 0.0):>6.3f} "
            f"{int(r.get('n_gold', 0)):>7}"
        )

    # Macro average
    f1s = [r["f1"] for r in results]
    ps  = [r["precision"] for r in results]
    rs  = [r["recall"] for r in results]
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_p  = sum(ps)  / len(ps)  if ps  else 0.0
    avg_r  = sum(rs)  / len(rs)  if rs  else 0.0

    footer = (
        f"  {'-' * 70}\n"
        f"  {'MACRO AVG':<35} "
        f"{avg_p:>6.3f} "
        f"{avg_r:>6.3f} "
        f"{avg_f1:>6.3f}\n"
        f"{'=' * 78}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def save_predictions(
    text: str,
    boundaries: List[int],
    output_path: Path,
) -> None:
    """
    Reconstruct .sent_split format: insert <EOS> after each boundary position.

    Args:
        text:        Clean text (no <EOS>).
        boundaries:  Predicted boundary char positions (sorted).
        output_path: Where to write the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    prev = 0
    for b in sorted(boundaries):
        parts.append(text[prev : b + 1])
        parts.append("<EOS>")
        prev = b + 1
    if prev < len(text):
        parts.append(text[prev:])

    output_path.write_text("".join(parts), encoding="utf-8")
