"""
scripts/eval_test.py — Run inference on a JSONL test set and evaluate results.

Reads the JSONL produced by build_test.py, runs the XLM-R model on each entry,
compares predicted boundaries to gold, and prints a sentence-by-sentence diff.

Legend in the diff:
  ✓  — correct boundary (predicted matches gold)
  ✗  — missed boundary (gold sentence not found)
  +  — false positive (extra split the model invented)

Usage:
    # Standard eval:
    python scripts/eval_test.py \\
        --input data/hard_test.jsonl \\
        --checkpoint checkpoints/best_xlmr_model.pt

    # Sweep multiple thresholds to find the best one:
    python scripts/eval_test.py \\
        --input data/hard_test.jsonl \\
        --checkpoint checkpoints/best_xlmr_model.pt \\
        --threshold 0.3 0.4 0.5 0.6 0.7

    # Use character tolerance (±N chars counts as correct):
    python scripts/eval_test.py \\
        --input data/hard_test.jsonl \\
        --checkpoint checkpoints/best_xlmr_model.pt \\
        --tolerance 2

    # Preview gold sentences without running the model:
    python scripts/eval_test.py \\
        --input data/hard_test.jsonl \\
        --dry-run

    # Save results to JSON for further analysis:
    python scripts/eval_test.py \\
        --input data/hard_test.jsonl \\
        --checkpoint checkpoints/best_xlmr_model.pt \\
        --output data/hard_test_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import boundary_f1, sentence_exact_match
from src.inference import predict_xlmr, probs_to_boundaries
from src.model import XLMRSentenceSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def boundaries_to_sentences(text: str, boundaries: List[int]) -> List[str]:
    """Reconstruct sentence strings from boundary char positions."""
    sentences: List[str] = []
    prev = 0
    for b in sorted(boundaries):
        sentences.append(text[prev : b + 1])
        prev = b + 1
    if prev < len(text) and text[prev:].strip():
        sentences.append(text[prev:])
    return sentences


def is_match(b: int, reference: Set[int], tolerance: int) -> bool:
    """Return True if boundary b is within ±tolerance of any boundary in reference."""
    return any(abs(b - r) <= tolerance for r in reference)


# ---------------------------------------------------------------------------
# Pretty diff
# ---------------------------------------------------------------------------


def print_diff(
    entry_id: str,
    text: str,
    gold_boundaries: List[int],
    pred_boundaries: List[int],
    tolerance: int = 0,
) -> None:
    """Print a sentence-by-sentence diff of gold vs predicted boundaries."""
    gold_set = set(gold_boundaries)
    pred_set = set(pred_boundaries)

    gold_sents = boundaries_to_sentences(text, gold_boundaries)
    pred_sents = boundaries_to_sentences(text, pred_boundaries)

    print(f"\n{'─' * 78}")
    print(f"  {entry_id}  —  gold: {len(gold_sents)} sentences  |  pred: {len(pred_sents)} sentences")
    print(f"{'─' * 78}")

    # Gold sentences with hit/miss marker
    for i, (sent, b) in enumerate(zip(gold_sents, gold_boundaries), 1):
        hit = is_match(b, pred_set, tolerance)
        marker = "✓" if hit else "✗"
        preview = sent.replace("\n", "↵")
        # Truncate long sentences but keep it readable
        if len(preview) > 68:
            preview = preview[:65] + "..."
        print(f"  [{i:2d}] {marker}  {preview!r}")

    # False positives — pred boundaries not near any gold
    fps = [b for b in pred_boundaries if not is_match(b, gold_set, tolerance)]
    if fps:
        print(f"\n  False positives (+{len(fps)}):")
        pred_boundary_to_sent = {b: s for b, s in zip(pred_boundaries, pred_sents)}
        for b in fps:
            sent = pred_boundary_to_sent.get(b, "?")
            preview = sent.replace("\n", "↵")
            if len(preview) > 65:
                preview = preview[:62] + "..."
            print(f"         +  {preview!r}")


# ---------------------------------------------------------------------------
# Per-entry evaluation
# ---------------------------------------------------------------------------


def evaluate_entry(
    entry: dict,
    model: Optional["XLMRSentenceSplitter"],
    tokenizer,
    threshold: float,
    device: torch.device,
    tolerance: int,
) -> Dict[str, float]:
    """
    Run inference and evaluation on one JSONL entry.

    Returns a metrics dict, or an empty dict in dry-run mode.
    """
    text: str       = entry["text"]
    gold: List[int] = entry["gold_boundaries"]

    # Dry-run: just show gold
    if model is None:
        gold_sents = boundaries_to_sentences(text, gold)
        print(f"\n{'─' * 78}")
        print(f"  {entry['id']}  (gold only — no model)")
        print(f"{'─' * 78}")
        for i, s in enumerate(gold_sents, 1):
            preview = s.replace("\n", "↵")
            if len(preview) > 68:
                preview = preview[:65] + "..."
            print(f"  [{i:2d}]  {preview!r}")
        return {}

    char_probs = predict_xlmr(model, tokenizer, text, device=device)
    pred = probs_to_boundaries(char_probs, threshold=threshold, text=text)

    print_diff(entry["id"], text, gold, pred, tolerance=tolerance)

    metrics = boundary_f1(set(pred), set(gold), tolerance=tolerance)
    metrics["exact_match"] = sentence_exact_match(pred, gold, text)
    metrics["n_gold"]      = float(len(gold))
    metrics["n_pred"]      = float(len(pred))

    p, r, f = metrics["precision"], metrics["recall"], metrics["f1"]
    print(
        f"\n  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
        f"ExactMatch={metrics['exact_match']:.3f}  "
        f"tol={tolerance}  thresh={threshold}"
    )

    return metrics


# ---------------------------------------------------------------------------
# Summary table across entries
# ---------------------------------------------------------------------------


def print_summary(per_entry: List[Dict], threshold: float) -> None:
    """Print a macro-average summary row."""
    if not per_entry:
        return
    f1s  = [m["f1"]          for m in per_entry]
    ps   = [m["precision"]   for m in per_entry]
    rs   = [m["recall"]      for m in per_entry]
    exms = [m["exact_match"] for m in per_entry]

    print(f"\n{'═' * 78}")
    print(
        f"  MACRO AVG  (threshold={threshold})"
        f"  P={sum(ps)/len(ps):.3f}"
        f"  R={sum(rs)/len(rs):.3f}"
        f"  F1={sum(f1s)/len(f1s):.3f}"
        f"  ExM={sum(exms)/len(exms):.3f}"
        f"  ({len(per_entry)} entr{'y' if len(per_entry)==1 else 'ies'})"
    )
    print(f"{'═' * 78}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate XLM-R sentence splitter on a JSONL test set."
    )
    parser.add_argument(
        "--input", required=True, metavar="FILE",
        help="JSONL file produced by build_test.py.",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="FILE",
        help="Path to model checkpoint (.pt). Required unless --dry-run.",
    )
    parser.add_argument(
        "--model-name", default="xlm-roberta-large",
        help="HuggingFace model name matching the checkpoint.",
    )
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=[0.5], metavar="T",
        help="Decision threshold(s). Pass multiple to sweep: --threshold 0.3 0.5 0.7",
    )
    parser.add_argument(
        "--tolerance", type=int, default=0, metavar="N",
        help="Char tolerance for boundary matching (0 = exact match).",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Save full results to this JSON file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print gold sentences only, skip model inference.",
    )
    args = parser.parse_args()

    # Load JSONL
    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        sys.exit(1)

    entries = [
        json.loads(line)
        for line in in_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info(f"Loaded {len(entries)} entr{'y' if len(entries)==1 else 'ies'} from {in_path}")

    # Dry-run: show gold and exit
    if args.dry_run:
        for entry in entries:
            evaluate_entry(entry, None, None, 0.5, torch.device("cpu"), args.tolerance)
        return

    if args.checkpoint is None:
        logger.error("--checkpoint is required unless --dry-run is set.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = XLMRSentenceSplitter.load_checkpoint(
        args.checkpoint, model_name=args.model_name, map_location=str(device)
    )
    model = model.to(device).eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_results: Dict[str, List[Dict]] = {}

    for threshold in args.threshold:
        print(f"\n{'═' * 78}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'═' * 78}")

        per_entry = []
        for entry in entries:
            m = evaluate_entry(entry, model, tokenizer, threshold, device, args.tolerance)
            if m:
                m["id"]        = entry["id"]
                m["threshold"] = threshold
                per_entry.append(m)

        print_summary(per_entry, threshold)
        all_results[str(threshold)] = per_entry

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
