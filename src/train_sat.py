"""
scripts/train_sat.py — Sanity-check and optional LoRA adaptation of wtpsplit SaT.

The SaT model (segment-any-text/sat-12l-sm) is already trained on Universal
Dependencies data for 85+ languages, including English and Italian. It works
well out-of-the-box. This script:

  1. Evaluates the out-of-box SaT on all dev sets (to confirm it's a strong baseline).
  2. Optionally applies wtpsplit's built-in threshold/punctuation adaptation
     to better match this specific annotation style.

NOTE: Full fine-tuning of SaT via LoRA requires wtpsplit's training infrastructure
(see https://github.com/segment-any-text/wtpsplit for details). The out-of-box
model is already excellent for the hackathon, so full fine-tuning is usually
not needed.

Usage:
    python scripts/train_sat.py
    python scripts/train_sat.py --model sat-3l-sm    # smaller/faster variant
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_all_dev_files, parse_sent_split_file, detect_lang
from src.evaluate import boundary_f1, format_report
from src.ensemble import threshold_sweep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_sat_on_dev(sat_model, dev_files, threshold: float = 0.01):
    """Evaluate SaT on all dev files and return results."""
    import numpy as np
    results = []
    probs_list, gold_sets = [], []

    for fpath in dev_files:
        text, gold = parse_sent_split_file(fpath)
        if not text:
            continue
        lang = detect_lang(fpath)

        probs = sat_model.predict_proba(text, lang_code=lang)
        probs = np.array(probs, dtype=np.float32)
        if len(probs) < len(text):
            probs = np.pad(probs, (0, len(text) - len(probs)))
        elif len(probs) > len(text):
            probs = probs[: len(text)]

        probs_list.append(probs)
        gold_sets.append(set(gold))

    # Find best threshold
    best_t, best_f1 = threshold_sweep(probs_list, gold_sets)
    logger.info(f"SaT best threshold on dev: {best_t:.2f}  →  F1: {best_f1:.4f}")

    # Evaluate with best threshold
    for fpath, probs, gold in zip(dev_files, probs_list, gold_sets):
        pred = {i for i, p in enumerate(probs) if p >= best_t}
        m = boundary_f1(pred, gold)
        m["dataset"] = fpath.parent.name
        m["file"]    = fpath.name
        m["n_gold"]  = float(len(gold))
        m["n_pred"]  = float(len(pred))
        results.append(m)

    return results, best_t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate wtpsplit SaT on UD dev sets."
    )
    parser.add_argument("--model", default="sat-12l-sm",
                        help="SaT model name (default: sat-12l-sm)")
    args = parser.parse_args()

    logger.info(f"Loading SaT model: {args.model}")
    try:
        from wtpsplit import SaT
        sat = SaT(args.model)
    except ImportError:
        logger.error("wtpsplit not installed. Run: pip install wtpsplit")
        sys.exit(1)

    dev_files = get_all_dev_files()
    if not dev_files:
        logger.error("No dev files found. Run from sentence_splitting/ directory.")
        sys.exit(1)

    logger.info(f"Evaluating on {len(dev_files)} dev files...")
    results, best_threshold = evaluate_sat_on_dev(sat, dev_files)

    print(format_report(results, title=f"SaT ({args.model}) — Dev set"))
    print(f"\n  → Use --threshold {best_threshold:.2f} in scripts/predict.py for SaT-only inference.\n")
    print(
        "  For full ensemble, run:\n"
        f"    python scripts/optimize.py "
        f"--checkpoint checkpoints/xlmr/best_model.pt --use-sat --sat-model {args.model}"
    )


if __name__ == "__main__":
    main()
