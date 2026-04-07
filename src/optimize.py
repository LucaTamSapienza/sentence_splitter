"""
scripts/optimize.py — Find optimal ensemble weights and decision threshold.

Run this AFTER training is complete (checkpoint must exist).
Sweeps the threshold and optionally optimises ensemble weights on the dev set.

Usage:
    # Tune threshold only (XLM-R only):
    python scripts/optimize.py --checkpoint checkpoints/xlmr/best_model.pt

    # Tune XLM-R + SaT ensemble weights and threshold:
    python scripts/optimize.py --checkpoint checkpoints/xlmr/best_model.pt --use-sat

    # Save optimal config back to configs/sat.yaml:
    python scripts/optimize.py --checkpoint checkpoints/xlmr/best_model.pt --use-sat --save
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_all_dev_files, parse_sent_split_file, detect_lang
from src.evaluate import boundary_f1, format_report
from src.inference import predict_xlmr, predict_sat
from src.ensemble import optimise_ensemble, threshold_sweep
from src.model import XLMRSentenceSplitter
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimise ensemble weights and decision threshold on dev set."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from training")
    parser.add_argument("--model-name", default="xlm-roberta-large",
                        help="HuggingFace model name (must match training)")
    parser.add_argument("--use-sat", action="store_true",
                        help="Include wtpsplit SaT model in ensemble")
    parser.add_argument("--sat-model", default="sat-12l-sm",
                        help="SaT model name (default: sat-12l-sm)")
    parser.add_argument("--save", action="store_true",
                        help="Write optimal settings back to configs/sat.yaml")
    parser.add_argument("--output", default="configs/optimal.json",
                        help="Where to save optimal settings as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load XLM-R -------------------------------------------------------
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = XLMRSentenceSplitter.load_checkpoint(
        args.checkpoint, model_name=args.model_name, map_location=str(device)
    )
    model = model.to(device).eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- Load SaT (optional) ----------------------------------------------
    sat_model = None
    if args.use_sat:
        logger.info(f"Loading SaT model: {args.sat_model}")
        try:
            from wtpsplit import SaT
            sat_model = SaT(args.sat_model)
            logger.info("SaT loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SaT: {e}. Continuing without it.")

    # ---- Dev files ---------------------------------------------------------
    dev_files = get_all_dev_files()
    if not dev_files:
        logger.error("No dev files found. Run from the sentence_splitting/ directory.")
        sys.exit(1)
    logger.info(f"Dev files: {len(dev_files)}")

    # ---- Optimise ----------------------------------------------------------
    if sat_model is not None:
        logger.info("Running ensemble optimisation (XLM-R + SaT + Rules)...")
        best_weights, best_threshold = optimise_ensemble(
            dev_files=dev_files,
            xlmr_model=model,
            tokenizer=tokenizer,
            sat_model=sat_model,
            device=device,
        )
    else:
        # XLM-R only — just sweep threshold
        logger.info("Running threshold sweep (XLM-R only)...")
        probs_list = []
        gold_sets  = []
        for fpath in dev_files:
            text, gold = parse_sent_split_file(fpath)
            if not text:
                continue
            probs_list.append(predict_xlmr(model, tokenizer, text, device=device))
            gold_sets.append(set(gold))

        best_threshold, best_f1 = threshold_sweep(probs_list, gold_sets)
        best_weights = {"xlmr": 1.0, "sat": 0.0, "rules": 0.0}
        logger.info(
            f"Best threshold: {best_threshold:.2f}  →  dev F1: {best_f1:.4f}"
        )

    # ---- Report on dev with optimal settings ------------------------------
    logger.info("\nFinal evaluation on dev set with optimal settings:")
    results = []
    for fpath in dev_files:
        text, gold = parse_sent_split_file(fpath)
        if not text:
            continue
        lang = detect_lang(fpath)

        xlmr_p = predict_xlmr(model, tokenizer, text, device=device)
        combined = xlmr_p * best_weights["xlmr"]

        if sat_model is not None and best_weights["sat"] > 0:
            sat_p     = predict_sat(sat_model, text, lang_code=lang)
            combined += sat_p * best_weights["sat"]

        if best_weights.get("rules", 0) > 0:
            from src.rules import rule_based_probs
            rule_p    = np.clip(
                (rule_based_probs(text, lang=lang) + 0.5) / 1.5, 0.0, 1.0
            )
            combined += rule_p.astype(np.float32) * best_weights["rules"]

        total_w = sum(best_weights.values())
        combined /= total_w

        pred = {i for i, p in enumerate(combined) if p >= best_threshold}
        m = boundary_f1(pred, set(gold))
        m["dataset"] = fpath.parent.name
        m["file"]    = fpath.name
        m["n_gold"]  = float(len(gold))
        m["n_pred"]  = float(len(pred))
        results.append(m)

    print(format_report(results, title="Dev set — optimal ensemble"))

    # ---- Save settings ----------------------------------------------------
    optimal = {
        "weights": best_weights,
        "threshold": best_threshold,
        "checkpoint": str(args.checkpoint),
        "sat_model": args.sat_model if sat_model else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(optimal, f, indent=2)
    logger.info(f"Saved optimal settings → {args.output}")


if __name__ == "__main__":
    main()
