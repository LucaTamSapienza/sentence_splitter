"""
src/ensemble.py — Ensemble predictor combining XLM-R, SaT, and rule-based components.

Ensemble strategy: weighted average of per-character probabilities.
  final_prob[i] = (w_xlmr * xlmr_prob[i] + w_sat * sat_prob[i] + w_rules * rule_prob[i])
                / (w_xlmr + w_sat + w_rules)

Weights are optimized on the combined dev set via scripts/optimize.py.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from transformers import AutoTokenizer

from .data import parse_sent_split_file, detect_lang
from .evaluate import boundary_f1
from .inference import predict_xlmr, predict_sat, probs_to_boundaries
from .rules import rule_based_probs

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines XLM-R, optional SaT, and rule-based components into one predictor.

    Usage:
        predictor = EnsemblePredictor(model, tokenizer, sat_model)
        char_probs = predictor.predict(text, lang="en")
        boundaries = probs_to_boundaries(char_probs, threshold=0.5)
    """

    def __init__(
        self,
        xlmr_model,
        tokenizer: AutoTokenizer,
        sat_model=None,
        xlmr_weight: float = 0.60,
        sat_weight: float = 0.35,
        rules_weight: float = 0.05,
        device: Optional[torch.device] = None,
    ) -> None:
        self.xlmr_model   = xlmr_model
        self.tokenizer    = tokenizer
        self.sat_model    = sat_model
        self.xlmr_weight  = xlmr_weight
        self.sat_weight   = sat_weight if sat_model is not None else 0.0
        self.rules_weight = rules_weight
        self.device       = device or (
            next(xlmr_model.parameters()).device
            if xlmr_model is not None
            else torch.device("cpu")
        )

    def predict(self, text: str, lang: str = "en") -> np.ndarray:
        """
        Predict per-character boundary probabilities.

        Returns np.ndarray of shape (len(text),) with values in [0, 1].
        """
        total_w = self.xlmr_weight + self.sat_weight + self.rules_weight
        if total_w == 0:
            raise ValueError("All ensemble weights are zero.")

        probs = np.zeros(len(text), dtype=np.float32)

        if self.xlmr_weight > 0 and self.xlmr_model is not None:
            p = predict_xlmr(self.xlmr_model, self.tokenizer, text, device=self.device)
            probs += p * self.xlmr_weight

        if self.sat_weight > 0 and self.sat_model is not None:
            p = predict_sat(self.sat_model, text, lang_code=lang)
            probs += p * self.sat_weight

        if self.rules_weight > 0:
            rule_scores = rule_based_probs(text, lang=lang)
            rule_probs  = np.clip((rule_scores + 0.5) / 1.5, 0.0, 1.0)
            probs      += rule_probs.astype(np.float32) * self.rules_weight

        probs /= total_w
        return probs

    def set_weights(
        self,
        xlmr: float,
        sat: float,
        rules: float,
    ) -> None:
        """Update ensemble weights (e.g., after optimisation)."""
        self.xlmr_weight  = xlmr
        self.sat_weight   = sat if self.sat_model is not None else 0.0
        self.rules_weight = rules


# ---------------------------------------------------------------------------
# Weight and threshold optimisation (run on dev set)
# ---------------------------------------------------------------------------


def optimise_ensemble(
    dev_files: List[Path],
    xlmr_model,
    tokenizer: AutoTokenizer,
    sat_model=None,
    device: Optional[torch.device] = None,
    threshold_grid: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Grid-search ensemble weights and threshold to maximise dev F1.

    Strategy:
      1. Pre-compute per-model char_probs for all dev files (expensive, done once).
      2. Use scipy.optimize.minimize to find optimal (w_xlmr, w_sat, w_rules).
      3. Grid-search the final threshold in [0.1, 0.9].

    Returns:
        best_weights: Dict with keys xlmr, sat, rules (normalised to sum=1).
        best_threshold: Float in (0, 1).
    """
    if threshold_grid is None:
        threshold_grid = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]

    logger.info("Pre-computing per-model predictions on dev files...")

    # Pre-compute probabilities (one pass per model per file)
    xlmr_probs_list: List[np.ndarray] = []
    sat_probs_list:  List[np.ndarray] = []
    rule_probs_list: List[np.ndarray] = []
    gold_sets:       List[set]        = []

    for fpath in dev_files:
        text, gold = parse_sent_split_file(fpath)
        if not text:
            continue

        lang = detect_lang(fpath)
        gold_sets.append(set(gold))

        if xlmr_model is not None:
            xlmr_probs_list.append(
                predict_xlmr(xlmr_model, tokenizer, text, device=device)
            )
        else:
            xlmr_probs_list.append(np.zeros(len(text), dtype=np.float32))

        if sat_model is not None:
            sat_probs_list.append(predict_sat(sat_model, text, lang_code=lang))
        else:
            sat_probs_list.append(np.zeros(len(text), dtype=np.float32))

        rule_scores = rule_based_probs(text, lang=lang)
        rule_probs_list.append(
            np.clip((rule_scores + 0.5) / 1.5, 0.0, 1.0).astype(np.float32)
        )

    logger.info(f"Pre-computed probs for {len(gold_sets)} dev files.")

    def _f1_for_weights_threshold(weights: np.ndarray, threshold: float) -> float:
        w_x, w_s, w_r = weights
        total = w_x + w_s + w_r
        if total == 0:
            return 0.0

        all_f1 = []
        for xp, sp, rp, gold in zip(
            xlmr_probs_list, sat_probs_list, rule_probs_list, gold_sets
        ):
            combined = (w_x * xp + w_s * sp + w_r * rp) / total
            pred = {i for i, p in enumerate(combined) if p >= threshold}
            all_f1.append(boundary_f1(pred, gold)["f1"])

        return sum(all_f1) / len(all_f1) if all_f1 else 0.0

    # Optimise weights via minimisation (we negate F1 since scipy minimises)
    best_f1 = -1.0
    best_weights = {"xlmr": 0.60, "sat": 0.35, "rules": 0.05}
    best_threshold = 0.5

    # Initial guess
    x0 = np.array([0.60, 0.35, 0.05])

    for threshold in threshold_grid:
        def objective(w):
            w = np.clip(w, 0.0, 1.0)
            return -_f1_for_weights_threshold(w, threshold)

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 0.01, "fatol": 0.001},
        )

        f1 = -result.fun
        if f1 > best_f1:
            best_f1 = f1
            w = np.clip(result.x, 0.0, None)
            total = w.sum()
            best_weights = {
                "xlmr":  float(w[0] / total),
                "sat":   float(w[1] / total),
                "rules": float(w[2] / total),
            }
            best_threshold = threshold
            x0 = result.x  # warm-start next threshold search

    logger.info(
        f"Optimal ensemble — weights: {best_weights}, "
        f"threshold: {best_threshold:.2f}, dev F1: {best_f1:.4f}"
    )
    return best_weights, best_threshold


def threshold_sweep(
    char_probs_list: List[np.ndarray],
    gold_sets: List[set],
    threshold_grid: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Sweep thresholds and return the one that maximises macro-average F1.

    Useful for tuning a single model without ensemble weight search.

    Returns:
        (best_threshold, best_f1)
    """
    if threshold_grid is None:
        threshold_grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    best_f1 = -1.0
    best_threshold = 0.5

    for t in threshold_grid:
        f1s = []
        for probs, gold in zip(char_probs_list, gold_sets):
            pred = {i for i, p in enumerate(probs) if p >= t}
            f1s.append(boundary_f1(pred, gold)["f1"])
        avg = sum(f1s) / len(f1s) if f1s else 0.0
        if avg > best_f1:
            best_f1 = avg
            best_threshold = t

    return best_threshold, best_f1
