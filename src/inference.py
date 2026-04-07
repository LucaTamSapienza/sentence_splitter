"""
src/inference.py — Full inference pipeline for sentence boundary detection.

End-to-end flow:
  raw text
    → tokenize into sliding windows
    → batched model forward pass
    → aggregate overlapping window logits (average)
    → map per-subtoken logits → per-character probabilities
    → threshold → boundary positions
    → insert <EOS> markers
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import tokenize_document, detect_lang

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XLM-R inference
# ---------------------------------------------------------------------------


def predict_xlmr(
    model,
    tokenizer: AutoTokenizer,
    text: str,
    window_size: int = 510,
    stride: int = 256,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run XLM-R sentence boundary detection on a text.

    Returns:
        np.ndarray of shape (len(text),) with values in [0, 1].
        char_probs[i] = probability that a sentence boundary ends at character i.
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize (no labels at inference time)
    from .data import parse_sent_split_file  # only for type hint
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    token_ids: List[int]              = encoding["input_ids"]
    offsets:   List[Tuple[int, int]]  = encoding["offset_mapping"]

    if not token_ids:
        return np.zeros(len(text))

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    full_len = window_size + 2

    # Build windows (no labels needed)
    windows: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    N = len(token_ids)
    start = 0
    while start < N:
        end = min(start + window_size, N)
        ids  = [cls_id] + token_ids[start:end] + [sep_id]
        mask = [1] * len(ids)
        pad  = full_len - len(ids)
        ids  += [pad_id] * pad
        mask += [0]      * pad
        windows.append((
            torch.tensor(ids,  dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            start,
        ))
        if end == N:
            break
        start += stride

    # Batched inference + overlapping window aggregation
    tok_logits_sum = np.zeros(N, dtype=np.float64)
    tok_logits_cnt = np.zeros(N, dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            b_ids  = torch.stack([w[0] for w in batch]).to(device)
            b_mask = torch.stack([w[1] for w in batch]).to(device)

            logits, _ = model(b_ids, b_mask)   # (B, 512)
            logits_np  = logits.cpu().float().numpy()

            for j, (_, _, doc_start) in enumerate(batch):
                content_len = min(window_size, N - doc_start)
                # Positions 1 to 1+content_len are content tokens (skip CLS at 0)
                w_logits = logits_np[j, 1 : 1 + content_len]
                tok_logits_sum[doc_start : doc_start + content_len] += w_logits
                tok_logits_cnt[doc_start : doc_start + content_len] += 1

    # Average overlapping windows
    safe_cnt = np.maximum(tok_logits_cnt, 1)
    avg_logits = tok_logits_sum / safe_cnt
    tok_probs  = 1.0 / (1.0 + np.exp(-avg_logits))  # sigmoid

    # Map per-subtoken probabilities → per-character probabilities
    # Each subtoken covers a character span; assign its probability to the
    # LAST character of that span (where the sentence boundary naturally falls).
    char_probs = np.zeros(len(text), dtype=np.float32)
    for i, (s, e) in enumerate(offsets):
        if e > s and i < len(tok_probs):
            char_probs[e - 1] = max(char_probs[e - 1], float(tok_probs[i]))

    return char_probs


# ---------------------------------------------------------------------------
# wtpsplit / SaT inference
# ---------------------------------------------------------------------------


def predict_sat(
    sat_model,
    text: str,
    lang_code: Optional[str] = None,
) -> np.ndarray:
    """
    Run wtpsplit SaT inference on a text.

    Args:
        sat_model:  Loaded SaT model (from wtpsplit import SaT; sat = SaT("sat-12l-sm")).
        text:       Input text.
        lang_code:  ISO 639-1 language code ("en" or "it"). Optional but recommended.

    Returns:
        np.ndarray of shape (len(text),) with values in [0, 1].
    """
    kwargs = {}
    if lang_code is not None:
        kwargs["lang_code"] = lang_code

    probs = sat_model.predict_proba(text, **kwargs)
    probs_arr = np.array(probs, dtype=np.float32)

    # wtpsplit may return an array shorter than len(text) by 1 in some versions
    if len(probs_arr) < len(text):
        probs_arr = np.pad(probs_arr, (0, len(text) - len(probs_arr)))
    elif len(probs_arr) > len(text):
        probs_arr = probs_arr[: len(text)]

    return probs_arr


# ---------------------------------------------------------------------------
# Threshold + boundary extraction
# ---------------------------------------------------------------------------


def probs_to_boundaries(
    char_probs: np.ndarray,
    threshold: float = 0.5,
    text: Optional[str] = None,
) -> List[int]:
    """
    Convert per-character probabilities to a sorted list of boundary positions.

    Optionally enforces hard rules:
      - Blank lines (paragraph breaks) are always boundaries.
      - The last non-whitespace character of the document is always a boundary.

    Args:
        char_probs: Array of length len(text).
        threshold:  Decision threshold (tune on dev set).
        text:       Original text (if provided, applies hard blank-line rule).

    Returns:
        Sorted list of boundary character indices.
    """
    boundaries = sorted(
        int(i) for i, p in enumerate(char_probs) if p >= threshold
    )

    if text is not None:
        from .rules import get_hard_boundaries
        hard = get_hard_boundaries(text)
        boundaries = sorted(set(boundaries) | set(hard))

        # Ensure document end is a boundary
        last = len(text) - 1
        while last >= 0 and text[last] in " \t\n":
            last -= 1
        if last >= 0 and last not in set(boundaries):
            boundaries.append(last)
            boundaries.sort()

    return boundaries


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def insert_eos_markers(text: str, boundaries: List[int]) -> str:
    """
    Reconstruct the .sent_split format by inserting <EOS> after each boundary.

    Args:
        text:       Clean text without <EOS>.
        boundaries: Sorted list of boundary char positions.

    Returns:
        Text with <EOS> inserted after each boundary position.
    """
    parts: List[str] = []
    prev = 0
    for b in sorted(boundaries):
        if b >= len(text):
            break
        parts.append(text[prev : b + 1])
        parts.append("<EOS>")
        prev = b + 1
    if prev < len(text):
        parts.append(text[prev:])
    return "".join(parts)


def predict_and_write(
    model,
    tokenizer: AutoTokenizer,
    input_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    sat_model=None,
    sat_weight: float = 0.0,
    xlmr_weight: float = 1.0,
    rules_weight: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[int]:
    """
    End-to-end: read input file → predict → write output with <EOS> markers.

    Returns predicted boundary positions.
    """
    text = input_path.read_text(encoding="utf-8")
    # Strip any existing <EOS> markers (eval mode)
    text = text.replace("<EOS>", "")

    lang = detect_lang(input_path)

    # XLM-R predictions
    xlmr_probs = predict_xlmr(model, tokenizer, text, device=device)

    # Optionally blend with SaT and rules
    char_probs = xlmr_probs * xlmr_weight

    if sat_model is not None and sat_weight > 0:
        sat_probs   = predict_sat(sat_model, text, lang_code=lang)
        char_probs += sat_probs * sat_weight

    if rules_weight > 0:
        from .rules import rule_based_probs
        rule_probs  = rule_based_probs(text, lang=lang)
        # Normalize rules to [0,1] range before blending
        rule_probs  = np.clip((rule_probs + 0.5) / 1.5, 0.0, 1.0)
        char_probs += rule_probs.astype(np.float32) * rules_weight

    total_weight = xlmr_weight + (sat_weight if sat_model else 0) + rules_weight
    if total_weight > 0:
        char_probs /= total_weight

    boundaries = probs_to_boundaries(char_probs, threshold=threshold, text=text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(insert_eos_markers(text, boundaries), encoding="utf-8")
    logger.info(f"Written {len(boundaries)} boundaries → {output_path}")

    return boundaries
