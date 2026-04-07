"""
src/data.py — Data loading and preprocessing for sentence boundary detection.

Key concepts:
  - .sent_split format: raw text with <EOS> marking sentence ends
  - We REMOVE <EOS> and record the character index of the last char of each sentence
  - For XLM-R: tokenize → sliding windows of 512 subtokens (510 content + CLS + SEP)
  - Labels: -100 for CLS/SEP/padding, 0/1 for content tokens
"""

import logging
import random
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from .utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

EOS_TAG = "<EOS>"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_sent_split_file(filepath: Path) -> Tuple[str, List[int]]:
    """
    Parse a .sent_split file into (clean_text, boundary_positions).

    Removes all <EOS> markers and returns the character index (0-based) of the
    LAST CHARACTER of each sentence in the clean text.

    Example:
        "Hello.<EOS> World."
        → clean_text = "Hello. World."
        → boundary_positions = [5]      # index of '.'

    Args:
        filepath: Path to a .sent_split file.

    Returns:
        clean_text:         Text with <EOS> markers removed.
        boundary_positions: Sorted list of 0-based char indices where sentences end.
    """
    raw = Path(filepath).read_text(encoding="utf-8")
    parts = raw.split(EOS_TAG)

    clean_text = "".join(parts)

    boundary_positions: List[int] = []
    offset = 0
    for part in parts[:-1]:  # last part has no following <EOS>
        offset += len(part)
        if offset > 0:
            pos = offset - 1
            # Deduplicate (handles rare <EOS><EOS> sequences)
            if not boundary_positions or boundary_positions[-1] != pos:
                boundary_positions.append(pos)

    return clean_text, boundary_positions


# ---------------------------------------------------------------------------
# Tokenization + window creation
# ---------------------------------------------------------------------------


def tokenize_document(
    text: str,
    boundaries: List[int],
    tokenizer: AutoTokenizer,
    window_size: int = 510,
    stride: int = 256,
) -> Tuple[
    List[Dict[str, torch.Tensor]],
    List[int],
    List[Tuple[int, int]],
    List[int],
]:
    """
    Tokenize a document and create overlapping windows for training/inference.

    Args:
        text:        Clean document text (no <EOS>).
        boundaries:  Boundary character positions from parse_sent_split_file.
        tokenizer:   HuggingFace tokenizer (XLM-RoBERTa).
        window_size: Max content tokens per window (excluding CLS/SEP).
        stride:      Step size between consecutive windows (use stride < window_size for overlap).

    Returns:
        windows:       List of dicts (input_ids, attention_mask, labels, doc_token_start).
                       Each tensor has shape (window_size + 2,) = (512,).
        token_ids:     Full token IDs for the document (no special tokens).
        offsets:       Per-token (char_start, char_end) offsets in original text.
        token_labels:  Per-token binary labels (1 = sentence boundary ends here).
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    token_ids: List[int] = encoding["input_ids"]
    offsets: List[Tuple[int, int]] = encoding["offset_mapping"]

    if not token_ids:
        return [], [], [], []

    # Build char → subtoken index (O(len(text)))
    # For each boundary position b, we want the subtoken whose span covers b.
    char_to_tok: List[int] = [-1] * len(text)
    for tok_idx, (s, e) in enumerate(offsets):
        for c in range(s, min(e, len(text))):
            char_to_tok[c] = tok_idx

    # Assign per-token labels
    token_labels: List[int] = [0] * len(token_ids)
    for b in boundaries:
        if 0 <= b < len(char_to_tok) and char_to_tok[b] >= 0:
            token_labels[char_to_tok[b]] = 1

    # Create sliding windows
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    full_len = window_size + 2  # CLS + content + SEP

    windows: List[Dict[str, torch.Tensor]] = []
    N = len(token_ids)
    start = 0

    while start < N:
        end = min(start + window_size, N)

        ids  = [cls_id] + token_ids[start:end]   + [sep_id]
        labs = [-100]   + token_labels[start:end] + [-100]
        mask = [1]      * len(ids)

        # Pad to full_len
        pad = full_len - len(ids)
        ids  += [pad_id] * pad
        labs += [-100]   * pad
        mask += [0]      * pad

        windows.append({
            "input_ids":       torch.tensor(ids,  dtype=torch.long),
            "attention_mask":  torch.tensor(mask, dtype=torch.long),
            "labels":          torch.tensor(labs, dtype=torch.long),
            "doc_token_start": start,
        })

        if end == N:
            break
        start += stride

    return windows, token_ids, offsets, token_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentSplitDataset(Dataset):
    """
    PyTorch Dataset of sliding-window examples for sentence boundary detection.

    Each item: dict with keys input_ids, attention_mask, labels — all shape (512,).

    Args:
        tokenizer:   XLM-RoBERTa tokenizer.
        file_paths:  List of .sent_split files to load.
        window_size: Content tokens per window (default 510 → full window 512).
        stride:      Sliding stride (default 256 = 50% overlap).
        augment:     Apply light data augmentation (context truncation).
        seed:        RNG seed for augmentation.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        file_paths: List[Path],
        window_size: int = 510,
        stride: int = 256,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.windows: List[Dict[str, torch.Tensor]] = []
        rng = random.Random(seed)

        for fpath in file_paths:
            text, boundaries = parse_sent_split_file(fpath)
            if not text:
                continue

            if augment:
                text, boundaries = _augment(text, boundaries, rng)

            wins, _, _, _ = tokenize_document(
                text, boundaries, tokenizer, window_size, stride
            )
            self.windows.extend(wins)

        logger.info(
            f"SentSplitDataset: {len(self.windows)} windows "
            f"from {len(file_paths)} file(s)"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        return {
            "input_ids":      w["input_ids"],
            "attention_mask": w["attention_mask"],
            "labels":         w["labels"],
        }


def _augment(
    text: str, boundaries: List[int], rng: random.Random
) -> Tuple[str, List[int]]:
    """
    Light augmentation: randomly trim leading/trailing context so the model
    learns to handle document starts and ends robustly.
    """
    if len(text) < 200:
        return text, boundaries

    if rng.random() < 0.2:
        cut = rng.randint(1, min(100, len(text) // 4))
        text = text[cut:]
        boundaries = [b - cut for b in boundaries if b - cut >= 0]

    if rng.random() < 0.2:
        cut = rng.randint(1, min(100, len(text) // 4))
        new_len = len(text) - cut
        text = text[:new_len]
        boundaries = [b for b in boundaries if b < new_len]

    return text, boundaries


# ---------------------------------------------------------------------------
# Multi-dataset combination with temperature sampling
# ---------------------------------------------------------------------------


def build_combined_dataset(
    dataset_file_lists: List[List[Path]],
    tokenizer: AutoTokenizer,
    window_size: int = 510,
    stride: int = 256,
    temperature: float = 2.0,
    augment: bool = True,
    seed: int = 42,
) -> Tuple["SentSplitDataset", List[float]]:
    """
    Build a combined training dataset from multiple UD corpora with
    temperature-based sampling weights.

    Temperature controls the balance between datasets:
      T=1.0  → natural (proportional) sampling
      T=2.0  → upsamples small datasets relative to large ones
      T→∞    → uniform across datasets regardless of size

    Returns:
        (combined_dataset, sample_weights) where sample_weights[i] is the
        weight for window i (for use with WeightedRandomSampler).
    """
    per_dataset_windows: List[List[Dict]] = []

    for i, file_list in enumerate(dataset_file_lists):
        ds = SentSplitDataset(
            tokenizer=tokenizer,
            file_paths=file_list,
            window_size=window_size,
            stride=stride,
            augment=augment,
            seed=seed + i,
        )
        per_dataset_windows.append(ds.windows)

    sizes = [len(w) for w in per_dataset_windows]
    total = sum(sizes)
    if total == 0:
        raise ValueError("No training windows found. Check data paths.")

    # w_d ∝ (n_d / total)^(1/T)
    raw_w = [(n / total) ** (1.0 / temperature) for n in sizes]
    z = sum(raw_w)
    dataset_weights = [w / z for w in raw_w]

    # Per-window sample weights
    sample_weights: List[float] = []
    for dw, size in zip(dataset_weights, sizes):
        per_w = dw / size if size > 0 else 0.0
        sample_weights.extend([per_w] * size)

    # Build a combined dataset by merging all windows
    combined = SentSplitDataset.__new__(SentSplitDataset)
    combined.windows = []
    combined.window_size = window_size
    combined.stride = stride
    for wins in per_dataset_windows:
        combined.windows.extend(wins)

    logger.info(
        f"Combined dataset: {len(combined.windows)} windows "
        f"from {len(dataset_file_lists)} datasets. Sizes: {sizes}. "
        f"Effective weights: {[f'{w:.3f}' for w in dataset_weights]}"
    )
    return combined, sample_weights


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------


def get_all_train_file_groups() -> List[List[Path]]:
    """
    Returns one group of files per UD dataset, containing only the train split.
    Used for temperature-based combined training.
    """
    groups = []
    for d in sorted(PROJECT_ROOT.glob("UD_*/")):
        files = sorted(d.glob("*-ud-train.sent_split"))
        if files:
            groups.append(files)
    return groups


def get_all_dev_files() -> List[Path]:
    """Return all dev .sent_split files sorted by path."""
    return sorted(PROJECT_ROOT.glob("UD_*/*-ud-dev.sent_split"))


def get_all_test_files() -> List[Path]:
    """Return all test .sent_split files sorted by path."""
    return sorted(PROJECT_ROOT.glob("UD_*/*-ud-test.sent_split"))


def detect_lang(filepath: Path) -> str:
    """Detect language from file path (en or it)."""
    name = filepath.stem.lower()
    if name.startswith("en_") or "english" in str(filepath).lower():
        return "en"
    if name.startswith("it_") or "italian" in str(filepath).lower():
        return "it"
    return "en"  # default fallback
