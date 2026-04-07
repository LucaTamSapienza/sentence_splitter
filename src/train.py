"""
src/train.py — Training loop for XLMRSentenceSplitter.

Features:
  - Mixed-precision (fp16) training via torch.cuda.amp
  - Gradient accumulation
  - Linear warmup + linear decay LR schedule
  - Per-epoch evaluation on all dev files
  - Early stopping + best-checkpoint saving
  - Temperature-based dataset sampling
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import (
    build_combined_dataset,
    get_all_dev_files,
    get_all_train_file_groups,
    parse_sent_split_file,
)
from .model import XLMRSentenceSplitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (mirrors configs/xlmr.yaml)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    model_name:     str   = "xlm-roberta-large"
    output_dir:     str   = "checkpoints/xlmr"

    # Data
    window_size:    int   = 510
    stride:         int   = 256
    temperature:    float = 2.0
    augment:        bool  = True
    seed:           int   = 42

    # Training
    batch_size:     int   = 16
    grad_accum:     int   = 2        # effective batch = batch_size * grad_accum
    num_epochs:     int   = 10
    learning_rate:  float = 2e-5
    warmup_ratio:   float = 0.1
    weight_decay:   float = 0.01
    max_grad_norm:  float = 1.0
    fp16:           bool  = True
    num_workers:    int   = 4

    # Focal loss
    focal_alpha:    float = 0.75
    focal_gamma:    float = 2.0

    # Early stopping
    patience:       int   = 3

    # Inference during eval
    boundary_threshold: float = 0.5


def train_from_config(cfg: TrainConfig) -> None:
    """Full training pipeline. Call from scripts/train_xlmr.py."""
    _setup_logging()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Model: {cfg.model_name}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ---- Training data ----------------------------------------------------
    logger.info("Building training dataset (this may take a few minutes)...")
    train_file_groups = get_all_train_file_groups()
    if not train_file_groups:
        raise FileNotFoundError(
            "No UD_*/*-ud-train.sent_split files found. "
            "Check that you are running from the sentence_splitting/ directory."
        )

    train_dataset, sample_weights = build_combined_dataset(
        dataset_file_lists=train_file_groups,
        tokenizer=tokenizer,
        window_size=cfg.window_size,
        stride=cfg.stride,
        temperature=cfg.temperature,
        augment=cfg.augment,
        seed=cfg.seed,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    dev_files = get_all_dev_files()
    logger.info(
        f"Train windows: {len(train_dataset)} | Dev files: {len(dev_files)} | "
        f"Steps/epoch: {len(train_loader)}"
    )

    # ---- Model ------------------------------------------------------------
    model = XLMRSentenceSplitter(
        model_name=cfg.model_name,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
    ).to(device)

    # ---- Optimizer (AdamW with weight-decay on non-bias params) -----------
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.learning_rate,
    )

    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler() if cfg.fp16 and device.type == "cuda" else None

    # ---- Training loop ----------------------------------------------------
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)

            if scaler is not None:
                with autocast():
                    _, loss = model(ids, masks, labs)
                scaler.scale(loss / cfg.grad_accum).backward()
            else:
                _, loss = model(ids, masks, labs)
                (loss / cfg.grad_accum).backward()

            epoch_loss += loss.item()

            if (step + 1) % cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        dev_f1   = _evaluate(model, tokenizer, dev_files, cfg, device)

        logger.info(
            f"Epoch {epoch:02d}/{cfg.num_epochs} | "
            f"loss: {avg_loss:.4f} | dev F1: {dev_f1:.4f}"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            ckpt = output_dir / "best_model.pt"
            model.save_checkpoint(str(ckpt))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    f"Early stopping after {epoch} epochs "
                    f"(no improvement for {cfg.patience} epochs). "
                    f"Best dev F1: {best_f1:.4f}"
                )
                return

    logger.info(f"Training complete. Best dev F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Internal evaluation helper
# ---------------------------------------------------------------------------


def _evaluate(
    model: XLMRSentenceSplitter,
    tokenizer: AutoTokenizer,
    dev_files: List[Path],
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    """Evaluate model on all dev files. Returns macro-average F1."""
    # Local import to avoid circular dependency
    from .inference import predict_xlmr
    from .evaluate import boundary_f1

    model.eval()
    all_f1: List[float] = []

    with torch.no_grad():
        for fpath in dev_files:
            text, gold_boundaries = parse_sent_split_file(fpath)
            if not text:
                continue

            char_probs = predict_xlmr(
                model, tokenizer, text,
                window_size=cfg.window_size,
                stride=cfg.stride,
                device=device,
            )

            pred = {i for i, p in enumerate(char_probs) if p >= cfg.boundary_threshold}
            gold = set(gold_boundaries)
            metrics = boundary_f1(pred, gold)
            all_f1.append(metrics["f1"])

    model.train()
    return sum(all_f1) / len(all_f1) if all_f1 else 0.0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
