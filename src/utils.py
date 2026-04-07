"""
src/utils.py — Shared utilities: config loading, seeding, logging.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return as a flat-ish dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Project root (sentence_splitting/)
PROJECT_ROOT = Path(__file__).parent.parent

# Convenience paths
DATA_DIR = PROJECT_ROOT                  # UD_*/ dirs live here
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)
