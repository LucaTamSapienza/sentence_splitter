"""
src/model.py — XLM-RoBERTa sentence boundary classifier.

Architecture:
  XLM-RoBERTa backbone → dropout → linear(hidden, 1) → per-token logit
  Loss: focal binary cross-entropy (handles 1:25 class imbalance)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary focal loss, ignoring positions where labels == -100.

    Formula:  FL = -α_t · (1 - p_t)^γ · log(p_t)
    where:
      p_t = σ(logit)  for positive labels
      p_t = 1-σ(logit) for negative labels
      α_t = α for positives, (1-α) for negatives

    Args:
        logits: Raw scores, any shape.
        labels: Int labels {-100, 0, 1}, same shape as logits.
        alpha:  Weight for positive (boundary) class. Set >0.5 to upweight positives.
        gamma:  Focal exponent. 0 = standard BCE; 2 = focus on hard examples.

    Returns:
        Scalar mean loss over non-ignored positions.
    """
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)

    mask    = labels != -100
    logits  = logits[mask]
    targets = labels[mask].float()

    if targets.numel() == 0:
        return logits.sum() * 0.0  # stay in compute graph, return 0

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    probs   = torch.sigmoid(logits)
    p_t     = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    weight  = alpha_t * (1.0 - p_t).pow(gamma)

    return (weight * bce).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class XLMRSentenceSplitter(nn.Module):
    """
    XLM-RoBERTa-large with a single linear boundary-detection head.

    Input:  (batch, 512) token IDs  [CLS + content + SEP + padding]
    Output: (batch, 512) per-token boundary logits

    At inference, only content token positions (1 to window_size) are used.
    Logits are averaged across overlapping windows, then mapped to chars.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        dropout: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:      (batch, seq_len) int64
            attention_mask: (batch, seq_len) int64
            labels:         (batch, seq_len) int64 optional; -100 for ignored positions

        Returns:
            logits: (batch, seq_len) float32 raw boundary scores
            loss:   scalar if labels provided, else None
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h   = self.dropout(out.last_hidden_state)   # (batch, seq, hidden)
        logits = self.classifier(h).squeeze(-1)      # (batch, seq)

        loss = None
        if labels is not None:
            loss = focal_bce_loss(
                logits, labels,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights to a .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))
        logger.info(f"Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model_name: str = "xlm-roberta-large",
        map_location: str = "cpu",
        **kwargs,
    ) -> "XLMRSentenceSplitter":
        """Load a model from a checkpoint file."""
        model = cls(model_name=model_name, **kwargs)
        state = torch.load(str(path), map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Checkpoint loaded ← {path}")
        return model
