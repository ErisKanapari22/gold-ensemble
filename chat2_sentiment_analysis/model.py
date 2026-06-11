# model.py
# Sentiment classification model for Chat 2.
#
# Architecture:
#   - Backbone: ProsusAI/finbert (pre-trained, frozen by default)
#   - Head:     Linear(768 → 256) → ReLU → Dropout → Linear(256 → 2)
#
# FinBERT outputs a [CLS] token embedding of size 768.
# We pass that through the classification head to get 2 logits:
#   0 = DIRECTIONAL, 1 = NEUTRAL
#
# Input:  tokenized headline (input_ids, attention_mask) — shape [batch, seq_len]
# Output: raw logits — shape [batch, 2]  (apply softmax for probabilities)

import torch
import torch.nn as nn
from transformers import BertModel

import config


class SentimentModel(nn.Module):
    """
    FinBERT + trainable classification head.

    The FinBERT body is frozen by default (UNFREEZE_FINBERT = False in config.py).
    Only the classification head weights are updated during training.
    This is fast, avoids overfitting on small datasets, and still leverages
    FinBERT's deep understanding of financial language.
    """

    def __init__(self):
        super().__init__()

        # Load pre-trained FinBERT body
        print("Loading FinBERT backbone from HuggingFace...")
        self.finbert = BertModel.from_pretrained(config.FINBERT_MODEL_NAME)

        # Freeze all FinBERT parameters by default
        if not config.UNFREEZE_FINBERT:
            for param in self.finbert.parameters():
                param.requires_grad = False
            print("FinBERT body: FROZEN (only classification head will train)")
        else:
            print("FinBERT body: UNFROZEN (full fine-tuning — this is slow)")

        # Classification head — sits on top of FinBERT's [CLS] token (size 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, config.HIDDEN_DIM),   # 768 → 256
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.N_CLASSES),  # 256 → 2
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [batch, seq_len] — tokenized headline
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding

        Returns:
            logits: [batch, 2] — raw scores for [DIRECTIONAL, NEUTRAL]
        """
        # Pass through FinBERT — we only need the [CLS] token output
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.last_hidden_state: [batch, seq_len, 768]
        # The [CLS] token is always at position 0 — it summarizes the whole sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # Pass through classification head
        logits = self.classifier(cls_embedding)  # [batch, 2]
        return logits