# model.py
# Transformer encoder for XAU/USD directional prediction.
# Input:  [batch, seq_len, n_features]
# Output: [batch, n_classes]  (raw logits — apply softmax for probabilities)

import math
import torch
import torch.nn as nn

import config


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al. 2017).
    Adds position information to the token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Transformer Encoder Model ─────────────────────────────────────────────────

class MarketPredictionModel(nn.Module):
    """
    Transformer encoder for time-series classification.

    Pipeline:
      raw features → linear projection → positional encoding
      → N transformer encoder layers → global average pool
      → fully connected classifier
    """

    def __init__(
        self,
        n_features: int = config.N_FEATURES,
        d_model: int    = config.D_MODEL,
        n_heads: int    = config.N_HEADS,
        n_layers: int   = config.N_LAYERS,
        n_classes: int  = config.N_CLASSES,
        dropout: float  = config.DROPOUT,
    ):
        super().__init__()

        # Project raw feature dim → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # input shape: [batch, seq, features]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features]
        x = self.input_proj(x)          # → [batch, seq_len, d_model]
        x = self.pos_enc(x)             # add positional info
        x = self.transformer(x)         # → [batch, seq_len, d_model]
        x = x.mean(dim=1)               # global average pool → [batch, d_model]
        logits = self.classifier(x)     # → [batch, n_classes]
        return logits