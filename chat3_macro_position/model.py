# chat3_macro_position/model.py
# MLP classifier for macro regime → Gold direction.
# Input:  [batch, n_features]   (tabular — no sequence dimension)
# Output: [batch, n_classes]    (raw logits)

import torch
import torch.nn as nn
import config


class MacroPositionModel(nn.Module):
    """
    Three-layer MLP with BatchNorm, ReLU, and Dropout.

    Why MLP and not Transformer?
      - Macro data is tabular (one row per day, ~15 features)
      - No meaningful sequential pattern within a single timestep
      - Dataset is small (~1000-1200 rows after feature engineering)
      - MLP + BatchNorm generalizes better than attention on small tabular data
    """

    def __init__(
        self,
        n_features: int = config.N_FEATURES,
        hidden_1:   int = config.HIDDEN_1,
        hidden_2:   int = config.HIDDEN_2,
        n_classes:  int = config.N_CLASSES,
        dropout:  float = config.DROPOUT,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]
        return self.network(x)   # → [batch, n_classes]