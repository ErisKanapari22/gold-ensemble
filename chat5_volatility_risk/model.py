# model.py
# Volatility & Risk classifier.
# Input:  [batch, n_features]  →  Output: [batch, n_classes]  (raw logits)
#
# Two architectures, toggled by config.USE_HIDDEN_LAYER:
#
#   Linear (default):
#       BatchNorm1d(12) → Linear(12 → 2)
#
#   MLP (if diagnose.py shows non-linear structure):
#       BatchNorm1d(12) → Linear(12 → 32) → ReLU → Dropout(0.2) → Linear(32 → 2)

import torch
import torch.nn as nn

import config


class VolatilityRiskModel(nn.Module):

    def __init__(
        self,
        n_features: int  = config.N_FEATURES,
        n_classes:  int  = config.N_CLASSES,
        hidden_dim: int  = config.HIDDEN_DIM,
        dropout:    float = config.DROPOUT,
        use_hidden: bool  = config.USE_HIDDEN_LAYER,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)

        if use_hidden:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            print(f"[model] Architecture: MLP  (hidden_dim={hidden_dim})")
        else:
            self.net = nn.Linear(n_features, n_classes)
            print("[model] Architecture: Linear (logistic regression baseline)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]
        x = self.bn(x)
        return self.net(x)   # → [batch, n_classes]