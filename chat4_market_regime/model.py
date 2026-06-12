# model.py
# Regime classifier for XAU/USD.
# Input:  [batch, n_features]   (single timestep, no sequence dimension)
# Output: [batch, n_classes]    (raw logits — apply softmax for probabilities)
#
# Architecture switches between pure logistic regression and a small MLP
# based on config.USE_HIDDEN_LAYER (set this after running diagnose.py).

import torch
import torch.nn as nn

import config


class MarketRegimeModel(nn.Module):
    """
    BatchNorm1d(n_features) -> Linear classifier (logistic regression),
    or BatchNorm1d(n_features) -> Linear -> ReLU -> Dropout -> Linear (small MLP)
    if config.USE_HIDDEN_LAYER is True.
    """

    def __init__(
        self,
        n_features: int = config.N_FEATURES,
        n_classes: int  = config.N_CLASSES,
        use_hidden: bool = config.USE_HIDDEN_LAYER,
        hidden_dim: int  = config.HIDDEN_DIM,
        dropout: float   = config.DROPOUT,
    ):
        super().__init__()

        self.bn = nn.BatchNorm1d(n_features)

        if use_hidden:
            self.classifier = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            # Pure logistic regression
            self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]
        x = self.bn(x)
        logits = self.classifier(x)  # -> [batch, n_classes]
        return logits