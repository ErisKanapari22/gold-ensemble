# train.py
# Trains the MarketPredictionModel and saves the best checkpoint.
# Run: python train.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data import get_dataloaders
from model import MarketPredictionModel


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train():
    set_seed(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader = get_dataloaders()

    # Model, loss, optimizer, scheduler
    model = MarketPredictionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                val_loss    += loss.item() * len(y_batch)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        avg_train_loss = train_loss / train_total
        avg_val_loss   = val_loss   / val_total
        train_acc      = train_correct / train_total
        val_acc        = val_correct   / val_total

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss:   {avg_val_loss:.4f}  Acc: {val_acc:.3f}"
        )

        # ── Checkpoint (best val loss) ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  ✓ Checkpoint saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()