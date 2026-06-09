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

    # Static class weights from config
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    weight_str = "  ".join(f"{n}={class_weights[i]:.1f}" for i, n in enumerate(config.CLASS_NAMES))
    print(f"Class weights: {weight_str}")

    # Model, loss, optimizer, scheduler
    model = MarketPredictionModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_loss = 0.0 
    patience_counter = 0

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
        all_val_true, all_val_preds = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                val_loss    += loss.item() * len(y_batch)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)
                all_val_true.extend(y_batch.cpu().tolist())
                all_val_preds.extend(preds.cpu().tolist())

        avg_train_loss = train_loss / train_total
        avg_val_loss   = val_loss   / val_total
        train_acc      = train_correct / train_total
        val_acc        = val_correct   / val_total

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss:   {avg_val_loss:.4f}  Acc: {val_acc:.3f}"
        )

        # Per-class recall via confusion matrix (pure torch, no sklearn)
        conf_matrix = torch.zeros(config.N_CLASSES, config.N_CLASSES, dtype=torch.long)
        for true, pred in zip(all_val_true, all_val_preds):
            conf_matrix[true][pred] += 1
        for i, name in enumerate(config.CLASS_NAMES):
            tp    = conf_matrix[i, i].item()
            total = conf_matrix[i].sum().item()
            recall = tp / total if total > 0 else 0.0
            print(f"    {name} recall: {recall:.3f}  ({tp}/{total})")

        # ── DIRECTIONAL F1 checkpoint ──
        dp      = conf_matrix[0, 0].item()
        dp_pred = conf_matrix[:, 0].sum().item()
        dp_true = conf_matrix[0].sum().item()
        prec    = dp / dp_pred if dp_pred > 0 else 0.0
        rec     = dp / dp_true if dp_true > 0 else 0.0
        dir_f1  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"    DIRECTIONAL F1: {dir_f1:.3f}  (prec={prec:.3f}  rec={rec:.3f})")

        if dir_f1 > best_val_loss:
            best_val_loss = dir_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "dir_f1": dir_f1,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  [saved] Checkpoint (DIRECTIONAL F1={dir_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {config.PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best DIRECTIONAL F1: {best_val_loss:.4f}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()