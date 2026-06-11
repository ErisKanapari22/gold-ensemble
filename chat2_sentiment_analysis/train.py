# train.py
# Fine-tunes the classification head on top of frozen FinBERT.
# Run: python train.py
#
# What gets trained: ONLY the classification head (Linear → ReLU → Dropout → Linear)
# What stays frozen: the entire FinBERT body (110M params untouched)
#
# Saves the best checkpoint based on DIRECTIONAL F1 (not val loss).

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

import config
from data import get_dataloaders
from model import SentimentModel


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

    # Model
    model = SentimentModel().to(device)

    # Count trainable vs frozen parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}  (classification head only)")

    # Class weights — upweight DIRECTIONAL to fight class imbalance
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    weight_str = "  ".join(f"{n}={class_weights[i]:.1f}" for i, n in enumerate(config.CLASS_NAMES))
    print(f"Class weights: {weight_str}")

    # Loss and optimizer
    # Note: optimizer only needs parameters where requires_grad=True
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_dir_f1      = 0.0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += len(labels)

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_true, all_val_preds      = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * len(labels)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)
                all_val_true.extend(labels.cpu().tolist())
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

        # ── Per-class recall via confusion matrix ──
        conf_matrix = torch.zeros(config.N_CLASSES, config.N_CLASSES, dtype=torch.long)
        for true, pred in zip(all_val_true, all_val_preds):
            conf_matrix[true][pred] += 1

        for i, name in enumerate(config.CLASS_NAMES):
            tp     = conf_matrix[i, i].item()
            total  = conf_matrix[i].sum().item()
            recall = tp / total if total > 0 else 0.0
            print(f"    {name} recall: {recall:.3f}  ({tp}/{total})")

        # ── DIRECTIONAL F1 — used for checkpointing ──
        dp      = conf_matrix[0, 0].item()
        dp_pred = conf_matrix[:, 0].sum().item()
        dp_true = conf_matrix[0].sum().item()
        prec    = dp / dp_pred if dp_pred > 0 else 0.0
        rec     = dp / dp_true if dp_true > 0 else 0.0
        dir_f1  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"    DIRECTIONAL F1: {dir_f1:.3f}  (prec={prec:.3f}  rec={rec:.3f})")

        # ── Checkpoint if best F1 ──
        if dir_f1 > best_dir_f1:
            best_dir_f1      = dir_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch":             epoch,
                    "model_state_dict":  model.state_dict(),
                    "dir_f1":            dir_f1,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  [saved] Checkpoint (DIRECTIONAL F1={dir_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {config.PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best DIRECTIONAL F1: {best_dir_f1:.4f}")
    print(f"Checkpoint saved at: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()