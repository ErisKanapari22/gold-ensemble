# data.py
# Loads and tokenizes headline data for Chat 2 — Sentiment Analysis Model.
# Supports two data sources (set DATA_SOURCE in config.py):
#   "hardcoded" — uses built-in labeled examples, works immediately, no CSV needed
#   "csv"       — loads from headlines.csv with columns: headline, label (0=DIRECTIONAL, 1=NEUTRAL)

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

import config


# ── Hardcoded labeled headlines ───────────────────────────────────────────────
# Label: 0 = DIRECTIONAL (strongly bullish or bearish for Gold)
#         1 = NEUTRAL     (unclear, mixed, or irrelevant)
#
# These are enough to fine-tune the classification head on top of FinBERT.
# Add more rows here to improve accuracy over time.

HARDCODED_HEADLINES = [
    # DIRECTIONAL — Bullish for Gold (UP)
    ("Gold surges as Fed signals pause in rate hikes", 0),
    ("XAU/USD jumps 2% after weaker-than-expected US jobs report", 0),
    ("Gold hits 6-month high as inflation data beats expectations", 0),
    ("Bullion rallies sharply on safe-haven demand amid geopolitical tensions", 0),
    ("Gold futures soar as dollar index drops to multi-month lows", 0),
    ("Fed dovish pivot sends gold prices sharply higher", 0),
    ("XAU breaks key resistance as real yields fall", 0),
    ("Gold demand surges as banking sector fears grip markets", 0),
    ("Central banks accelerate gold purchases, prices spike", 0),
    ("Inflation unexpectedly rises, gold jumps on safe-haven flows", 0),
    ("Gold rises as US debt ceiling fears escalate", 0),
    ("XAU/USD climbs as risk-off sentiment dominates markets", 0),
    ("Gold soars past $2000 as recession fears mount", 0),
    ("Weak US retail sales data pushes gold sharply higher", 0),
    ("Gold rallies after surprise Fed rate cut", 0),

    # DIRECTIONAL — Bearish for Gold (DOWN)
    ("Gold slides as Fed signals more aggressive rate hikes ahead", 0),
    ("XAU/USD tumbles after strong US jobs report reduces rate cut hopes", 0),
    ("Gold falls sharply as dollar strengthens on hawkish Fed comments", 0),
    ("Bullion drops to 3-month low as risk appetite returns", 0),
    ("Gold sells off after hotter-than-expected CPI print", 0),
    ("XAU collapses as Treasury yields surge to cycle highs", 0),
    ("Gold breaks below $1900 on strong US economic data", 0),
    ("Dollar surges, gold suffers largest weekly loss in months", 0),
    ("Gold retreats as Fed officials push back on rate cut expectations", 0),
    ("XAU/USD plunges as equities rally and safe-haven demand fades", 0),
    ("Gold drops as trade deal optimism reduces uncertainty", 0),
    ("Bullion slides after US GDP growth beats forecasts", 0),
    ("Gold pressured lower as FOMC minutes reveal hawkish tone", 0),
    ("XAU falls as DXY climbs to yearly high on strong macro data", 0),
    ("Gold loses ground as global risk sentiment improves sharply", 0),

    # NEUTRAL
    ("Gold prices steady ahead of key Fed meeting this week", 1),
    ("XAU/USD trades in narrow range as markets await inflation data", 1),
    ("Analysts divided on gold outlook for the second half of 2025", 1),
    ("Gold holds near $1950 with little directional conviction", 1),
    ("Mixed signals from Fed leave gold traders on the sidelines", 1),
    ("Gold market quiet ahead of US holiday weekend", 1),
    ("XAU/USD consolidates after last week's volatile session", 1),
    ("Gold investors cautious as conflicting data clouds the outlook", 1),
    ("Bullion price unchanged as dollar and yields move sideways", 1),
    ("Gold treads water as traders weigh recession vs inflation risks", 1),
    ("XAU steady with no major catalysts expected this week", 1),
    ("Gold range-bound as market awaits next CPI release", 1),
    ("No clear trend for gold as macro picture remains uncertain", 1),
    ("Gold flat on low volume session ahead of earnings season", 1),
    ("XAU/USD little changed as geopolitical situation stabilizes", 1),
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class HeadlineDataset(Dataset):
    """
    PyTorch Dataset for tokenized headlines.
    Each item: input_ids, attention_mask, label
    """

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_dataloaders():
    """
    Full pipeline: load headlines → tokenize → split → DataLoaders.

    Returns:
        train_loader, val_loader
    """
    # 1. Load raw headlines and labels
    if config.DATA_SOURCE == "csv":
        if not os.path.exists(config.CSV_PATH):
            raise FileNotFoundError(
                f"CSV not found at {config.CSV_PATH}. "
                "Set DATA_SOURCE = 'hardcoded' in config.py to use built-in data."
            )
        df = pd.read_csv(config.CSV_PATH)
        if "headline" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: 'headline', 'label' (0=DIRECTIONAL, 1=NEUTRAL)")
        headlines = df["headline"].tolist()
        labels    = df["label"].tolist()
        print(f"Loaded {len(headlines)} headlines from CSV.")
    else:
        headlines = [h for h, _ in HARDCODED_HEADLINES]
        labels    = [l for _, l in HARDCODED_HEADLINES]
        print(f"Using {len(headlines)} hardcoded labeled headlines.")

    # 2. Print class distribution
    total = len(labels)
    for cls, name in enumerate(config.CLASS_NAMES):
        count = labels.count(cls)
        print(f"  {name:>12}: {count:4d}  ({count / total * 100:.1f}%)")

    # 3. Stratified train/val split — keeps class ratios balanced in both splits
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        headlines,
        labels,
        test_size=1 - config.TRAIN_RATIO,
        random_state=config.SEED,
        stratify=labels,
    )
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    # 4. Tokenize using FinBERT's own tokenizer
    print("Loading FinBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)

    def tokenize(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_TOKEN_LENGTH,
            return_tensors="pt",   # returns dict of PyTorch tensors
        )

    # Convert to plain lists so tokenizer handles them correctly
    train_encodings = tokenize(list(train_texts))
    val_encodings   = tokenize(list(val_texts))

    # 5. Build Datasets
    train_dataset = HeadlineDataset(train_encodings, list(train_labels))
    val_dataset   = HeadlineDataset(val_encodings,   list(val_labels))

    # 6. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader