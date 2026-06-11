# config.py
# Central configuration for Chat 2 — Sentiment Analysis Model
# All hyperparameters and paths live here. Do not hardcode values in other files.

import os
from dotenv import load_dotenv

# Load .env file from the root of the project (one level up from this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── NewsAPI ────────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")   # Set this in your .env file
NEWS_KEYWORDS = ["gold", "XAU", "bullion", "Fed", "inflation", "interest rates", "DXY", "dollar"]
NEWS_LOOKBACK_DAYS = 1                          # How many days back to fetch headlines

# ── Model / FinBERT ────────────────────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"        # HuggingFace model ID
MAX_TOKEN_LENGTH   = 128                        # Max tokens per headline
UNFREEZE_FINBERT   = False                      # True = fine-tune full model (slow), False = head only

# Classification head architecture
HIDDEN_DIM = 256
DROPOUT    = 0.2
N_CLASSES  = 2                                  # 0=DIRECTIONAL, 1=NEUTRAL
CLASS_NAMES   = ["DIRECTIONAL", "NEUTRAL"]
CLASS_WEIGHTS = [2.0, 1.0]                      # Upweight DIRECTIONAL to fight class imbalance

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 16                              # Small — our dataset is small
LEARNING_RATE = 2e-5                            # Low LR for fine-tuning on top of FinBERT
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10                             # Early stopping patience (DIRECTIONAL F1)
TRAIN_RATIO   = 0.80

# ── Data source ───────────────────────────────────────────────────────────────
# "hardcoded" = use built-in labeled examples in data.py (default, works immediately)
# "csv"       = load from headlines.csv (use this if you have your own labeled data)
DATA_SOURCE = "hardcoded"
CSV_PATH    = os.path.join(os.path.dirname(__file__), "headlines.csv")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sentiment_best.pt")