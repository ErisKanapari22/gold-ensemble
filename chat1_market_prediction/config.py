# config.py
# Central configuration for Chat 1 — Market Prediction Model
# All hyperparameters and paths live here. Do not hardcode values in other files.

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER = "GC=F"                   # Gold futures on Yahoo Finance
TIMEFRAME = "1h"                  # Candle interval
LOOKBACK = 60                     # Sequence length fed to the model (candles)
TRAIN_RATIO = 0.80                # Chronological split — no shuffling
LABEL_THRESHOLD = 0.002           # ±0.2% next-close move → UP / DOWN, else NEUTRAL
ATR_FILTER = 0.5                  # move must also exceed ATR_FILTER * ATR14 (price units) to be labeled UP/DOWN

# Feature columns produced by data.py (order must stay consistent)
FEATURE_COLS = [
    "returns",
    "sma20",
    "ema9",
    "rsi14",
    "atr14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "bb_width",
]

# ── Model ─────────────────────────────────────────────────────────────────────
N_FEATURES = len(FEATURE_COLS)    # Input dimension per timestep
D_MODEL = 64                      # Transformer hidden dimension
N_HEADS = 4                       # Attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS = 2                      # Transformer encoder layers
DROPOUT = 0.2
N_CLASSES = 2                     # DIRECTIONAL=0, NEUTRAL=1
CLASS_NAMES = ["DIRECTIONAL", "NEUTRAL"]
CLASS_WEIGHTS = [2.0, 1.0]       # DIRECTIONAL gets 3x weight to counter class imbalance

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10                     # Early stopping: halt if val loss doesn't improve for this many epochs

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "market_pred_best.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_CACHE_PATH = os.path.join(BASE_DIR, "data_cache.csv")  # optional local CSV cache