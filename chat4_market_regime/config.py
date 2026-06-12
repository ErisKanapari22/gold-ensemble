# config.py
# Central configuration for Chat 4 — Market Regime Detector
# All hyperparameters and paths live here. Do not hardcode values in other files.

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER = "GC=F"                   # Gold futures on Yahoo Finance
TIMEFRAME = "1h"                  # Candle interval
TRAIN_RATIO = 0.80                # Chronological split — no shuffling

# ── Feature engineering windows ─────────────────────────────────────────────
ADX_PERIOD = 14
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
BB_WINDOW = 20
ATR_PERIOD = 14
ADX_SLOPE_PERIOD = 5
RANGE_LOOKBACK = 20

# ── Labeling ──────────────────────────────────────────────────────────────────
# DIRECTIONAL if ADX14 > ADX_THRESHOLD, else NEUTRAL.
# diagnose.py may suggest tuning this value.
ADX_THRESHOLD = 25.0

# Feature columns produced by data.py (order must stay consistent)
FEATURE_COLS = [
    "adx14",
    "ema20_norm",
    "ema50_norm",
    "ema200_norm",
    "ema_spread_20_50",
    "bb_width",
    "atr14_norm",
    "adx_slope5",
    "range_position",
    "hh_ll_count",
]

# ── Model ─────────────────────────────────────────────────────────────────────
N_FEATURES = len(FEATURE_COLS)
N_CLASSES = 2                      # DIRECTIONAL=0, NEUTRAL=1
CLASS_NAMES = ["DIRECTIONAL", "NEUTRAL"]
CLASS_WEIGHTS = [2.0, 1.0]          # secondary safeguard on top of WeightedRandomSampler

# Architecture toggle — set/confirm AFTER reading diagnose.py output
USE_HIDDEN_LAYER = False           # False -> pure logistic regression (BatchNorm + Linear)
HIDDEN_DIM = 16                    # only used if USE_HIDDEN_LAYER = True
DROPOUT = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10                      # Early stopping on DIRECTIONAL F1

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "market_regime_best.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_CACHE_PATH = os.path.join(BASE_DIR, "data_cache.csv")