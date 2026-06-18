# config.py
# Central configuration for Chat 5 — Volatility & Risk Model.
# All hyperparameters and paths live here. Do not hardcode values in other files.

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER      = "GC=F"
VIX_TICKER  = "^VIX"
TIMEFRAME   = "1h"
TRAIN_RATIO = 0.80

# ── Label thresholds ───────────────────────────────────────────────────────────
VOL_THRESHOLD    = 1.1    # ATR14/ATR50 ratio must exceed this for DIRECTIONAL
RETURN_THRESHOLD = 0.005  # |5-bar price return| must also exceed this (0.5 %)
# Both conditions must hold — filters vol spikes that are just noise (no price move)

# ── Feature set ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "atr14",          # Short-term volatility
    "atr50",          # Medium-term volatility baseline
    "atr_ratio",      # ATR14 / ATR50  — core feature: is current vol abnormal?
    "atr14_slope",    # 5-bar slope of ATR14  — is vol rising or falling?
    "bb_width",       # Bollinger Band width (20-bar, 2σ)  — compression / expansion
    "bb_width_slope", # 5-bar slope of BB width
    "hv20",           # 20-bar historical vol (annualized)
    "hv_ratio",       # HV20 / HV60  — short vs long-term vol regime
    "gap_ratio",      # |Open - prev_Close| / ATR14  — overnight gap / news risk
    "vol_ratio",      # Volume / Volume_MA20  — does volume confirm vol spike?
    "vix",            # VIX (forward-filled daily → 1H)  — global risk sentiment
    "return_5",       # 5-bar price return  — direction and energy proxy
]
N_FEATURES = len(FEATURE_COLS)   # 12

# ── Model ─────────────────────────────────────────────────────────────────────
N_CLASSES        = 2
CLASS_NAMES      = ["DIRECTIONAL", "NEUTRAL"]
CLASS_WEIGHTS    = [2.0, 1.0]    # DIRECTIONAL upweighted; do NOT use inverse-freq
USE_HIDDEN_LAYER = False         # Set True if diagnose.py shows non-linear structure
HIDDEN_DIM       = 32
DROPOUT          = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10               # Early stopping on DIRECTIONAL F1

# ── Inference guard ────────────────────────────────────────────────────────────
EXTREME_VOL_RATIO = 2.0   # ATR14/ATR50 > this → override to NEUTRAL (news gap)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "vol_risk_best.pt")
SCALER_PATH     = os.path.join(BASE_DIR, "scaler.pkl")
DATA_CACHE_PATH = os.path.join(BASE_DIR, "data_cache.csv")
VIX_CACHE_PATH  = os.path.join(BASE_DIR, "vix_cache.csv")