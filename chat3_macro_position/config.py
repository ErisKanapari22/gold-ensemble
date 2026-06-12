# chat3_macro_position/config.py
# Central configuration for Chat 3 — Macro/World Position Model

import os
from dotenv import load_dotenv

# Load .env from the repo root (one level up from this folder)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── FRED API ───────────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Data ───────────────────────────────────────────────────────────────────────
GOLD_TICKER   = "GC=F"        # Gold futures — yfinance
DXY_TICKER    = "DX-Y.NYB"   # Dollar Index — yfinance
SPX_TICKER    = "^GSPC"       # S&P 500 — yfinance (risk-on/off proxy)
DATA_YEARS    = 5             # How many years of history to fetch
FORWARD_DAYS  = 20            # Predict Gold return N days ahead

# Label thresholds
LABEL_THRESHOLD = 0.015       # |20d forward return| > 1.5% → DIRECTIONAL
ATR_FILTER      = 0.5         # move must also exceed ATR_FILTER * 20d ATR

# FRED series IDs
FRED_SERIES = {
    "fed_funds":   "FEDFUNDS",   # Fed Funds Rate (monthly)
    "cpi":         "CPIAUCSL",   # CPI all items (monthly)
    "yield_10y":   "DGS10",      # 10Y Treasury yield (daily)
    "yield_2y":    "DGS2",       # 2Y Treasury yield (daily)
    "yield_spread":"T10Y2Y",     # 10Y minus 2Y spread (daily)
}

# ── Model ──────────────────────────────────────────────────────────────────────
N_FEATURES   = 15             # Number of engineered features (see data.py)
HIDDEN_1     = 48
HIDDEN_2     = 24
DROPOUT      = 0.2
N_CLASSES    = 2
CLASS_NAMES  = ["DIRECTIONAL", "NEUTRAL"]
CLASS_WEIGHTS = [1.0, 1.0]   # DIRECTIONAL weighted higher to counter imbalance

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-2
PATIENCE      = 20            # Early stopping on DIRECTIONAL F1
TRAIN_RATIO   = 0.80

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "macro_best.pt")
SCALER_PATH     = os.path.join(BASE_DIR, "scaler.pkl")
DATA_CACHE_PATH = os.path.join(BASE_DIR, "macro_data_cache.csv")