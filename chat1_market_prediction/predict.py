# predict.py
# Runtime inference for Chat 1 — Market Prediction Model.
# Loads the trained model + scaler, runs inference on the latest 60 candles,
# and returns the standardized signal dict consumed by the Layer 2 aggregator.
#
# Usage (standalone):
#   python predict.py
#
# Usage (imported by aggregator):
#   from chat1_market_prediction.predict import get_signal

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import joblib
import yfinance as yf

import config
from data import engineer_features
from model import MarketPredictionModel


# Label index → string
IDX_TO_SIGNAL = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}


def load_model_and_scaler():
    """Load the trained model checkpoint and the fitted scaler."""
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {config.CHECKPOINT_PATH}. "
            "Run train.py first."
        )
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(
            f"No scaler found at {config.SCALER_PATH}. "
            "Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketPredictionModel().to(device)
    ckpt  = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler, device


def fetch_latest_candles(n: int = config.LOOKBACK + 50) -> pd.DataFrame:
    """
    Download the most recent `n` 1H candles for GC=F.
    We fetch extra candles to absorb the indicator warmup period.
    """
    raw = yf.download(
        config.TICKER,
        period="60d",
        interval=config.TIMEFRAME,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.tail(n)


def get_signal(ohlcv_df: pd.DataFrame = None) -> dict:
    """
    Main inference function.

    Args:
        ohlcv_df: Optional pre-fetched OHLCV DataFrame.
                  If None, data is downloaded from Yahoo Finance.

    Returns:
        dict with keys: model, asset, signal, confidence, timestamp
    """
    model, scaler, device = load_model_and_scaler()

    # 1. Get raw OHLCV
    if ohlcv_df is None:
        ohlcv_df = fetch_latest_candles()

    # 2. Engineer features
    df = engineer_features(ohlcv_df)

    # 3. Take the last LOOKBACK rows
    if len(df) < config.LOOKBACK:
        raise ValueError(
            f"Not enough data after feature engineering. "
            f"Need {config.LOOKBACK}, got {len(df)}."
        )
    window = df[config.FEATURE_COLS].values[-config.LOOKBACK:]  # [60, n_features]

    # 4. Scale using the saved scaler
    window_scaled = scaler.transform(window).astype(np.float32)

    # 5. Build tensor: [1, 60, n_features]
    x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # 6. Forward pass
    with torch.no_grad():
        logits = model(x)                         # [1, 3]
        probs  = torch.softmax(logits, dim=1)[0]  # [3]

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    signal_str  = IDX_TO_SIGNAL[pred_idx]

    # 7. Build output dict (keys must match aggregator contract exactly)
    output = {
        "model":      "market_prediction",
        "asset":      "XAU/USD",
        "signal":     signal_str,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return output


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_signal()
    print("\n=== Chat 1 Signal Output ===")
    for k, v in result.items():
        print(f"  {k}: {v}")