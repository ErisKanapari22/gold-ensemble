# predict.py
# Runtime inference for Chat 4 — Market Regime Detector.
# Loads the trained model + scaler, runs inference on the latest candle,
# and returns the standardized signal dict consumed by the Layer 2 aggregator.
#
# Usage (standalone):
#   python predict.py
#
# Usage (imported by aggregator):
#   from chat4_market_regime.predict import get_signal

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import joblib
import yfinance as yf

import config
from data import engineer_features
from model import MarketRegimeModel


# Binary model: 0=DIRECTIONAL, 1=NEUTRAL
# Direction (UP vs DOWN) is reconstructed from the sign of EMA20-EMA50 spread.


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
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)

    # Build model with the architecture recorded at training time
    use_hidden = ckpt.get("use_hidden_layer", config.USE_HIDDEN_LAYER)
    model = MarketRegimeModel(use_hidden=use_hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler, device


def fetch_latest_candles(n: int = 300) -> pd.DataFrame:
    """
    Download recent 1H candles for GC=F. We fetch extra candles
    (n=300) to cover EMA200 + indicator warmup periods.
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

    if len(df) < 1:
        raise ValueError("Not enough data after feature engineering to make a prediction.")

    # 3. Take the LAST row only (no sequence — single-timestep model)
    last_row = df[config.FEATURE_COLS].values[-1:].astype(np.float32)  # [1, n_features]

    # 4. Scale using the saved scaler
    row_scaled = scaler.transform(last_row).astype(np.float32)

    # 5. Build tensor: [1, n_features]
    x = torch.tensor(row_scaled, dtype=torch.float32).to(device)

    # 6. Forward pass — binary: 0=DIRECTIONAL, 1=NEUTRAL
    with torch.no_grad():
        logits = model(x)                         # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]  # [2]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    # 7. Reconstruct UP / DOWN / NEUTRAL
    if pred_idx == 0:  # DIRECTIONAL — direction from sign of EMA20-EMA50 spread
        spread = float(df["ema_spread_20_50"].values[-1])
        signal_str = "UP" if spread > 0 else "DOWN"
    else:
        signal_str = "NEUTRAL"

    # 8. Build output dict (keys must match aggregator contract exactly)
    output = {
        "model":      "market_regime",
        "asset":      "XAU/USD",
        "signal":     signal_str,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return output


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_signal()
    print("\n=== Chat 4 Signal Output ===")
    for k, v in result.items():
        print(f"  {k}: {v}")