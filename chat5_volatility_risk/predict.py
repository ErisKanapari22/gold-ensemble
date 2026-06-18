# predict.py
# Runtime inference for Chat 5 — Volatility & Risk Model.
# Returns the standardised signal dict consumed by the Layer 2 aggregator.
#
# Usage (standalone):  python predict.py
# Usage (imported):    from chat5_volatility_risk.predict import get_signal

import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import joblib
import yfinance as yf

import config
from data import engineer_features, _align_vix
from model import VolatilityRiskModel


def load_model_and_scaler():
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {config.CHECKPOINT_PATH}. Run train.py first."
        )
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(
            f"No scaler at {config.SCALER_PATH}. Run train.py first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True)
    use_hidden = ckpt.get("use_hidden_layer", config.USE_HIDDEN_LAYER)
    model  = VolatilityRiskModel(use_hidden=use_hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler, device


def _fetch_fresh_gold(n_days: int = 60) -> pd.DataFrame:
    """Always downloads fresh 1H Gold data — bypasses the training cache."""
    raw = yf.download(
        config.TICKER,
        period=f"{n_days}d",
        interval=config.TIMEFRAME,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _fetch_fresh_vix() -> pd.Series:
    """Always downloads fresh daily VIX data — bypasses the training cache."""
    raw = yf.download(
        config.VIX_TICKER,
        period="10d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].squeeze()


def get_signal(ohlcv_df: pd.DataFrame = None) -> dict:
    """
    Main inference entry point.

    Args:
        ohlcv_df: Optional pre-fetched OHLCV DataFrame.
                  If None, fresh data is downloaded from Yahoo Finance.

    Returns:
        dict: {model, asset, signal, confidence, timestamp}
    """
    model, scaler, device = load_model_and_scaler()

    # 1. Fetch data
    if ohlcv_df is None:
        ohlcv_df = _fetch_fresh_gold()
    vix_series = _fetch_fresh_vix()

    # 2. Engineer features
    df = engineer_features(ohlcv_df, vix_series)
    if df.empty:
        raise ValueError("Feature engineering produced an empty DataFrame.")

    # 3. Extract latest single row
    row = df[config.FEATURE_COLS].values[-1:].astype(np.float32)   # shape [1, 12]

    # 4. Raw values needed for direction reconstruction and safety check
    atr_ratio_val = float(df["atr_ratio"].values[-1])
    return_5_val  = float(df["return_5"].values[-1])

    # 5. Scale
    row_scaled = scaler.transform(row).astype(np.float32)

    # 6. Forward pass
    x = torch.tensor(row_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)                         # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]  # [2]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    # 7. Determine signal string
    if atr_ratio_val > config.EXTREME_VOL_RATIO:
        # Extreme spike guard — likely a news gap or chaotic event.
        # Trading into this is dangerous; override to NEUTRAL.
        warnings.warn(
            f"[Chat 5] ATR14/ATR50 = {atr_ratio_val:.2f} exceeds EXTREME_VOL_RATIO "
            f"({config.EXTREME_VOL_RATIO}). Overriding to NEUTRAL (news-gap guard)."
        )
        signal_str = "NEUTRAL"
        confidence = min(confidence, 0.50)

    elif pred_idx == 0:   # DIRECTIONAL — reconstruct UP / DOWN from 5-bar return
        signal_str = "UP" if return_5_val > 0 else "DOWN"

    else:                 # NEUTRAL
        signal_str = "NEUTRAL"

    # 8. Build output dict — contract identical across all 7 Chats
    output = {
        "model":      "volatility_risk",
        "asset":      "XAU/USD",
        "signal":     signal_str,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return output


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_signal()
    print("\n=== Chat 5 Signal Output ===")
    for k, v in result.items():
        print(f"  {k}: {v}")