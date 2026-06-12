# chat3_macro_position/predict.py
# Runtime inference for Chat 3 — Macro/World Position Model.
# Loads trained model + scaler, fetches latest macro data,
# and returns the standardised signal dict for the Layer 2 aggregator.
#
# Usage (standalone):
#   python predict.py
#
# Usage (imported by aggregator):
#   from chat3_macro_position.predict import get_signal

import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import joblib

import config
from data import build_feature_dataframe, FEATURE_COLS


# ── Load model + scaler ────────────────────────────────────────────────────────

def load_model_and_scaler():
    from model import MacroPositionModel

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
    model  = MacroPositionModel().to(device)
    ckpt   = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler, device


# ── Direction reconstruction ───────────────────────────────────────────────────

def reconstruct_direction(row: pd.Series) -> tuple[str, bool]:
    """
    Score macro factors bullish (+1) or bearish (-1) for Gold.
    Returns (direction_str, is_ambiguous).

    Given the model's DIRECTIONAL F1 is modest (~0.22), this scoring acts
    as a sanity layer: even when the model says DIRECTIONAL, we only commit
    to UP/DOWN if the underlying macro factors agree. Mixed signals -> NEUTRAL.

    Gold bullish factors:
      - real_yield_delta  < 0   (real yields falling)
      - dxy_chg20         < 0   (DXY weakening)
      - fed_delta         < 0   (Fed turning dovish)
      - cpi_surprise      > 0   (inflation surprising upside)
      - spx_ret20         < 0   (risk-off: capital into Gold)

    Gold bearish factors: opposite of each.
    """
    score = 0

    # Real yield trend — strongest single driver (corr=0.15 in diagnostics)
    if row["real_yield_delta"] < -0.10:
        score += 2
    elif row["real_yield_delta"] < 0:
        score += 1
    elif row["real_yield_delta"] > 0.10:
        score -= 2
    elif row["real_yield_delta"] > 0:
        score -= 1

    # DXY trend
    if row["dxy_chg20"] < -1.0:
        score += 2
    elif row["dxy_chg20"] < 0:
        score += 1
    elif row["dxy_chg20"] > 1.0:
        score -= 2
    elif row["dxy_chg20"] > 0:
        score -= 1

    # Fed stance (dovish = bullish for Gold)
    if row["fed_delta"] < -0.10:
        score += 1
    elif row["fed_delta"] > 0.10:
        score -= 1

    # CPI surprise (inflation shock = bullish for Gold)
    if row["cpi_surprise"] > 0.05:
        score += 1
    elif row["cpi_surprise"] < -0.05:
        score -= 1

    # Risk sentiment (risk-off = Gold bullish)
    if row["spx_ret20"] < -3.0:
        score += 1
    elif row["spx_ret20"] > 3.0:
        score -= 1

    is_ambiguous = score == 0

    if score > 0:
        return "UP", is_ambiguous
    elif score < 0:
        return "DOWN", is_ambiguous
    else:
        return "NEUTRAL", True


# ── Main inference function ────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame = None) -> dict:
    """
    Main inference function.

    Args:
        df: Optional pre-built feature DataFrame (from build_feature_dataframe).
            If None, fresh data is fetched from FRED + yfinance.

    Returns:
        dict with keys: model, asset, signal, confidence, timestamp
    """
    model, scaler, device = load_model_and_scaler()

    # 1. Get latest macro features
    if df is None:
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=config.DATA_YEARS * 365)).strftime("%Y-%m-%d")
        df = build_feature_dataframe(start, end)

    if len(df) == 0:
        raise ValueError("Feature DataFrame is empty — check FRED API key and data fetch.")

    # 2. Take the most recent row
    latest_row = df[FEATURE_COLS].iloc[-1]
    features   = latest_row.values.astype(np.float32).reshape(1, -1)   # [1, n_features]

    # 3. Scale
    features_scaled = scaler.transform(features)

    # 4. Forward pass
    x = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)                         # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]  # [2]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    # 5. Reconstruct signal
    if pred_idx == 0:   # DIRECTIONAL
        direction, is_ambiguous = reconstruct_direction(df.iloc[-1])

        if is_ambiguous:
            # Mixed macro signals — fall back to NEUTRAL, cap confidence
            signal_str = "NEUTRAL"
            confidence = min(confidence, 0.55)
        else:
            signal_str = direction
    else:
        signal_str = "NEUTRAL"

    # 6. Build output dict
    output = {
        "model":      "macro_position",
        "asset":      "XAU/USD",
        "signal":     signal_str,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return output


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_signal()
    print("\n=== Chat 3 Signal Output ===")
    for k, v in result.items():
        print(f"  {k}: {v}")