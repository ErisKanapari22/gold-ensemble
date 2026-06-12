# data.py
# Fetches XAU/USD OHLCV data, engineers regime features, generates labels,
# scales features (train-only fit), and returns DataLoaders.
# NOTE: no sliding-window sequences — each row is one independent sample,
# since the target architecture (logistic regression / small MLP) is non-sequential.

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import ta

import config


# ── Dataset ───────────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    """Each sample is a (N_FEATURES,) vector and a binary label."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Raw data fetching / caching ──────────────────────────────────────────────

def fetch_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load OHLCV from local CSV cache if present, else download from Yahoo Finance
    and cache it. Flattens yfinance MultiIndex columns (Chat 1 pattern).
    """
    if use_cache and os.path.exists(config.DATA_CACHE_PATH):
        raw = pd.read_csv(config.DATA_CACHE_PATH, index_col=0, parse_dates=True)
        return raw

    print(f"Downloading {config.TICKER} ({config.TIMEFRAME}) ...")
    raw = yf.download(
        config.TICKER,
        period="2y",
        interval=config.TIMEFRAME,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.to_csv(config.DATA_CACHE_PATH)
    print(f"Cached -> {config.DATA_CACHE_PATH}")
    return raw


# ── Helper: HH/LL structure streak ───────────────────────────────────────────

def compute_hh_ll_streak(high: pd.Series, low: pd.Series) -> np.ndarray:
    """
    Signed streak counter:
      +N -> N consecutive candles with higher-high AND higher-low (uptrend structure)
      -N -> N consecutive candles with lower-high AND lower-low (downtrend structure)
       0 -> structure broken / mixed candle
    """
    high_v = high.values
    low_v = low.values
    streak = np.zeros(len(high_v), dtype=np.float32)

    for i in range(1, len(high_v)):
        higher_high = high_v[i] > high_v[i - 1]
        higher_low  = low_v[i]  > low_v[i - 1]
        lower_high  = high_v[i] < high_v[i - 1]
        lower_low   = low_v[i]  < low_v[i - 1]

        if higher_high and higher_low:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif lower_high and lower_low:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0.0

    return streak


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and returns one with engineered regime features.
    Reuses Chat 1 patterns: flatten MultiIndex, .squeeze(), dropna after warmup.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # ── ADX (core trend-strength indicator) ──
    adx_obj = ta.trend.ADXIndicator(high, low, close, window=config.ADX_PERIOD)
    df["adx14"] = adx_obj.adx()

    # ── EMAs (normalized vs Close, so they're comparable across price levels) ──
    ema20  = ta.trend.ema_indicator(close, window=config.EMA_FAST)
    ema50  = ta.trend.ema_indicator(close, window=config.EMA_MID)
    ema200 = ta.trend.ema_indicator(close, window=config.EMA_SLOW)

    df["ema20_norm"]       = (ema20 - close) / close
    df["ema50_norm"]       = (ema50 - close) / close
    df["ema200_norm"]      = (ema200 - close) / close
    df["ema_spread_20_50"] = (ema20 - ema50) / close   # >0 = bullish alignment, <0 = bearish

    # ── Bollinger Band width (range vs expansion) ──
    bb = ta.volatility.BollingerBands(close, window=config.BB_WINDOW, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()

    # ── ATR normalized (volatility context, scale-independent) ──
    atr = ta.volatility.average_true_range(high, low, close, window=config.ATR_PERIOD)
    df["atr14_norm"] = atr / close

    # ── ADX slope (is trend strength rising or falling) ──
    df["adx_slope5"] = df["adx14"] - df["adx14"].shift(config.ADX_SLOPE_PERIOD)

    # ── Price position within recent N-bar range (0 = at low, 1 = at high) ──
    roll_max = high.rolling(config.RANGE_LOOKBACK).max()
    roll_min = low.rolling(config.RANGE_LOOKBACK).min()
    df["range_position"] = (close - roll_min) / (roll_max - roll_min)

    # ── Higher-high/lower-low structural streak ──
    df["hh_ll_count"] = compute_hh_ll_streak(high, low)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Label Generation ──────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame, adx_threshold: float):
    """
    Binary labeling: DIRECTIONAL (0) vs NEUTRAL (1).

    NOT forward-looking: label[i] is based on the CURRENT candle's ADX14.
      ADX14[i] > adx_threshold -> DIRECTIONAL (0)
      else                     -> NEUTRAL (1)

    Direction for DIRECTIONAL samples is the sign of EMA20-EMA50 spread
    at that same candle (+1=UP trend, -1=DOWN trend, 0=NEUTRAL).

    Returns:
        labels:     np.ndarray — 0=DIRECTIONAL, 1=NEUTRAL
        directions: np.ndarray — +1=UP, -1=DOWN, 0=NEUTRAL
    """
    adx    = df["adx14"].values
    spread = df["ema_spread_20_50"].values

    labels     = np.where(adx > adx_threshold, 0, 1)
    directions = np.where(
        labels == 0,
        np.where(spread > 0, 1, -1),
        0,
    )
    return labels, directions


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_dataloaders(use_cache: bool = True):
    """
    Full pipeline: fetch -> engineer -> label -> scale -> split -> DataLoaders.

    Returns:
        train_loader, val_loader
    """
    raw = fetch_data(use_cache=use_cache)
    df = engineer_features(raw)

    labels, directions = generate_labels(df, config.ADX_THRESHOLD)
    n_total = len(labels)

    print(f"Label distribution (ADX_THRESHOLD={config.ADX_THRESHOLD}):")
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"  {name:>12}: {count:5d}  ({count / n_total * 100:.1f}%)")

    feature_matrix = df[config.FEATURE_COLS].values.astype(np.float32)

    # Chronological split — no shuffling
    split = int(len(feature_matrix) * config.TRAIN_RATIO)
    X_train, X_val = feature_matrix[:split], feature_matrix[split:]
    y_train, y_val = labels[:split], labels[split:]

    # Scale — fit ONLY on train, transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved -> {config.SCALER_PATH}")

    # WeightedRandomSampler for training-set class balance (Chat 3 lesson)
    class_counts = np.bincount(y_train, minlength=config.N_CLASSES).astype(np.float32)
    class_weights_for_sampler = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights_for_sampler[y_train]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        RegimeDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
    )
    val_loader = DataLoader(
        RegimeDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    return train_loader, val_loader


if __name__ == "__main__":
    # Force fresh download + cache (per setup guide step 3)
    fetch_data(use_cache=False)