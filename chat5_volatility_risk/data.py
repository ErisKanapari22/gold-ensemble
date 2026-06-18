# data.py
# Fetches GC=F (1H) and ^VIX (daily, forward-filled), engineers all 12 volatility
# features, generates dual-condition DIRECTIONAL/NEUTRAL labels, and returns
# DataLoaders with WeightedRandomSampler.
# Also exposes build_features_and_labels() for diagnose.py.

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

class VolatilityDataset(Dataset):
    """Single-row tabular samples — no sequence window needed."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_gold() -> pd.DataFrame:
    """Download 1H GC=F data (2y) or return cached CSV."""
    if os.path.exists(config.DATA_CACHE_PATH):
        print(f"Loading Gold data from cache: {config.DATA_CACHE_PATH}")
        return pd.read_csv(config.DATA_CACHE_PATH, index_col=0, parse_dates=True)

    print(f"Downloading {config.TICKER} (1H, 2y) ...")
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
    print(f"  Cached -> {config.DATA_CACHE_PATH}")
    return raw


def fetch_vix() -> pd.Series:
    """Download daily VIX data (2y) or return cached CSV."""
    if os.path.exists(config.VIX_CACHE_PATH):
        print(f"Loading VIX data from cache: {config.VIX_CACHE_PATH}")
        df = pd.read_csv(config.VIX_CACHE_PATH, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].squeeze()

    print(f"Downloading {config.VIX_TICKER} (1d, 2y) ...")
    raw = yf.download(
        config.VIX_TICKER,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.to_csv(config.VIX_CACHE_PATH)
    print(f"  Cached -> {config.VIX_CACHE_PATH}")
    return raw["Close"].squeeze()


# ── VIX Alignment ─────────────────────────────────────────────────────────────

def _align_vix(vix_series: pd.Series, target_index) -> pd.Series:
    """
    Forward-fill daily VIX into a 1H Gold index.
    Converts both sides to tz-naive UTC midnight dates before reindexing,
    handling mixed-timezone strings that yfinance/CSV round-trips can produce.
    """
    vix_dates = pd.to_datetime(vix_series.index, utc=True).normalize().tz_localize(None)
    vix_clean = pd.Series(vix_series.values, index=vix_dates)
    vix_clean = vix_clean[~vix_clean.index.duplicated(keep="last")]

    gold_dates = pd.to_datetime(target_index, utc=True).normalize().tz_localize(None)

    result = vix_clean.reindex(gold_dates, method="ffill")
    result.index = target_index
    return result


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(gold_df: pd.DataFrame, vix_series: pd.Series) -> pd.DataFrame:
    """
    Builds all 12 features in config.FEATURE_COLS.
    Requires at least ~70 bars for ATR50 + HV60 warmup.
    """
    df = gold_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    open_  = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    # ── ATR features ──────────────────────────────────────────────────────────
    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["atr50"] = ta.volatility.average_true_range(high, low, close, window=50)
    df["atr_ratio"]   = df["atr14"] / df["atr50"].replace(0, np.nan)
    df["atr14_slope"] = df["atr14"].diff(5) / 5.0   # change per bar over 5 bars

    # ── Bollinger Band width ───────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width"]       = bb.bollinger_wband()
    df["bb_width_slope"] = df["bb_width"].diff(5) / 5.0

    # ── Historical volatility (annualised for 1H data) ────────────────────────
    log_ret   = np.log(close / close.shift(1))
    annualise = np.sqrt(252 * 24)              # 1H bars: 252 trading days × 24 h
    hv60      = log_ret.rolling(60).std() * annualise
    df["hv20"]    = log_ret.rolling(20).std() * annualise
    df["hv_ratio"] = df["hv20"] / hv60.replace(0, np.nan)

    # ── Gap ratio (overnight / news gap risk) ─────────────────────────────────
    prev_close     = close.shift(1)
    df["gap_ratio"] = (open_ - prev_close).abs() / df["atr14"].replace(0, np.nan)

    # ── Volume ratio ──────────────────────────────────────────────────────────
    vol_ma20        = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # ── VIX (forward-filled daily → 1H) ──────────────────────────────────────
    df["vix"] = _align_vix(vix_series, df.index).values

    # ── 5-bar price return (direction + energy proxy) ─────────────────────────
    df["return_5"] = (close - close.shift(5)) / close.shift(5)

    # Drop NaN from indicator warmup
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Label Generation ──────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> np.ndarray:
    """
    DIRECTIONAL (0): ATR14/ATR50 > VOL_THRESHOLD  AND  |5-bar return| > RETURN_THRESHOLD
    NEUTRAL (1):     everything else

    Dual-condition gate mirrors Chat 1's ATR_FILTER philosophy:
    a vol spike with no accompanying price move is noise, not signal.
    """
    atr_ratio = df["atr_ratio"].values
    ret5      = df["return_5"].values
    return np.where(
        (atr_ratio > config.VOL_THRESHOLD) & (np.abs(ret5) > config.RETURN_THRESHOLD),
        0,   # DIRECTIONAL
        1,   # NEUTRAL
    )


# ── Main Export ───────────────────────────────────────────────────────────────

def build_features_and_labels():
    """
    Full data pipeline without DataLoaders.
    Returns (feature_matrix [n, 12], labels [n], df) — used by diagnose.py.
    """
    gold = fetch_gold()
    vix  = fetch_vix()
    df   = engineer_features(gold, vix)
    labels = generate_labels(df)

    n_total = len(labels)
    print(f"\nLabel distribution  "
          f"(VOL_THRESHOLD={config.VOL_THRESHOLD}, RETURN_THRESHOLD={config.RETURN_THRESHOLD}):")
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"  {name:>12}: {count:5d}  ({count / n_total * 100:.1f}%)")

    feature_matrix = df[config.FEATURE_COLS].values.astype(np.float32)
    return feature_matrix, labels, df


def get_dataloaders():
    """
    Full pipeline → DataLoaders.
    Returns: train_loader, val_loader
    """
    features, labels, _ = build_features_and_labels()

    # Chronological split — no shuffling
    split   = int(len(features) * config.TRAIN_RATIO)
    X_train = features[:split];  X_val = features[split:]
    y_train = labels[:split];    y_val = labels[split:]

    # Scale: fit ONLY on train, transform both
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved -> {config.SCALER_PATH}")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # WeightedRandomSampler (primary imbalance handler)
    class_counts  = np.bincount(y_train)
    samp_weights  = (1.0 / np.maximum(class_counts, 1))[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(samp_weights, dtype=torch.float32),
        num_samples=len(y_train),
        replacement=True,
    )

    train_loader = DataLoader(
        VolatilityDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
    )
    val_loader = DataLoader(
        VolatilityDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader