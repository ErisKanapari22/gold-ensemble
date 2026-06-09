# data.py
# Fetches XAU/USD OHLCV data, engineers features, generates labels,
# scales features (train-only fit), and returns DataLoaders.

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import ta

import config


# ── Dataset ───────────────────────────────────────────────────────────────────

class GoldSequenceDataset(Dataset):
    """
    Sliding-window dataset. Each sample is a (LOOKBACK, N_FEATURES) tensor
    and a class label: 0=UP, 1=DOWN, 2=NEUTRAL.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and returns one with engineered feature columns.
    Uses the `ta` library for indicator calculations.
    """
    df = df.copy()

    # Flatten MultiIndex columns if present (yfinance ≥0.2 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure Series are 1-D (squeeze out any extra dimension)
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # Returns (log)
    df["returns"] = np.log(close / close.shift(1))

    # Trend
    df["sma20"] = ta.trend.sma_indicator(close, window=20)
    df["ema9"]  = ta.trend.ema_indicator(close, window=9)

    # Momentum
    df["rsi14"] = ta.momentum.rsi(close, window=14)

    # MACD
    macd_obj = ta.trend.MACD(close)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # Volatility
    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    # Drop rows with NaN from indicator warmup
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_labels(df: pd.DataFrame, threshold: float, atr_filter: float):
    """
    Binary labeling: DIRECTIONAL (0) vs NEUTRAL (1).

    A candle is DIRECTIONAL when both conditions hold:
      1. |pct move| exceeds `threshold`
      2. absolute price move exceeds `atr_filter * ATR14`  (filters tick noise)

    Also returns a `directions` array (+1=UP, -1=DOWN, 0=NEUTRAL) so predict.py
    can reconstruct the original UP/DOWN/NEUTRAL signal from the binary prediction.

    Returns:
        labels:     np.ndarray — 0=DIRECTIONAL, 1=NEUTRAL
        directions: np.ndarray — +1=UP, -1=DOWN, 0=NEUTRAL
    """
    close      = df["Close"].values
    atr_values = df["atr14"].values
    labels     = []
    directions = []
    for i in range(len(close) - 1):
        pct      = (close[i + 1] - close[i]) / close[i]
        min_move = atr_filter * atr_values[i]
        if abs(pct) > threshold and abs(pct * close[i]) > min_move:
            labels.append(0)                            # DIRECTIONAL
            directions.append(1 if pct > 0 else -1)    # +1=UP, -1=DOWN
        else:
            labels.append(1)                            # NEUTRAL
            directions.append(0)
    return np.array(labels), np.array(directions)


# ── Sliding Window ────────────────────────────────────────────────────────────

def build_sequences(features: np.ndarray, labels: np.ndarray, lookback: int):
    """
    Convert flat feature array into overlapping windows.
    Each window X[i] = features[i : i+lookback], label = labels[i+lookback-1]
    """
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i + lookback])
        y.append(labels[i + lookback - 1])
    return np.array(X), np.array(y)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_dataloaders(csv_path: str = None):
    """
    Full pipeline: fetch → engineer → label → scale → split → DataLoaders.
    Pass csv_path to load from a local file instead of downloading.

    Returns:
        train_loader, val_loader, n_features (int)
    """
    # 1. Fetch data
    if csv_path:
        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        raw.columns = [c.capitalize() for c in raw.columns]
    else:
        print(f"Downloading {config.TICKER} ({config.TIMEFRAME}) ...")
        raw = yf.download(
            config.TICKER,
            period="2y",
            interval=config.TIMEFRAME,
            auto_adjust=True,
            progress=False,
        )
        # yfinance ≥0.2.x returns a MultiIndex columns (Price, Ticker).
        # Flatten to single-level so ta indicators receive 1-D Series.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.to_csv(config.DATA_CACHE_PATH)

    # 2. Engineer features
    df = engineer_features(raw)

    # 3. Labels (needs Close; computed before we drop the price columns)
    labels_full, directions_full = generate_labels(df, config.LABEL_THRESHOLD, config.ATR_FILTER)
    n_total = len(labels_full)
    print(f"Label distribution (threshold=+-{config.LABEL_THRESHOLD*100:.1f}%, ATR_FILTER={config.ATR_FILTER}):")
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels_full == cls).sum())
        print(f"  {name:>12}: {count:5d}  ({count / n_total * 100:.1f}%)")
    # Trim df to match label length (last row has no forward label)
    df = df.iloc[: len(labels_full)].reset_index(drop=True)

    # 4. Extract feature matrix
    feature_matrix = df[config.FEATURE_COLS].values.astype(np.float32)

    # 5. Chronological train/val split on the flat data BEFORE building sequences
    split = int(len(feature_matrix) * config.TRAIN_RATIO)
    train_feat, val_feat = feature_matrix[:split], feature_matrix[split:]
    train_lbl,  val_lbl  = labels_full[:split],    labels_full[split:]

    # 6. Scale — fit ONLY on train, transform both
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    val_feat   = scaler.transform(val_feat)

    # 7. Persist scaler for predict.py
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved -> {config.SCALER_PATH}")

    # 8. Build sliding-window sequences
    X_train, y_train = build_sequences(train_feat, train_lbl, config.LOOKBACK)
    X_val,   y_val   = build_sequences(val_feat,   val_lbl,   config.LOOKBACK)

    print(f"Train sequences: {len(X_train)} | Val sequences: {len(X_val)}")

    # 9. DataLoaders — shuffle train, never shuffle val (time-series order)
    train_loader = DataLoader(
        GoldSequenceDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        GoldSequenceDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader