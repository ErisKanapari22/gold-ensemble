# chat3_macro_position/data.py
# Downloads macro data from FRED + yfinance, engineers features,
# generates binary labels, scales, and returns DataLoaders.

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from fredapi import Fred
from datetime import datetime, timedelta

import config


# ── Dataset ────────────────────────────────────────────────────────────────────

class MacroDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── FRED Fetcher ───────────────────────────────────────────────────────────────

def fetch_fred_series(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series and return as a named daily-indexed Series."""
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.name = series_id
    return s


# ── Feature Engineering ────────────────────────────────────────────────────────

def build_feature_dataframe(start: str, end: str) -> pd.DataFrame:
    """
    Downloads all raw data, aligns to a daily index, and engineers features.
    Monthly series are forward-filled and lagged by 1 period to prevent leakage.
    """
    if not config.FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY is not set. Add it to your .env file at the repo root.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=config.FRED_API_KEY)

    print("Fetching FRED data...")
    fed_funds    = fetch_fred_series(fred, config.FRED_SERIES["fed_funds"],    start, end)
    cpi          = fetch_fred_series(fred, config.FRED_SERIES["cpi"],          start, end)
    yield_10y    = fetch_fred_series(fred, config.FRED_SERIES["yield_10y"],    start, end)
    yield_2y     = fetch_fred_series(fred, config.FRED_SERIES["yield_2y"],     start, end)
    yield_spread = fetch_fred_series(fred, config.FRED_SERIES["yield_spread"], start, end)

    print("Fetching yfinance data (DXY, SPX, Gold)...")
    dxy_raw  = yf.download(config.DXY_TICKER, start=start, end=end,
                           interval="1d", auto_adjust=True, progress=False)
    spx_raw  = yf.download(config.SPX_TICKER, start=start, end=end,
                           interval="1d", auto_adjust=True, progress=False)
    gold_raw = yf.download(config.GOLD_TICKER, start=start, end=end,
                           interval="1d", auto_adjust=True, progress=False)

    # Flatten MultiIndex if present
    for df in [dxy_raw, spx_raw, gold_raw]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Build a daily business-day index aligned to Gold trading days
    daily_idx = gold_raw.index

    def to_daily(series: pd.Series, lag: bool = False) -> pd.Series:
        """Reindex a series to daily, forward-fill gaps, optionally lag 1 period."""
        s = series.reindex(daily_idx, method="ffill")
        if lag:
            s = s.shift(1)   # use last known value — never leak current period
        return s

    # ── Assemble raw daily columns ─────────────────────────────────────────────
    df = pd.DataFrame(index=daily_idx)

    # Gold close (used for labels, not as a feature)
    df["gold_close"] = gold_raw["Close"].squeeze()

    # Daily yields — no lag needed (same-day public data)
    df["yield_10y"]    = to_daily(yield_10y)
    df["yield_2y"]     = to_daily(yield_2y)
    df["yield_spread"] = to_daily(yield_spread)

    # Monthly macro — lag by 1 period (previous month's reading)
    df["fed_funds"]  = to_daily(fed_funds, lag=True)
    df["cpi_raw"]    = to_daily(cpi,       lag=True)

    # DXY and SPX
    df["dxy"]  = dxy_raw["Close"].squeeze().reindex(daily_idx, method="ffill")
    df["spx"]  = spx_raw["Close"].squeeze().reindex(daily_idx, method="ffill")

    df.dropna(inplace=True)

    # ── Engineer features ──────────────────────────────────────────────────────

    # 1. CPI YoY %
    df["cpi_yoy"] = df["cpi_raw"].pct_change(252) * 100   # ~252 trading days ≈ 1 year

    # 2. CPI surprise proxy: deviation from its own 60-day rolling mean
    df["cpi_surprise"] = df["cpi_raw"] - df["cpi_raw"].rolling(60).mean()

    # 3. Real yield = 10Y nominal − CPI YoY
    df["real_yield"] = df["yield_10y"] - df["cpi_yoy"]

    # 4. Real yield 3-month delta (63 trading days)
    df["real_yield_delta"] = df["real_yield"].diff(63)

    # 5. 10Y yield 20-day delta
    df["yield_10y_delta"] = df["yield_10y"].diff(20)

    # 6. Yield spread direction: positive = normal, negative = inverted
    df["spread_delta"] = df["yield_spread"].diff(20)

    # 7. Fed Funds 3-month delta (hawkish/dovish trend)
    df["fed_delta"] = df["fed_funds"].diff(63)

    # 8. DXY 20-day % change
    df["dxy_chg20"] = df["dxy"].pct_change(20) * 100

    # 9. DXY value (normalized later by scaler)
    df["dxy_level"] = df["dxy"]

    # 10. SPX 20-day return (risk-on / risk-off proxy)
    df["spx_ret20"] = df["spx"].pct_change(20) * 100

    # 11. Fed Funds level
    df["fed_level"] = df["fed_funds"]

    # 12. 10Y yield level
    df["yield_10y_level"] = df["yield_10y"]

    # 13. 2Y yield level
    df["yield_2y_level"] = df["yield_2y"]

    # 14. Yield spread level
    df["spread_level"] = df["yield_spread"]

    # 15. Real yield level (redundant with real_yield but kept for symmetry)
    df["real_yield_level"] = df["real_yield"]

    df.dropna(inplace=True)
    df.reset_index(inplace=True)       # bring Date back as a column
    df.rename(columns={"index": "Date", "Date": "Date"}, inplace=True)

    return df


# ── Feature columns (must match config.N_FEATURES = 15) ───────────────────────

FEATURE_COLS = [
    "cpi_yoy",
    "cpi_surprise",
    "real_yield",
    "real_yield_delta",
    "real_yield_level",
    "yield_10y_delta",
    "yield_10y_level",
    "yield_2y_level",
    "spread_level",
    "spread_delta",
    "fed_level",
    "fed_delta",
    "dxy_level",
    "dxy_chg20",
    "spx_ret20",
]


# ── Label Generation ───────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Binary label based on Gold 20-day forward return.
      DIRECTIONAL (0): |forward_return| > LABEL_THRESHOLD AND > ATR_FILTER * 20d ATR
      NEUTRAL     (1): everything else
    """
    close = df["gold_close"].values
    n     = len(close)
    labels = np.ones(n, dtype=np.int64)   # default: NEUTRAL

    # Rolling 20-day ATR proxy (using close-to-close std as a simple daily vol estimate)
    gold_series = pd.Series(close)
    daily_ret   = gold_series.pct_change().abs()
    atr20       = daily_ret.rolling(20).mean() * close   # price units

    for i in range(n - config.FORWARD_DAYS):
        fwd_ret = (close[i + config.FORWARD_DAYS] - close[i]) / close[i]
        atr_val = atr20.iloc[i] if not np.isnan(atr20.iloc[i]) else 0.0
        min_move = config.ATR_FILTER * atr_val * config.FORWARD_DAYS

        if abs(fwd_ret) > config.LABEL_THRESHOLD and abs(fwd_ret * close[i]) > min_move:
            labels[i] = 0   # DIRECTIONAL

    # Last FORWARD_DAYS rows have no valid forward label — mark NEUTRAL (will be trimmed)
    labels[n - config.FORWARD_DAYS:] = 1
    return labels


# ── Main Entry Point ───────────────────────────────────────────────────────────

def get_dataloaders():
    """
    Full pipeline: fetch → engineer → label → scale → split → DataLoaders.
    Caches the raw feature DataFrame to CSV on first run.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=config.DATA_YEARS * 365)).strftime("%Y-%m-%d")

    # Load from cache if available
    if os.path.exists(config.DATA_CACHE_PATH):
        print(f"Loading cached data from {config.DATA_CACHE_PATH}")
        df = pd.read_csv(config.DATA_CACHE_PATH, parse_dates=["Date"])
    else:
        df = build_feature_dataframe(start, end)
        df.to_csv(config.DATA_CACHE_PATH, index=False)
        print(f"Data cached -> {config.DATA_CACHE_PATH}")

    # Labels
    labels = generate_labels(df)

    # Trim last FORWARD_DAYS rows (no valid forward label)
    df     = df.iloc[: len(df) - config.FORWARD_DAYS].reset_index(drop=True)
    labels = labels[: len(df)]

    n_total = len(labels)
    print(f"\nLabel distribution (threshold={config.LABEL_THRESHOLD*100:.1f}%, forward={config.FORWARD_DAYS}d):")
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"  {name:>12}: {count:5d}  ({count / n_total * 100:.1f}%)")

    # Feature matrix
    feature_matrix = df[FEATURE_COLS].values.astype(np.float32)

    # Chronological split
    split      = int(n_total * config.TRAIN_RATIO)
    train_feat = feature_matrix[:split]
    val_feat   = feature_matrix[split:]
    train_lbl  = labels[:split]
    val_lbl    = labels[split:]

    # Scale — fit only on train
    scaler     = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    val_feat   = scaler.transform(val_feat)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved -> {config.SCALER_PATH}")

    # DataLoaders
    train_loader = DataLoader(
        MacroDataset(train_feat, train_lbl),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        MacroDataset(val_feat, val_lbl),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train samples: {len(train_lbl)} | Val samples: {len(val_lbl)}\n")
    return train_loader, val_loader


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_dataloaders()