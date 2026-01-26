from __future__ import annotations

import pandas as pd

from utils.timeframes import merge_higher_timeframes


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept common FX candle column variants and normalize to:
    timestamp, open, high, low, close
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    ts_col = cols_lower.get("timestamp") or cols_lower.get("time") or cols_lower.get("date") or cols_lower.get("datetime")
    if ts_col is None:
        raise ValueError(f"CSV must contain a timestamp/time/date column. Found: {list(df.columns)}")

    df = df.rename(columns={ts_col: "timestamp"})

    for name in ["open", "high", "low", "close"]:
        src = cols_lower.get(name)
        if src is None:
            if name not in df.columns:
                raise ValueError(f"CSV must contain {name}. Found: {list(df.columns)}")
        else:
            df = df.rename(columns={src: name})

    return df


def _coerce_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df.index.name = "timestamp"

    # Drop duplicate timestamps deterministically
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    return df


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    if bad.any():
        df = df.loc[~bad]
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Deterministic OHLC resampling. No lookahead (uses first/max/min/last).
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    resampled = df.resample(rule, closed="right", label="left").agg(agg).dropna()
    resampled.index.name = "timestamp"
    return resampled


def load_eurusd_timeframes(m5_csv_path: str = "data/EUR_USD_M5.csv") -> dict[str, pd.DataFrame]:
    """
    Load EUR/USD M5 CSV and deterministically build:
      - M5 base
      - H1 via resample
      - H4 via resample
      - D1 via resample
    Then merges H4/D1 context INTO H1 using backward-only asof alignment
    (via utils.timeframes.merge_higher_timeframes) to prevent lookahead.
    """
    raw = pd.read_csv(m5_csv_path)
    raw = _standardize_columns(raw)
    raw = _coerce_timestamp_index(raw)
    raw = _validate_ohlc(raw)

    m5 = raw[["open", "high", "low", "close"]].copy()

    h1 = _resample_ohlc(m5, "1H")
    h4 = _resample_ohlc(m5, "4H")
    d1 = _resample_ohlc(m5, "1D")

    # Backward-only alignment: each H1 bar gets the most recent completed H4/D1 bar
    h1 = merge_higher_timeframes(h1, h4, d1)

    return {"M5": m5, "H1": h1, "H4": h4, "D1": d1}


if __name__ == "__main__":
    frames = load_eurusd_timeframes()
    for k, v in frames.items():
        print(k, len(v), v.index.min(), "->", v.index.max())
