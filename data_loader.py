"""EUR/USD CSV loader with deterministic resampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from utils.timeframes import TIMEFRAME_RULES, expected_timedelta

logger = logging.getLogger(__name__)

TIME_COLUMNS = ("timestamp", "time", "datetime")


def _detect_time_column(columns: list[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for candidate in TIME_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        f"CSV must include one of {TIME_COLUMNS} columns; found: {columns}"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    time_col = _detect_time_column(df.columns.tolist())
    lower_map = {col.lower(): col for col in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in lower_map]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {lower_map[col]: col for col in required}
    if "volume" in lower_map:
        rename_map[lower_map["volume"]] = "volume"
    rename_map[time_col] = "timestamp"

    normalized = df.rename(columns=rename_map)
    return normalized


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df = df.sort_index()
    duplicate_count = df.index.duplicated().sum()
    if duplicate_count:
        logger.warning("Dropping %s duplicate timestamps", duplicate_count)
        df = df[~df.index.duplicated(keep="first")]
    return df


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    violations = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    violation_count = int(violations.sum())
    if violation_count:
        logger.warning("Dropping %s rows with invalid OHLC", violation_count)
        df = df.loc[~violations]
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    resampled = df.resample(rule, label="right", closed="right").agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled.index.name = "timestamp"
    return resampled


def load_eurusd_timeframes(
    m5_csv_path: str | Path = "data/EUR_USD_M5.csv",
) -> Dict[str, pd.DataFrame]:
    """Load EUR/USD M5 data and resample into higher timeframes."""
    m5_csv_path = Path(m5_csv_path)
    df = pd.read_csv(m5_csv_path)
    df = _normalize_columns(df)
    df = _coerce_timestamp(df)
    df = _validate_ohlc(df)

    data = {"M5": df}
    data["H1"] = _resample_ohlc(df, TIMEFRAME_RULES["H1"])
    data["H4"] = _resample_ohlc(df, TIMEFRAME_RULES["H4"])
    data["D1"] = _resample_ohlc(df, TIMEFRAME_RULES["D1"])
    return data


def _print_summary(data: Dict[str, pd.DataFrame]) -> None:
    for timeframe, df in data.items():
        if df.empty:
            print(f"{timeframe}: 0 rows")
            continue
        start_ts = df.index.min()
        end_ts = df.index.max()
        print(f"{timeframe}: {len(df)} rows | {start_ts} -> {end_ts}")

    for timeframe in ("H1", "H4", "D1"):
        df = data[timeframe]
        print(f"\n{timeframe} head(3):")
        print(df.head(3))
        print(f"\n{timeframe} tail(3):")
        print(df.tail(3))

    for timeframe, df in data.items():
        if df.empty:
            continue
        expected = expected_timedelta(timeframe)
        gaps = df.index.to_series().diff().dropna()
        oversized = gaps[gaps > expected]
        if not oversized.empty:
            logger.warning(
                "%s timeframe has %s gaps larger than %s",
                timeframe,
                len(oversized),
                expected,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frames = load_eurusd_timeframes()
    _print_summary(frames)
