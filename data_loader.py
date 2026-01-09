"""Data loading utilities for EUR/USD timeframes."""

from __future__ import annotations

import logging
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _load_price_data(path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV or Parquet."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV dataframe to a new timeframe."""
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.set_index("time")
    resampled = (
        data.resample(rule)
        .agg(
            open="first",
            high="max",
            low="min",
            close="last",
            volume="sum",
        )
        .dropna()
        .reset_index()
    )
    return resampled


def load_eurusd_timeframes(
    m5_csv_path: str = "data/EUR_USD_M5.csv",
) -> Dict[str, pd.DataFrame]:
    """Load EUR/USD price data across timeframes."""
    if os.path.exists(m5_csv_path):
        m5_df = _load_price_data(m5_csv_path)
    else:
        parquet_path = m5_csv_path.replace(".csv", ".parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"No data found at {m5_csv_path} or {parquet_path}")
        logger.info("CSV not found. Falling back to %s", parquet_path)
        m5_df = _load_price_data(parquet_path)

    timeframes = {"M5": m5_df}
    for tf, rule in [("H1", "1H"), ("H4", "4H"), ("D1", "1D")]:
        parquet_name = f"data/EUR_USD_{tf}.parquet"
        if tf == "D1":
            parquet_name_alt = "data/EUR_USD_D.parquet"
        else:
            parquet_name_alt = None

        if os.path.exists(parquet_name):
            timeframes[tf] = _load_price_data(parquet_name)
            continue
        if parquet_name_alt and os.path.exists(parquet_name_alt):
            timeframes[tf] = _load_price_data(parquet_name_alt)
            continue

        logger.info("Resampling M5 data to %s", tf)
        timeframes[tf] = _resample_ohlc(m5_df, rule)

    return timeframes
