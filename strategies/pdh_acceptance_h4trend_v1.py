"""Strategy V1: PDH/PDL acceptance with H4 trend confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyV1Config:
    """Configuration for Strategy V1."""

    cost_pips_roundtrip: float = 1.2
    max_hold_bars: int = 48
    quiet: bool = False


def _log(level: int, message: str, quiet: bool) -> None:
    if not quiet:
        logger.log(level, message)


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def rolling_percentile(series: pd.Series, lookback: int = 252) -> pd.Series:
    def percentile_rank(window: pd.Series) -> float:
        ranks = window.rank(pct=True)
        return float(ranks.iloc[-1] * 100)

    return series.rolling(lookback).apply(percentile_rank, raw=False)


def compute_h4_context(h4_df: pd.DataFrame) -> pd.DataFrame:
    h4 = h4_df.copy()
    h4["ema50"] = compute_ema(h4["close"], 50)
    h4["ema200"] = compute_ema(h4["close"], 200)
    h4["slope"] = h4["ema50"] - h4["ema50"].shift(3)
    h4["trend_dir"] = np.where(
        (h4["ema50"] > h4["ema200"]) & (h4["slope"] > 0),
        "LONG",
        np.where(
            (h4["ema50"] < h4["ema200"]) & (h4["slope"] < 0),
            "SHORT",
            "NONE",
        ),
    )
    h4["atr14"] = compute_atr(h4, 14)
    h4["atr_pct"] = rolling_percentile(h4["atr14"], 252)
    h4["vol_ok"] = h4["atr_pct"].between(40, 80, inclusive="both")
    return h4


def merge_h4_context(h1_df: pd.DataFrame, h4_context: pd.DataFrame) -> pd.DataFrame:
    h1 = h1_df.copy()
    h4_context = h4_context[
        ["time", "ema50", "ema200", "slope", "trend_dir", "atr_pct", "vol_ok"]
    ].sort_values("time")
    merged = pd.merge_asof(
        h1.sort_values("time"),
        h4_context,
        on="time",
        direction="backward",
    )
    return merged


def map_prior_day_levels(h1_df: pd.DataFrame, d1_df: pd.DataFrame) -> pd.DataFrame:
    d1 = d1_df[["time", "high", "low"]].copy().sort_values("time")
    d1["pdh"] = d1["high"].shift(1)
    d1["pdl"] = d1["low"].shift(1)
    d1_levels = d1[["time", "pdh", "pdl"]]
    merged = pd.merge_asof(
        h1_df.sort_values("time"),
        d1_levels,
        on="time",
        direction="backward",
    )
    return merged


def prepare_strategy_data(
    h1_df: pd.DataFrame, h4_df: pd.DataFrame, d1_df: pd.DataFrame
) -> pd.DataFrame:
    h4_context = compute_h4_context(h4_df)
    h1_with_h4 = merge_h4_context(h1_df, h4_context)
    h1_with_levels = map_prior_day_levels(h1_with_h4, d1_df)
    return h1_with_levels


def _signal_at_index(h1: pd.DataFrame, idx: int) -> str | None:
    if idx + 1 >= len(h1):
        return None
    row = h1.iloc[idx]
    next_row = h1.iloc[idx + 1]

    if row["trend_dir"] == "LONG" and row["vol_ok"]:
        if row["high"] > row["pdh"]:
            if row["close"] > row["pdh"] and next_row["close"] > row["pdh"]:
                return "LONG"

    if row["trend_dir"] == "SHORT" and row["vol_ok"]:
        if row["low"] < row["pdl"]:
            if row["close"] < row["pdl"] and next_row["close"] < row["pdl"]:
                return "SHORT"

    return None


def find_signal_indices(h1: pd.DataFrame) -> List[Tuple[int, str]]:
    signals: List[Tuple[int, str]] = []
    for idx in range(len(h1) - 1):
        direction = _signal_at_index(h1, idx)
        if direction:
            signals.append((idx, direction))
    return signals


def _evaluate_intrabar_exit(
    direction: str, bar: pd.Series, sl: float, tp: float
) -> Tuple[str | None, float | None]:
    if direction == "LONG":
        hit_sl = bar["low"] <= sl
        hit_tp = bar["high"] >= tp
        if hit_sl and hit_tp:
            return "SL", sl
        if hit_sl:
            return "SL", sl
        if hit_tp:
            return "TP", tp
    else:
        hit_sl = bar["high"] >= sl
        hit_tp = bar["low"] <= tp
        if hit_sl and hit_tp:
            return "SL", sl
        if hit_sl:
            return "SL", sl
        if hit_tp:
            return "TP", tp
    return None, None


def simulate_trades(
    h1: pd.DataFrame, config: StrategyV1Config
) -> List[Dict[str, float]]:
    trades: List[Dict[str, float]] = []
    cost_price = config.cost_pips_roundtrip * 0.0001
    last_trade_date = None

    idx = 0
    while idx < len(h1) - 1:
        direction = _signal_at_index(h1, idx)
        if not direction:
            idx += 1
            continue

        entry_time = h1.iloc[idx + 1]["time"]
        entry_date = pd.to_datetime(entry_time).date()
        if last_trade_date == entry_date:
            idx += 1
            continue

        entry_price = h1.iloc[idx + 1]["close"]
        if direction == "LONG":
            sweep_extreme = min(h1.iloc[idx]["low"], h1.iloc[idx + 1]["low"])
            sl = sweep_extreme - 0.0002
        else:
            sweep_extreme = max(h1.iloc[idx]["high"], h1.iloc[idx + 1]["high"])
            sl = sweep_extreme + 0.0002

        risk = abs(entry_price - sl)
        if risk == 0:
            idx += 1
            continue

        tp = (
            entry_price + 2.5 * risk
            if direction == "LONG"
            else entry_price - 2.5 * risk
        )

        _log(
            logging.DEBUG,
            f"Entry {direction} at {entry_time} price {entry_price:.5f}",
            config.quiet,
        )

        entry_idx = idx + 1
        exit_reason = None
        exit_price = None
        exit_time = None
        exit_idx = None

        last_index = len(h1) - 1
        max_exit_idx = min(entry_idx + config.max_hold_bars, last_index)
        for bar_idx in range(entry_idx, max_exit_idx + 1):
            bar = h1.iloc[bar_idx]
            exit_reason, exit_price = _evaluate_intrabar_exit(
                direction, bar, sl, tp
            )
            if exit_reason:
                exit_time = bar["time"]
                exit_idx = bar_idx
                break

            if direction == "LONG":
                regime_ok = bar["trend_dir"] == "LONG"
            else:
                regime_ok = bar["trend_dir"] == "SHORT"

            if not regime_ok:
                exit_reason = "REGIME"
                exit_price = bar["close"]
                exit_time = bar["time"]
                exit_idx = bar_idx
                break

        if exit_reason is None:
            exit_reason = "TIME"
            exit_idx = max_exit_idx
            exit_row = h1.iloc[exit_idx]
            exit_time = exit_row["time"]
            exit_price = exit_row["close"]

        if direction == "LONG":
            pnl_price = exit_price - entry_price - cost_price
        else:
            pnl_price = entry_price - exit_price - cost_price

        pnl_pips = pnl_price / 0.0001
        r_multiple = pnl_price / risk
        duration_bars = exit_idx - entry_idx + 1 if exit_idx is not None else 0

        trade = {
            "entry_time": entry_time,
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_price": pnl_price,
            "pnl_pips": pnl_pips,
            "R_multiple": r_multiple,
            "duration_bars": duration_bars,
            "pdh": h1.iloc[entry_idx]["pdh"],
            "pdl": h1.iloc[entry_idx]["pdl"],
            "atr_pct": h1.iloc[entry_idx]["atr_pct"],
            "ema50": h1.iloc[entry_idx]["ema50"],
            "ema200": h1.iloc[entry_idx]["ema200"],
            "slope": h1.iloc[entry_idx]["slope"],
        }
        trades.append(trade)
        last_trade_date = entry_date
        idx = exit_idx + 1 if exit_idx is not None else entry_idx + 1

    return trades


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "avg_R": 0.0,
            "expectancy_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
            "avg_hold_bars": 0.0,
        }

    winrate = float((trades["pnl_price"] > 0).mean())
    avg_r = float(trades["R_multiple"].mean())
    pos_r = trades.loc[trades["R_multiple"] > 0, "R_multiple"].sum()
    neg_r = trades.loc[trades["R_multiple"] < 0, "R_multiple"].sum()
    profit_factor = float(pos_r / abs(neg_r)) if neg_r != 0 else float("inf")

    equity = trades["R_multiple"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "trades": int(len(trades)),
        "winrate": winrate,
        "avg_R": avg_r,
        "expectancy_R": avg_r,
        "profit_factor": profit_factor,
        "max_drawdown_R": max_dd,
        "avg_hold_bars": float(trades["duration_bars"].mean()),
    }


def run_strategy_v1(
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    d1_df: pd.DataFrame,
    config: StrategyV1Config | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    config = config or StrategyV1Config()
    _log(logging.INFO, "Preparing data for Strategy V1", config.quiet)
    strategy_data = prepare_strategy_data(h1_df, h4_df, d1_df)
    _log(logging.INFO, "Simulating Strategy V1 trades", config.quiet)
    trades = simulate_trades(strategy_data, config)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df)
    return trades_df, summary
