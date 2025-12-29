"""feature_engineering.py"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def compute_atr(high:  np.ndarray, low: np. ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = 0
    atr = np.zeros_like(tr)
    atr[period - 1] = tr[:  period].  mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr

def compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)

    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)

    avg_gain[period] = gain[: period].mean()
    avg_loss[period] = loss[:period].mean()

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_loss))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: np.ndarray, fast:  int = 12, slow: int = 26, signal: int = 9):
    """MACD (fast EMA, slow EMA, signal)."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index (simplified)."""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )

    atr_val = compute_atr(high, low, close, period)
    atr_val = np.where(atr_val == 0, 1, atr_val)

    plus_di = 100 * compute_ema(plus_dm, period) / atr_val
    minus_di = 100 * compute_ema(minus_dm, period) / atr_val

    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    di_sum = np.where(di_sum == 0, 1, di_sum)

    dx = 100 * di_diff / di_sum
    adx = compute_ema(dx, period)

    return adx

def compute_bollinger_bands(close: np. ndarray, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close). rolling(period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

class FeatureEngine:
    """Computes all trading features per timeframe."""

    def __init__(self):
        self.feature_cache = {}
        logger.info("FeatureEngine initialized")

    def compute_features(self, instrument: str, timeframe: str, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for an instrument/timeframe."""

        if len(candles) < 200:
            logger.warning(
                f"{instrument} {timeframe}:    insufficient data ({len(candles)} < 200)"
            )
            return pd.DataFrame()

        close = candles["close"].values
        high = candles["high"].values
        low = candles["low"].values
        open_ = candles["open"].values

        features = pd.DataFrame(index=candles.index)
        features["time"] = candles["time"].values
        features["close"] = close
        features["open"] = open_
        features["high"] = high
        features["low"] = low

        # Volatility & Returns
        returns = np.diff(close, prepend=close[0]) / close[0]
        features["returns"] = returns
        features["log_returns"] = np.log(close / np.roll(close, 1))
        features. loc[0, "log_returns"] = 0

        features["volatility_20"] = pd.Series(features["log_returns"]).rolling(20).std().values
        features["volatility_50"] = pd.Series(features["log_returns"]).rolling(50).std().values

        # ATR
        features["atr_14"] = compute_atr(high, low, close, 14)
        features["atr_50"] = compute_atr(high, low, close, 50)

        # EMAs
        features["ema_20"] = compute_ema(close, 20)
        features["ema_50"] = compute_ema(close, 50)
        features["ema_200"] = compute_ema(close, 200)

        # RSI
        features["rsi_14"] = compute_rsi(close, 14)

        # MACD
        macd_line, signal_line, histogram = compute_macd(close, 12, 26, 9)
        features["macd_line"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # ADX
        features["adx_14"] = compute_adx(high, low, close, 14)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, 20, 2.0)
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = bb_upper - bb_lower

        bb_range = bb_upper - bb_lower
        bb_range = np.where(bb_range == 0, 1, bb_range)
        features["bb_zscore"] = (close - bb_middle) / bb_range

        # Range & Wick
        features["range"] = high - low
        features["body"] = np.abs(close - open_)
        features["wick_up"] = high - np.maximum(close, open_)
        features["wick_down"] = np.minimum(close, open_) - low
        features["wick_ratio"] = features["wick_up"] / (features["range"] + 1e-8)

        # Regime
        features["trend_regime"] = np.sign(features["ema_20"] - features["ema_50"])

        atr_pct = pd.Series(features["atr_14"]).rolling(100).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50,
            raw=False
        )
        features["atr_pctl_100"] = atr_pct. values

        self.feature_cache[(instrument, timeframe)] = features
        return features

    def get_features(self, instrument: str, timeframe: str, last_n:  int = None) -> pd.DataFrame:
        """Retrieve cached features."""
        key = (instrument, timeframe)
        if key not in self.feature_cache:
            return pd.DataFrame()

        features = self.feature_cache[key]
        if last_n: 
            return features.tail(last_n).copy()
        return features. copy()