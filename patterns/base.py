import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PatternSignal:
    """Structured output from a pattern recognizer."""
    
    pattern_id: str
    timeframe: str
    direction: int
    confidence: float
    regime_compatibility: float
    historical_expectancy_R:  float
    historical_win_rate: float
    sample_size: int
    timestamp: pd.Timestamp
    setup_type: str
    
    def to_dict(self):
        """Convert to dict for logging/storage."""
        return {
            "pattern_id": self.pattern_id,
            "timeframe":  self.timeframe,
            "direction": self.direction,
            "confidence": self.confidence,
            "regime_compatibility": self. regime_compatibility,
            "historical_expectancy_R": self. historical_expectancy_R,
            "historical_win_rate":  self.historical_win_rate,
            "sample_size": self.sample_size,
            "timestamp": str(self.timestamp),
            "setup_type": self.setup_type,
        }


def validate_features_length(features: pd.DataFrame, min_length: int) -> bool:
    """Validate that features DataFrame has sufficient data."""
    return not features.empty and len(features) >= min_length


def get_price_extremes(features: pd.DataFrame, lookback: int) -> tuple[float, float]:
    """Get highest high and lowest low over lookback period."""
    high_vals = features["high"].iloc[-lookback:].values
    low_vals = features["low"].iloc[-lookback:].values
    return np.max(high_vals), np.min(low_vals)


def detect_breakout(current_high: float, current_low: float, prior_close: float,
                   structure_high: float, structure_low: float) -> tuple[bool, bool]:
    """Detect bullish or bearish breakout.
    
    Returns:
        tuple: (bullish_breakout, bearish_breakout)
    """
    bullish_breakout = current_high > structure_high and prior_close < structure_high
    bearish_breakout = current_low < structure_low and prior_close > structure_low
    return bullish_breakout, bearish_breakout


class Pattern:
    """Base class for all pattern recognition modules."""
    
    def __init__(self, pattern_id: str):
        self.pattern_id = pattern_id
        self.logger = logging.getLogger(f"{__name__}.{pattern_id}")
    
    def recognize(self, instrument: str, timeframe: str,
                  features: pd.DataFrame, context: dict) -> Optional[PatternSignal]:
        """Override in subclasses."""
        pass
    
    def _get_latest_value(self, features: pd.DataFrame, column: str) -> Optional[float]:
        """Safely retrieve latest value."""
        if features. empty or column not in features.columns:
            return None
        val = features[column].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_sma(self, data: np.ndarray, period: int) -> Optional[float]:
        """Compute SMA."""
        if len(data) < period:
            return None
        return float(np.mean(data[-period:]))
    
    def _get_percentile(self, data: np. ndarray, period: int) -> Optional[float]: 
        """Compute percentile rank."""
        if len(data) < period:
            return None
        window = data[-period:]
        latest = data[-1]
        rank = (window < latest).sum() / period * 100
        return float(rank)