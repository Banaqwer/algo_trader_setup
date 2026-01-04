import logging
import pandas as pd
import numpy as np
from patterns.base import Pattern, PatternSignal, validate_features_length, get_price_extremes

logger = logging.getLogger(__name__)

class LiquiditySweepPattern(Pattern):
    """Pattern 2: Liquidity Sweep / Stop Hunt."""
    
    def __init__(self):
        super().__init__("LIQUIDITY_SWEEP")
    
    def recognize(self, instrument: str, timeframe: str,
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Detect sweep and reversal."""
        
        if not validate_features_length(features, 30):
            return None
        
        lookback = 20
        high_vals = features["high"].iloc[-lookback:].values
        low_vals = features["low"]. iloc[-lookback:].values
        close = self._get_latest_value(features, "close")
        
        swing_high = np.max(high_vals[:-1])
        swing_low = np. min(low_vals[:-1])
        
        current_high = high_vals[-1]
        current_low = low_vals[-1]
        
        high_swept = current_high > swing_high
        low_swept = current_low < swing_low
        
        if not (high_swept or low_swept):
            return None
        
        wick_ratio = self._get_latest_value(features, "wick_ratio")
        
        if wick_ratio is None or wick_ratio < 0.55:
            return None
        
        spread_pips = context.get("spread_pips", 1.5)
        liquidity_state = context.get("liquidity_state", "normal")
        spread_penalty = 0.7 if liquidity_state == "wide" else 1.0
        
        if high_swept: 
            direction = -1
            confidence = 0.55
        elif low_swept: 
            direction = 1
            confidence = 0.55
        else:
            return None
        
        if direction == -1 and close and close < swing_high:
            confidence = min(0.75, confidence + 0.15)
        elif direction == 1 and close and close > swing_low:
            confidence = min(0.75, confidence + 0.15)
        
        confidence *= spread_penalty
        
        trend_regime = context.get("htf_trend", 0)
        regime_fit = 0.65 if trend_regime == 0 else 0.55
        
        return PatternSignal(
            pattern_id=self. pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=regime_fit,
            historical_expectancy_R=0.38,
            historical_win_rate=0.54,
            sample_size=312,
            timestamp=features.index[-1],
            setup_type="mean_reversion",
        )