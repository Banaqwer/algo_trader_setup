import logging
import pandas as pd
import numpy as np
from patterns.base import Pattern, PatternSignal, validate_features_length, detect_breakout

logger = logging.getLogger(__name__)

class FailedBreakoutPattern(Pattern):
    """Pattern 6: Failed Breakout / Acceptance-Rejection."""
    
    def __init__(self):
        super().__init__("FAILED_BREAKOUT")
    
    def recognize(self, instrument: str, timeframe: str,
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Detect failed breakout setup."""
        
        if not validate_features_length(features, 40):
            return None
        
        lookback_structure = 20
        high_vals = features["high"].iloc[-lookback_structure:].values
        low_vals = features["low"].iloc[-lookback_structure:].values
        close_vals = features["close"].iloc[-lookback_structure:].values
        
        structure_high = np.max(high_vals[:-1])
        structure_low = np.min(low_vals[:-1])
        
        current_close = close_vals[-1]
        current_high = high_vals[-1]
        current_low = low_vals[-1]
        prior_close = close_vals[-2]
        
        bullish_breakout, bearish_breakout = detect_breakout(
            current_high, current_low, prior_close, structure_high, structure_low
        )
        
        if not (bullish_breakout or bearish_breakout):
            return None
        
        recent_closes = close_vals[-3:]
        recent_highs = high_vals[-3:]
        recent_lows = low_vals[-3:]
        
        if bullish_breakout:
            failure = (
                current_close < structure_high or
                current_close < prior_close or
                (recent_closes[-1] < recent_highs[0] and recent_closes[0] < recent_highs[0])
            )
            if failure:
                direction = -1
                confidence = 0.55
            else:
                return None
        elif bearish_breakout:
            failure = (
                current_close > structure_low or
                current_close > prior_close or
                (recent_closes[-1] > recent_lows[0] and recent_closes[0] > recent_lows[0])
            )
            if failure:
                direction = 1
                confidence = 0.55
            else:
                return None
        else: 
            return None
        
        rsi_vals = features["rsi_14"].iloc[-5:].values
        rsi_declining = rsi_vals[-1] < rsi_vals[0]
        
        if rsi_declining: 
            confidence = min(0.75, confidence + 0.10)
        
        vol_regime = context.get("volatility_regime", "medium")
        regime_fit = 0.65 if vol_regime == "medium" else 0.55
        
        return PatternSignal(
            pattern_id=self. pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=regime_fit,
            historical_expectancy_R=0.47,
            historical_win_rate=0.60,
            sample_size=201,
            timestamp=features.index[-1],
            setup_type="mean_reversion",
        )