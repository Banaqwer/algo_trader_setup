import logging
import pandas as pd
import numpy as np
from patterns.base import Pattern, PatternSignal, validate_features_length

logger = logging.getLogger(__name__)

class VolatilityCompressionPattern(Pattern):
    """Pattern 1: Volatility Compression → Expansion."""
    
    def __init__(self):
        super().__init__("VOL_COMPRESSION")
    
    def recognize(self, instrument: str, timeframe: str,
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Detect compression and expansion."""
        
        if not validate_features_length(features, 1):
            return None
        
        atr_14 = self._get_latest_value(features, "atr_14")
        atr_14_pctl = self._get_latest_value(features, "atr_pctl_100")
        bb_width = self._get_latest_value(features, "bb_width")
        close = self._get_latest_value(features, "close")
        bb_upper = self._get_latest_value(features, "bb_upper")
        bb_lower = self._get_latest_value(features, "bb_lower")
        adx = self._get_latest_value(features, "adx_14")
        
        if any(v is None for v in [atr_14, atr_14_pctl, bb_width, close, bb_upper, bb_lower, adx]):
            return None
        
        compression_detected = (atr_14_pctl <= 20.0 and bb_width > 0)
        
        if not compression_detected:
            return None
        
        atr_last_5 = features["atr_14"].iloc[-5:].values
        atr_expanding = atr_last_5[-1] > atr_last_5[0] * 1.15
        
        if not atr_expanding:
            return None
        
        bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        if bb_position > 0.6:
            direction = 1
        elif bb_position < 0.4:
            direction = -1
        else:
            rsi = self._get_latest_value(features, "rsi_14")
            direction = 1 if rsi and rsi > 50 else -1 if rsi and rsi < 50 else 0
        
        if direction == 0:
            return None
        
        compression_score = (1.0 - atr_14_pctl / 100.0)
        expansion_score = min(1.0, (atr_last_5[-1] / atr_last_5[0] - 1.0) * 5)
        confidence = 0.5 * compression_score + 0.5 * expansion_score
        confidence = np.clip(confidence, 0.3, 0.95)
        
        vol_regime = context.get("volatility_regime", "medium")
        adx_val = adx or 20
        regime_fit = 1.0 if vol_regime == "high" else 0.7 if vol_regime == "medium" else 0.4
        regime_fit *= min(1.0, adx_val / 30.0)
        
        return PatternSignal(
            pattern_id=self.pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=regime_fit,
            historical_expectancy_R=0.45,
            historical_win_rate=0.58,
            sample_size=247,
            timestamp=features.index[-1],
            setup_type="breakout",
        )