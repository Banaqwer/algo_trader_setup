import logging
import pandas as pd
import numpy as np
from patterns. base import Pattern, PatternSignal

logger = logging.getLogger(__name__)

class HTFRangePattern(Pattern):
    """Pattern 4: HTF Range Extremes."""
    
    def __init__(self):
        super().__init__("HTF_RANGE")
    
    def recognize(self, instrument: str, timeframe: str, 
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Evaluate HTF range position."""
        
        htf_range_position = context.get("htf_range_position", None)
        
        if htf_range_position is None:
            return None
        
        close = self._get_latest_value(features, "close")
        if close is None:
            return None
        
        if htf_range_position <= 0.15:
            position_type = "lower_extreme"
            direction = 1
            confidence = 0.6
            regime_fit = 0.8
        elif htf_range_position >= 0.85:
            position_type = "upper_extreme"
            direction = -1
            confidence = 0.6
            regime_fit = 0.8
        elif 0.35 <= htf_range_position <= 0.65:
            return None
        else:
            position_type = "mid_extreme"
            direction = 1 if htf_range_position > 0.5 else -1
            confidence = 0.45
            regime_fit = 0.6
        
        atr_14 = self._get_latest_value(features, "atr_14")
        if atr_14 is not None:
            if position_type == "lower_extreme" and htf_range_position > 0.2:
                confidence = min(0.75, confidence + 0.15)
            elif position_type == "upper_extreme" and htf_range_position < 0.8:
                confidence = min(0.75, confidence + 0.15)
        
        return PatternSignal(
            pattern_id=self. pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=regime_fit,
            historical_expectancy_R=0.52,
            historical_win_rate=0.62,
            sample_size=189,
            timestamp=features.index[-1],
            setup_type="mean_reversion",
        )