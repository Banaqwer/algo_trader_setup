import logging
import pandas as pd
import numpy as np
from patterns.base import Pattern, PatternSignal

logger = logging.getLogger(__name__)

class SessionBiasPattern(Pattern):
    """Pattern 3: Session Bias."""
    
    def __init__(self):
        super().__init__("SESSION_BIAS")
        self.session_stats = {}
    
    def update_stats(self, key: tuple, win_rate: float, expectancy: float, count: int):
        """Update session statistics."""
        self.session_stats[key] = (win_rate, expectancy, count)
    
    def recognize(self, instrument: str, timeframe: str, 
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Detect session-biased setups."""
        
        session = context.get("session", "unknown")
        if session == "unknown":
            return None
        
        setup_type = context.get("current_setup_type", "unknown")
        key = (instrument, session, setup_type)
        
        if key not in self.session_stats:
            return None
        
        win_rate, expectancy, count = self. session_stats[key]
        
        if win_rate < 0.55 or count < 20:
            return None
        
        rsi = self._get_latest_value(features, "rsi_14")
        ema_20 = self._get_latest_value(features, "ema_20")
        ema_50 = self._get_latest_value(features, "ema_50")
        
        if ema_20 is None or ema_50 is None: 
            return None
        
        if ema_20 > ema_50:
            direction = 1
        elif ema_20 < ema_50:
            direction = -1
        else:
            return None
        
        confidence = min(0.75, 0.5 + (win_rate - 0.5) * 0.5)
        confidence *= min(1.0, count / 100.0)
        
        return PatternSignal(
            pattern_id=self.pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=0.6,
            historical_expectancy_R=expectancy,
            historical_win_rate=win_rate,
            sample_size=count,
            timestamp=features.index[-1],
            setup_type=setup_type,
        )