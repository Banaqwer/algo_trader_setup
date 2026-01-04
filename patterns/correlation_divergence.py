import logging
import pandas as pd
import numpy as np
from patterns. base import Pattern, PatternSignal

logger = logging.getLogger(__name__)

class CorrelationDivergencePattern(Pattern):
    """Pattern 5: Correlation Divergence."""
    
    def __init__(self):
        super().__init__("CORRELATION_DIVERGENCE")
    
    CORR_PAIRS = {
        "EUR_USD": ("GBP_USD", 0.82),
        "GBP_USD": ("EUR_USD", 0.82),
        "AUD_USD": ("NZD_USD", 0.85),
        "NZD_USD": ("AUD_USD", 0.85),
        "USD_CAD": ("USD_JPY", 0.70),
        "USD_JPY": ("USD_CAD", 0.70),
    }
    
    def recognize(self, instrument: str, timeframe: str,
                  features: pd.DataFrame, context: dict) -> PatternSignal:
        """Detect correlation divergence."""
        
        if instrument not in self.CORR_PAIRS:
            return None
        
        corr_pair, expected_corr = self.CORR_PAIRS[instrument]
        
        inst_returns = features["log_returns"].iloc[-20: ].values
        pair_returns = context.get(f"{corr_pair}_returns", None)
        
        if pair_returns is None or len(pair_returns) < 20:
            return None
        
        inst_returns_recent = inst_returns[-20:]
        pair_returns_recent = pair_returns[-20:] if isinstance(pair_returns, np.ndarray) else np.array(pair_returns[-20:])
        
        inst_norm = (inst_returns_recent - np.mean(inst_returns_recent)) / (np.std(inst_returns_recent) + 1e-8)
        pair_norm = (pair_returns_recent - np. mean(pair_returns_recent)) / (np.std(pair_returns_recent) + 1e-8)
        
        rolling_corr = np.corrcoef(inst_norm[-10:], pair_norm[-10:])[0, 1]
        
        if rolling_corr < expected_corr - 0.25:
            confidence = 0.5
            regime_fit = 0.6
        elif rolling_corr < expected_corr - 0.10:
            confidence = 0.35
            regime_fit = 0.5
        else:
            return None
        
        inst_momentum = np.mean(inst_norm[-5:])
        pair_momentum = np.mean(pair_norm[-5:])
        
        if inst_momentum > pair_momentum: 
            direction = -1
        elif inst_momentum < pair_momentum:
            direction = 1
        else:
            return None
        
        return PatternSignal(
            pattern_id=self.pattern_id,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            regime_compatibility=regime_fit,
            historical_expectancy_R=0.41,
            historical_win_rate=0.56,
            sample_size=134,
            timestamp=features.index[-1],
            setup_type="mean_reversion",
        )