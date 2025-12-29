"""pattern_aggregation.py"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

@dataclass
class AggregationResult:
    """Result of pattern aggregation."""
    npes: float
    direction: int
    confidence: float
    contributing_patterns: List[str]
    pattern_breakdown: Dict[str, float]
    expected_value_R:  float
    regime_fit: float
    trade_allowed: bool

class PatternAggregator:
    """Aggregates pattern signals using NPES."""
    
    def __init__(self, min_npes_baseline: float = 0.15, min_npes_risk_off: float = 0.25):
        self.min_npes_baseline = min_npes_baseline
        self.min_npes_risk_off = min_npes_risk_off
        logger.info("PatternAggregator initialized")
    
    def aggregate(self, signals: List, context: Dict) -> AggregationResult:
        """Aggregate pattern signals."""
        
        if not signals:
            return AggregationResult(
                npes=0.0,
                direction=0,
                confidence=0.0,
                contributing_patterns=[],
                pattern_breakdown={},
                expected_value_R=0.0,
                regime_fit=0.0,
                trade_allowed=False,
            )
        
        contributions = {}
        long_contributions = []
        short_contributions = []
        
        for signal in signals: 
            log_sample_size = np.log(1.0 + signal.sample_size)
            contribution = (
                signal.confidence *
                signal.regime_compatibility *
                max(0, signal.historical_expectancy_R) *
                log_sample_size
            )
            
            contributions[signal.pattern_id] = contribution
            
            if signal.direction > 0:
                long_contributions.append((signal.pattern_id, contribution))
            elif signal.direction < 0:
                short_contributions.append((signal.pattern_id, contribution))
        
        long_sum = sum(c for _, c in long_contributions)
        short_sum = sum(c for _, c in short_contributions)
        
        net_contribution = long_sum - short_sum
        gross_contribution = long_sum + short_sum
        
        if gross_contribution > 0:
            npes = abs(net_contribution / gross_contribution)
        else:
            npes = 0.0
        
        if long_sum > short_sum:
            direction = 1
        elif short_sum > long_sum: 
            direction = -1
        else:
            direction = 0
        
        if direction == 1:
            contrib_signals = [s for s in signals if s. direction > 0]
        elif direction == -1:
            contrib_signals = [s for s in signals if s.direction < 0]
        else:
            contrib_signals = []
        
        if contrib_signals:
            avg_confidence = np.mean([s.confidence for s in contrib_signals])
            avg_ev = np.mean([s.historical_expectancy_R for s in contrib_signals])
            avg_regime_fit = np.mean([s.regime_compatibility for s in contrib_signals])
        else:
            avg_confidence = 0.0
            avg_ev = 0.0
            avg_regime_fit = 0.0
        
        current_drawdown = context.get("current_drawdown", 0.0)
        min_threshold = self.min_npes_risk_off if current_drawdown > 0.01 else self.min_npes_baseline
        
        trade_allowed = (npes > min_threshold) and (direction != 0)
        
        contributing_patterns = [s.pattern_id for s in contrib_signals]
        
        return AggregationResult(
            npes=npes,
            direction=direction,
            confidence=avg_confidence,
            contributing_patterns=contributing_patterns,
            pattern_breakdown=contributions,
            expected_value_R=avg_ev,
            regime_fit=avg_regime_fit,
            trade_allowed=trade_allowed,
        )