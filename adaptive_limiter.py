import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AdaptiveTradeLimiter:
    """Adjust max trades per day based on recent performance and confidence."""
    
    def __init__(self, settings):
        self.settings = settings
        self.daily_stats = {}  # Track per-day performance
        self.confidence_history = []  # Track recent confidences
    
    def calculate_max_trades(self, account_state, recent_trades: list) -> int:
        """Calculate adaptive max trades for today."""
        
        if not self.settings.adaptive_trading:
            return self.settings. max_trades_per_day
        
        base = self.settings.base_max_trades_per_day
        
        # Get today's stats
        today = datetime.utcnow().date()
        today_stats = self._get_today_stats(recent_trades)
        
        # Calculate win rate
        win_rate = self._calculate_win_rate(recent_trades, days=5)  # Last 5 days
        
        # Calculate average confidence
        avg_confidence = self._calculate_avg_confidence(recent_trades, days=5)
        
        # Start with base
        adaptive_trades = base
        
        # Adjust down if win rate is bad (< 40%)
        if win_rate < 0.40:
            adaptive_trades = max(self.settings.min_trades_per_day, int(base * 0.5))
            logger.info(
                f"Win rate low ({win_rate:.1%}): Reducing to {adaptive_trades} trades"
            )
        
        # Adjust up if win rate is good (> 60%) AND confidence is high
        elif win_rate > 0.60 and avg_confidence > 0.70:
            adaptive_trades = min(self.settings.max_trades_per_day, int(base * 1.5))
            logger.info(
                f"Strong performance ({win_rate:.1%}, conf={avg_confidence:.1%}): "
                f"Increasing to {adaptive_trades} trades"
            )
        
        # Adjust based on confidence alone
        confidence_adjustment = int(
            base * (avg_confidence - 0.5) * self.settings.confidence_multiplier
        )
        adaptive_trades = max(
            self.settings.min_trades_per_day,
            min(self.settings.max_trades_per_day, adaptive_trades + confidence_adjustment)
        )
        
        # Check daily loss - reduce if we've lost money today
        daily_pnl = account_state.daily_pnl if account_state else 0.0
        if daily_pnl < 0:
            loss_percent = abs(daily_pnl) / account_state.initial_capital
            if loss_percent > 0.01:  # Lost more than 1%
                adaptive_trades = max(
                    self.settings.min_trades_per_day,
                    int(adaptive_trades * 0.5)
                )
                logger.warning(
                    f"Daily loss ({loss_percent:.2%}): Reducing to {adaptive_trades} trades"
                )
        
        logger.info(
            f"Adaptive limit:  {adaptive_trades} trades (WR={win_rate:.1%}, "
            f"Conf={avg_confidence:.1%}, Daily PnL={daily_pnl:+.2f})"
        )
        
        return adaptive_trades
    
    def _get_today_stats(self, recent_trades: list) -> Dict:
        """Get today's trade statistics."""
        today = datetime.utcnow().date()
        today_trades = [
            t for t in recent_trades
            if hasattr(t, 'exit_time') and t.exit_time. date() == today
        ]
        
        winning = sum(1 for t in today_trades if t.pnl_usd > 0)
        losing = sum(1 for t in today_trades if t.pnl_usd < 0)
        
        return {
            "total":  len(today_trades),
            "winning": winning,
            "losing": losing,
            "pnl":  sum(t.pnl_usd for t in today_trades),
        }
    
    def _calculate_win_rate(self, recent_trades: list, days: int = 5) -> float:
        """Calculate win rate from last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent = [
            t for t in recent_trades
            if hasattr(t, 'exit_time') and t.exit_time > cutoff
        ]
        
        if not recent: 
            return 0.5  # Neutral if no data
        
        winning = sum(1 for t in recent if t.pnl_usd > 0)
        return winning / len(recent)
    
    def _calculate_avg_confidence(self, recent_trades:  list, days: int = 5) -> float:
        """Calculate average signal confidence from last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent = [
            t for t in recent_trades
            if hasattr(t, 'entry_time') and t.entry_time > cutoff
        ]
        
        if not recent:
            return 0.60  # Default neutral confidence
        
        avg_conf = sum(getattr(t, 'confidence', 0.60) for t in recent) / len(recent)
        return avg_conf
    
    def should_take_trade(
        self,
        current_trades_today:  int,
        signal_confidence:  float,
        max_allowed: int
    ) -> Tuple[bool, str]:
        """Check if we should take a trade given current state."""
        
        # Rule 1: Confidence minimum
        if signal_confidence < self. settings.min_confidence_to_trade:
            return False, (
                f"Confidence {signal_confidence:.1%} < "
                f"{self.settings. min_confidence_to_trade:.1%}"
            )
        
        # Rule 2: Daily limit
        if current_trades_today >= max_allowed:
            return False, f"Daily limit {max_allowed} reached"
        
        return True, "OK"