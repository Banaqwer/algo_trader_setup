"""risk_manager.py"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class AccountState:
    """Tracks equity, positions, drawdown, and loss limits."""
    
    def __init__(self, initial_capital: float = 5000.0):
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self. peak_equity = initial_capital
        self. current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        self.day_start_balance = initial_capital
        self.day_start_time = None
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        
        self.week_start_balance = initial_capital
        self.week_start_time = None
        self.weekly_pnl = 0.0
        self.weekly_loss_limit_hit = False
        
        self.open_positions = {}
        self.position_count_by_symbol = {}
        self.total_position_count = 0
        
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.max_leverage_used = 0.0
        
        self.closed_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.gross_volume_traded = 0.0
    
    def update_equity(self, new_balance: float, timestamp: pd.Timestamp):
        """Update current equity and track drawdown."""
        
        self.current_balance = new_balance
        
        if new_balance > self.peak_equity:
            self.peak_equity = new_balance
        
        self.current_drawdown = (self.peak_equity - new_balance) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        self.daily_pnl = new_balance - self.day_start_balance
        self.weekly_pnl = new_balance - self.week_start_balance
    
    def add_position(self, trade_id: str, instrument: str, units: int, entry_price: float):
        """Register an open position."""
        
        self.open_positions[trade_id] = {
            "instrument": instrument,
            "units": units,
            "entry_price": entry_price,
            "entry_time": datetime.utcnow(),
        }
        
        self.position_count_by_symbol[instrument] = self.position_count_by_symbol.get(instrument, 0) + 1
        self.total_position_count += 1
    
    def remove_position(self, trade_id: str, exit_price: float, pnl: float):
        """Close a position and update stats."""
        
        if trade_id not in self.open_positions:
            logger.warning(f"Trade {trade_id} not found in open positions")
            return
        
        pos = self.open_positions. pop(trade_id)
        instrument = pos["instrument"]
        
        self.position_count_by_symbol[instrument] -= 1
        self.total_position_count -= 1
        
        self.total_trades += 1
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1
        
        self.closed_pnl += pnl
        self.current_balance += pnl
    
    def reset_daily(self, timestamp: pd.Timestamp):
        """Reset daily limits."""
        self.day_start_balance = self.current_balance
        self. day_start_time = timestamp
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        logger.info(f"Daily reset at {timestamp}")
    
    def reset_weekly(self, timestamp:  pd.Timestamp):
        """Reset weekly limits."""
        self. week_start_balance = self. current_balance
        self.week_start_time = timestamp
        self.weekly_pnl = 0.0
        self. weekly_loss_limit_hit = False
        logger.info(f"Weekly reset at {timestamp}")
    
    def to_dict(self) -> Dict:
        """Export state as dict."""
        return {
            "current_balance": self.current_balance,
            "peak_equity":  self.peak_equity,
            "current_drawdown": self. current_drawdown,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self. weekly_pnl,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "closed_pnl": self.closed_pnl,
            "open_positions": len(self.open_positions),
        }

class RiskManager:
    """Enforces risk constraints and halt logic."""
    
    def __init__(self, account_state: AccountState, config: Dict):
        self.account = account_state
        self.config = config
        
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.005)
        self.max_daily_loss = config.get("max_daily_loss", 0.02)
        self.max_weekly_loss = config.get("max_weekly_loss", 0.05)
        self.max_open_positions_total = config.get("max_open_positions_total", 3)
        self.max_open_positions_per_symbol = config.get("max_open_positions_per_symbol", 1)
        self.max_leverage = config.get("max_leverage", 5.0)
        
        logger.info(f"RiskManager initialized:   max_dd_daily={self.max_daily_loss:.1%}, "
                   f"max_dd_weekly={self. max_weekly_loss:.1%}")
    
    def check_can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed given current state."""
        
        if self.account.daily_pnl < -self.account.day_start_balance * self.max_daily_loss:
            self.account.daily_loss_limit_hit = True
            return False, f"Daily loss limit hit: {self.account.daily_pnl:.2f} USD"
        
        if self.account.weekly_pnl < -self.account.week_start_balance * self.max_weekly_loss:
            self.account.weekly_loss_limit_hit = True
            return False, f"Weekly loss limit hit:  {self.account.weekly_pnl:.2f} USD"
        
        return True, "OK"
    
    def check_position_limits(self, instrument: str) -> Tuple[bool, str]: 
        """Check if new position can be opened."""
        
        if self. account.total_position_count >= self.max_open_positions_total: 
            return False, f"Max open positions ({self.max_open_positions_total}) reached"
        
        symbol_count = self.account.position_count_by_symbol.get(instrument, 0)
        if symbol_count >= self.max_open_positions_per_symbol: 
            return False, f"Max positions per symbol ({self.max_open_positions_per_symbol}) reached for {instrument}"
        
        return True, "OK"
    
    def check_leverage(self, units: int, entry_price: float) -> Tuple[bool, str]:
        """Check if position leverage is within limits."""
        
        notional = abs(units) * entry_price
        leverage = notional / self.account.current_balance if self.account.current_balance > 0 else 999
        
        if leverage > self. max_leverage:
            return False, f"Leverage ({leverage:. 2f}) exceeds max ({self.max_leverage})"
        
        return True, "OK"
    
    def check_risk_per_trade(self, units: int, sl_price: float, entry_price: float) -> Tuple[bool, str]:
        """Check if trade risk is within limit."""
        
        stop_distance = abs(entry_price - sl_price)
        risk_usd = abs(units) * stop_distance
        max_risk = self.account.current_balance * self.max_risk_per_trade
        
        if risk_usd > max_risk:
            return False, f"Risk USD ({risk_usd:.2f}) exceeds max ({max_risk:.2f})"
        
        return True, "OK"
    
    def validate_trade(self, instrument: str, units: int, 
                      entry_price: float, sl_price: float) -> Tuple[bool, str]: 
        """Run all risk checks."""
        
        checks = [
            self.check_can_trade(),
            self.check_position_limits(instrument),
            self.check_leverage(units, entry_price),
            self.check_risk_per_trade(units, sl_price, entry_price),
        ]
        
        for allowed, msg in checks:
            if not allowed:
                return False, msg
        
        return True, "OK"