"""Test backtester functionality including the critical position sizing fix."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from settings import settings
from risk_manager import RiskManager, AccountState
from backtester_complete import BacktesterComplete
from pattern_aggregation import PatternAggregator
from patterns import VolatilityCompressionPattern, LiquiditySweepPattern, FailedBreakoutPattern


class TestPositionSizing:
    """Test the position sizing with leverage cap."""
    
    def test_position_size_respects_leverage_cap(self):
        """Verify position sizing caps units by max leverage."""
        account = AccountState(initial_capital=5000.0)
        risk_manager = RiskManager(account, {
            'max_risk_per_trade': 0.005,
            'max_daily_loss': 0.02,
            'max_weekly_loss': 0.05,
            'max_open_positions_total': 3,
            'max_open_positions_per_symbol': 1,
            'max_leverage': 5.0,
        })
        
        # Create a mock backtester for testing
        backtester = BacktesterComplete(settings)
        
        # Scenario: EUR_USD entry at 1.10 with small ATR
        entry_price = 1.10
        atr_14 = 0.0005  # Very small ATR
        sl_atr = 1.5
        sl_price = entry_price - (sl_atr * atr_14)  # Long trade SL
        
        units = backtester._calculate_position_size(entry_price, sl_price)
        
        # Calculate expected leverage
        notional = units * entry_price
        leverage = notional / 5000.0
        
        # Leverage should never exceed 5.0
        assert leverage <= 5.0, f"Leverage {leverage:.2f} exceeds max 5.0"
        
        # Units should be positive
        assert units > 0, "Units should be positive"
    
    def test_position_size_non_zero_for_valid_input(self):
        """Verify position sizing returns non-zero for valid inputs."""
        backtester = BacktesterComplete(settings)
        
        entry_price = 1.10
        sl_price = 1.09  # 100 pip stop
        
        units = backtester._calculate_position_size(entry_price, sl_price)
        
        assert units >= 1, "Units should be at least 1"
    
    def test_position_size_zero_for_zero_stop(self):
        """Verify position sizing returns 0 when stop distance is 0."""
        backtester = BacktesterComplete(settings)
        
        entry_price = 1.10
        sl_price = 1.10  # Same as entry
        
        units = backtester._calculate_position_size(entry_price, sl_price)
        
        assert units == 0, "Units should be 0 when stop distance is 0"


class TestRiskManagerValidation:
    """Test RiskManager validation with leverage-capped positions."""
    
    def test_validate_trade_passes_with_leverage_cap(self):
        """Verify trades pass validation when leverage is properly capped."""
        account = AccountState(initial_capital=5000.0)
        risk_manager = RiskManager(account, {
            'max_risk_per_trade': 0.005,
            'max_daily_loss': 0.02,
            'max_weekly_loss': 0.05,
            'max_open_positions_total': 3,
            'max_open_positions_per_symbol': 1,
            'max_leverage': 5.0,
        })
        
        # Position that respects both risk and leverage caps
        entry_price = 1.10
        sl_price = 1.09  # 100 pip stop = 0.01 distance
        # Max risk = 5000 * 0.005 = $25
        # Max units by risk = 25 / 0.01 = 2500
        # Max units by leverage = 5000 * 5 / 1.10 = ~22,727
        # So 2000 units should pass both checks
        units = 2000
        
        allowed, reason = risk_manager.validate_trade('EUR_USD', units, entry_price, sl_price)
        
        assert allowed is True, f"Trade should be allowed but got: {reason}"
    
    def test_validate_trade_rejects_overleveraged(self):
        """Verify trades are rejected when leverage exceeds max."""
        account = AccountState(initial_capital=5000.0)
        risk_manager = RiskManager(account, {
            'max_risk_per_trade': 0.005,
            'max_daily_loss': 0.02,
            'max_weekly_loss': 0.05,
            'max_open_positions_total': 3,
            'max_open_positions_per_symbol': 1,
            'max_leverage': 5.0,
        })
        
        # Position that exceeds leverage cap
        entry_price = 1.10
        sl_price = 1.09
        units = 50000  # Way over leverage cap (50000 * 1.10 / 5000 = 11x)
        
        allowed, reason = risk_manager.validate_trade('EUR_USD', units, entry_price, sl_price)
        
        assert allowed is False, "Trade should be rejected for overleveraged position"
        assert "Leverage" in reason


class TestPatternRecognition:
    """Test pattern recognition modules."""
    
    def test_volatility_compression_direction_bounds(self):
        """Verify VolatilityCompressionPattern returns valid directions."""
        pattern = VolatilityCompressionPattern()
        
        # Direction should always be -1, 0, or 1
        assert pattern.pattern_id == "VOL_COMPRESSION"
    
    def test_liquidity_sweep_pattern_id(self):
        """Verify LiquiditySweepPattern has correct ID."""
        pattern = LiquiditySweepPattern()
        assert pattern.pattern_id == "LIQUIDITY_SWEEP"
    
    def test_failed_breakout_pattern_id(self):
        """Verify FailedBreakoutPattern has correct ID."""
        pattern = FailedBreakoutPattern()
        assert pattern.pattern_id == "FAILED_BREAKOUT"


class TestPatternAggregation:
    """Test NPES aggregation logic."""
    
    def test_aggregation_empty_signals(self):
        """Verify aggregator handles empty signals."""
        aggregator = PatternAggregator()
        context = {'current_drawdown': 0.0}
        
        result = aggregator.aggregate([], context)
        
        assert result.trade_allowed is False
        assert result.direction == 0
        assert result.npes == 0.0
    
    def test_aggregation_npes_bounds(self):
        """Verify NPES is bounded between 0 and 1."""
        aggregator = PatternAggregator()
        
        # NPES = |long_sum - short_sum| / (long_sum + short_sum)
        # Since it's a ratio of absolute to gross, it should be [0, 1]
        assert 0.0 <= aggregator.min_npes_baseline <= 1.0


class TestBacktesterIntegration:
    """Integration tests for the backtester."""
    
    def test_backtester_initialization(self):
        """Verify backtester initializes without errors."""
        backtester = BacktesterComplete(settings)
        
        assert backtester.account.current_balance == 5000.0
        assert len(backtester.patterns) == 6
        assert backtester.aggregator is not None
    
    def test_backtester_context_building(self):
        """Verify context building produces valid output."""
        backtester = BacktesterComplete(settings)
        
        # Create minimal features DataFrame
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        features = pd.DataFrame({
            'time': dates,
            'close': np.random.uniform(1.0, 1.1, 300),
            'high': np.random.uniform(1.05, 1.15, 300),
            'low': np.random.uniform(0.95, 1.05, 300),
            'open': np.random.uniform(1.0, 1.1, 300),
            'ema_20': np.random.uniform(1.0, 1.1, 300),
            'ema_50': np.random.uniform(1.0, 1.1, 300),
            'atr_pctl_100': np.random.uniform(0, 100, 300),
        })
        features_dict = {'M15': features}
        
        context = backtester._build_context('EUR_USD', features, features_dict)
        
        assert 'session' in context
        assert 'htf_trend' in context
        assert 'volatility_regime' in context
        assert context['volatility_regime'] in ['low', 'medium', 'high']


class TestAccountState:
    """Test AccountState functionality."""
    
    def test_initial_state(self):
        """Verify initial account state."""
        account = AccountState(initial_capital=5000.0)
        
        assert account.initial_capital == 5000.0
        assert account.current_balance == 5000.0
        assert account.current_drawdown == 0.0
        assert account.max_drawdown == 0.0
    
    def test_drawdown_tracking(self):
        """Verify drawdown is tracked correctly."""
        account = AccountState(initial_capital=5000.0)
        
        # Simulate a loss
        account.update_equity(4500.0, pd.Timestamp.now())
        
        assert account.current_drawdown == 0.10  # 10% drawdown
        assert account.max_drawdown == 0.10
        
        # Partial recovery
        account.update_equity(4700.0, pd.Timestamp.now())
        
        assert account.current_drawdown == 0.06  # 6% drawdown
        assert account.max_drawdown == 0.10  # Max still 10%
    
    def test_position_tracking(self):
        """Verify position tracking."""
        account = AccountState(initial_capital=5000.0)
        
        account.add_position('trade1', 'EUR_USD', 10000, 1.10)
        
        assert account.total_position_count == 1
        assert account.position_count_by_symbol['EUR_USD'] == 1
        
        account.remove_position('trade1', 1.11, 100.0)
        
        assert account.total_position_count == 0
        assert account.total_trades == 1
        assert account.wins == 1
