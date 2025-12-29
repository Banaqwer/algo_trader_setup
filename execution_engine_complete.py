"""execution_engine_complete.py"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import uuid

from trading_utils import calculate_spread_adjustment, calculate_slippage_adjustment

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Complete record of a closed trade."""

    trade_id: str
    instrument: str
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_timeframe: str

    exit_time: Optional[pd. Timestamp] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None

    units:  int = 0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_atr_multiple: float = 0.0
    tp_atr_multiple:  float = 0.0
    atr_at_entry: float = 0.0

    patterns_triggered: List[str] = field(default_factory=list)
    npes_score: float = 0.0
    expected_value_R: float = 0.0
    model_confidence: float = 0.0
    regime_at_entry: str = ""
    session_at_entry: str = ""

    gross_pnl_usd: float = 0.0
    pnl_after_spread: float = 0.0
    pnl_after_slippage: float = 0.0
    pnl_usd: float = 0.0
    pnl_atr: float = 0.0

    mae_usd: float = 0.0
    mae_atr: float = 0.0
    mae_pct: float = 0.0

    mfe_usd: float = 0.0
    mfe_atr: float = 0.0
    mfe_pct:  float = 0.0

    entry_spread_cost: float = 0.0
    entry_slippage_cost: float = 0.0
    exit_spread_cost: float = 0.0
    exit_slippage_cost: float = 0.0

    bars_held: int = 0

    def to_dict(self) -> Dict:
        """Export as dict."""
        d = self.__dict__. copy()
        d['entry_time'] = str(d['entry_time'])
        d['exit_time'] = str(d['exit_time']) if d['exit_time'] else None
        d['patterns_triggered'] = ",".join(d['patterns_triggered'])
        return d

@dataclass
class OpenTrade:
    """Trade currently open."""

    trade_id: str
    instrument: str
    direction: int
    units: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_timeframe: str

    sl_price: float
    tp_price: float
    sl_atr_multiple: float
    tp_atr_multiple: float
    atr_at_entry: float

    patterns_triggered: List[str]
    npes_score: float
    expected_value_R:  float
    model_confidence: float
    regime_at_entry:  str
    session_at_entry: str

    peak_price: float = 0.0
    peak_price_time: Optional[pd.Timestamp] = None
    trough_price: float = 0.0
    trough_price_time: Optional[pd. Timestamp] = None
    bars_held: int = 0

    def update_extremes(self, current_price:  float, timestamp: pd.Timestamp):
        """Update peak/trough for MAE/MFE."""

        if self.direction > 0:
            if current_price > self.peak_price:
                self.peak_price = current_price
                self.peak_price_time = timestamp
            if current_price < self.trough_price or self.trough_price == 0:
                self.trough_price = current_price
                self.trough_price_time = timestamp
        else:
            if current_price < self.trough_price or self.trough_price == 0:
                self.trough_price = current_price
                self.trough_price_time = timestamp
            if current_price > self. peak_price:
                self. peak_price = current_price
                self.peak_price_time = timestamp

class SimulatedBroker:
    """Simulates order execution with spread and slippage."""

    def __init__(self, account_balance: float, config: Dict):
        self.balance = account_balance
        self. config = config
        self.open_trades: Dict[str, OpenTrade] = {}
        self.closed_trades: List[TradeRecord] = []

        self.total_trades = 0
        self.total_gross_pnl = 0.0
        self. total_costs = 0.0

        logger.info(f"SimulatedBroker initialized:  balance=${self.balance:.2f}")

    def place_order(
        self,
        instrument: str,
        direction: int,
        units:  int,
        entry_price:  float,
        sl_price: float,
        tp_price: float,
        atr_14: float,
        sl_atr:  float,
        tp_atr: float,
        patterns:  List[str],
        npes:  float,
        model_confidence: float,
        expected_value_R: float,
        regime: str,
        session: str,
        timeframe: str,
        timestamp: pd.Timestamp,
        spread_pips: float = 1.5,
        slippage_bps: float = 1.5,
    ) -> Optional[str]:
        """Place an order."""

        # Apply spread adjustment
        entry_price_filled, spread_price = calculate_spread_adjustment(
            entry_price, direction, spread_pips, instrument
        )
        entry_spread_cost = spread_price * units

        # Apply slippage adjustment
        entry_price_filled, slippage_price = calculate_slippage_adjustment(
            entry_price_filled, direction, slippage_bps
        )
        entry_slippage_cost = slippage_price * abs(units)

        trade_id = str(uuid.uuid4())[:8]
        trade = OpenTrade(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            units=units,
            entry_price=entry_price_filled,
            entry_time=timestamp,
            entry_timeframe=timeframe,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_atr_multiple=sl_atr,
            tp_atr_multiple=tp_atr,
            atr_at_entry=atr_14,
            patterns_triggered=patterns,
            npes_score=npes,
            expected_value_R=expected_value_R,
            model_confidence=model_confidence,
            regime_at_entry=regime,
            session_at_entry=session,
            peak_price=entry_price_filled if direction > 0 else 0,
            trough_price=entry_price_filled if direction < 0 else float('inf'),
        )

        self.open_trades[trade_id] = trade
        self.total_trades += 1

        logger.info(
            f"Trade {trade_id} opened: {instrument} {direction: +d} @ {entry_price_filled:.5f}"
        )

        return trade_id

    def update_price(
        self, trade_id: str, current_price: float, timestamp: pd.Timestamp
    ) -> Optional[Tuple[str, TradeRecord]]:
        """Update trade with current price.  Check for SL/TP/timeout exit."""

        if trade_id not in self.open_trades:
            return None

        trade = self.open_trades[trade_id]
        trade.update_extremes(current_price, timestamp)
        trade.bars_held += 1

        # Check SL hit
        if trade.direction > 0:
            if current_price <= trade.sl_price:
                return self._close_trade(trade_id, trade. sl_price, "SL", timestamp)
        else:
            if current_price >= trade.sl_price:
                return self._close_trade(trade_id, trade.sl_price, "SL", timestamp)

        # Check TP hit
        if trade. direction > 0:
            if current_price >= trade.tp_price:
                return self._close_trade(trade_id, trade.tp_price, "TP", timestamp)
        else:
            if current_price <= trade.tp_price:
                return self._close_trade(trade_id, trade.tp_price, "TP", timestamp)

        return None

    def close_trade_by_timeout(
        self, trade_id: str, current_price: float, timestamp: pd.Timestamp
    ) -> Optional[TradeRecord]:
        """Close trade on timeout."""

        if trade_id not in self.open_trades:
            return None

        return self._close_trade(trade_id, current_price, "TIME", timestamp)[1]

    def _close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_type: str,
        timestamp: pd.Timestamp,
    ) -> Tuple[str, TradeRecord]: 
        """Close a trade and compute full P&L."""

        open_trade = self.open_trades. pop(trade_id)

        # Apply spread adjustment (note: direction reversed for exit)
        exit_price_filled, spread_price = calculate_spread_adjustment(
            exit_price, -open_trade.direction, 1.5, open_trade.instrument
        )
        exit_spread_cost = spread_price * abs(open_trade.units)

        # Apply slippage adjustment (note: direction reversed for exit)
        slippage_bps = self.config.get("slippage_bps", 1.5)
        exit_price_filled, slippage_price = calculate_slippage_adjustment(
            exit_price_filled, -open_trade.direction, slippage_bps
        )
        exit_slippage_cost = slippage_price * abs(open_trade.units)

        if open_trade.direction > 0:
            gross_pnl_usd = (exit_price - open_trade.entry_price) * open_trade.units
            mae_usd = (
                open_trade.trough_price - open_trade. entry_price
            ) * open_trade.units
            mfe_usd = (
                open_trade.peak_price - open_trade.entry_price
            ) * open_trade.units
        else:
            gross_pnl_usd = (
                open_trade.entry_price - exit_price
            ) * abs(open_trade.units)
            mae_usd = (
                open_trade.peak_price - open_trade.entry_price
            ) * abs(open_trade.units)
            mfe_usd = (
                open_trade.entry_price - open_trade.trough_price
            ) * abs(open_trade.units)

        pnl_after_spread = gross_pnl_usd - exit_spread_cost
        pnl_after_slippage = pnl_after_spread - exit_slippage_cost
        pnl_usd = pnl_after_slippage

        pnl_atr = (
            pnl_usd / (open_trade.atr_at_entry * abs(open_trade.units))
            if open_trade.atr_at_entry > 0
            else 0
        )
        mae_atr = (
            mae_usd / (open_trade.atr_at_entry * abs(open_trade.units))
            if open_trade.atr_at_entry > 0
            else 0
        )
        mfe_atr = (
            mfe_usd / (open_trade.atr_at_entry * abs(open_trade.units))
            if open_trade.atr_at_entry > 0
            else 0
        )

        mae_pct = (
            mae_usd / max(abs(open_trade.units) * open_trade.entry_price, 1) * 100
        )
        mfe_pct = (
            mfe_usd / max(abs(open_trade.units) * open_trade.entry_price, 1) * 100
        )

        record = TradeRecord(
            trade_id=trade_id,
            instrument=open_trade.instrument,
            direction=open_trade.direction,
            entry_time=open_trade.entry_time,
            entry_price=open_trade.entry_price,
            entry_timeframe=open_trade.entry_timeframe,
            exit_time=timestamp,
            exit_price=exit_price_filled,
            exit_type=exit_type,
            units=open_trade.units,
            sl_price=open_trade. sl_price,
            tp_price=open_trade.tp_price,
            sl_atr_multiple=open_trade.sl_atr_multiple,
            tp_atr_multiple=open_trade.tp_atr_multiple,
            atr_at_entry=open_trade.atr_at_entry,
            patterns_triggered=open_trade.patterns_triggered,
            npes_score=open_trade.npes_score,
            expected_value_R=open_trade.expected_value_R,
            model_confidence=open_trade.model_confidence,
            regime_at_entry=open_trade.regime_at_entry,
            session_at_entry=open_trade.session_at_entry,
            gross_pnl_usd=gross_pnl_usd,
            pnl_after_spread=pnl_after_spread,
            pnl_after_slippage=pnl_after_slippage,
            pnl_usd=pnl_usd,
            pnl_atr=pnl_atr,
            mae_usd=mae_usd,
            mae_atr=mae_atr,
            mae_pct=mae_pct,
            mfe_usd=mfe_usd,
            mfe_atr=mfe_atr,
            mfe_pct=mfe_pct,
            entry_spread_cost=exit_spread_cost,
            entry_slippage_cost=exit_slippage_cost,
            exit_spread_cost=exit_spread_cost,
            exit_slippage_cost=exit_slippage_cost,
            bars_held=open_trade. bars_held,
        )

        self.closed_trades.append(record)
        self.total_gross_pnl += gross_pnl_usd
        self.total_costs += exit_spread_cost + exit_slippage_cost
        self.balance += pnl_usd

        logger.info(
            f"Trade {trade_id} closed: {exit_type} @ {exit_price_filled:.5f} | "
            f"PnL: {pnl_usd:+.2f} USD ({pnl_atr:+.2f}R)"
        )

        return exit_type, record

    def get_open_trade_count(self) -> int:
        return len(self.open_trades)

    def get_open_trades_for_instrument(self, instrument: str) -> List[OpenTrade]: 
        return [
            t for t in self.open_trades.values() if t.instrument == instrument
        ]