import logging
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from settings import settings
from safety import block_execution_for_backtest
from data_downloader import DataDownloader
from feature_engineering import FeatureEngine
from patterns import (
    VolatilityCompressionPattern,
    LiquiditySweepPattern,
    SessionBiasPattern,
    HTFRangePattern,
    CorrelationDivergencePattern,
    FailedBreakoutPattern,
)
from pattern_aggregation import PatternAggregator
from models.model_manager import model_manager
from execution_engine_complete import SimulatedBroker, TradeRecord
from risk_manager import RiskManager, AccountState
from self_learning import SelfLearningEngine
from adaptive_limiter import AdaptiveTradeLimiter
from trading_utils import detect_session

logger = logging.getLogger(__name__)


class BacktesterComplete:
    """Full event-driven backtester with adaptive trade limiting."""

    def __init__(self, settings):
        self.settings = settings
        block_execution_for_backtest(settings)

        self.downloader = DataDownloader(settings)
        self.feature_engine = FeatureEngine()

        self.patterns = [
            VolatilityCompressionPattern(),
            LiquiditySweepPattern(),
            SessionBiasPattern(),
            HTFRangePattern(),
            CorrelationDivergencePattern(),
            FailedBreakoutPattern(),
        ]

        self.aggregator = PatternAggregator()

        self.account = AccountState(initial_capital=5000.0)
        self.risk_manager = RiskManager(
            self.account,
            {
                "max_risk_per_trade": 0.005,
                "max_daily_loss": 0.02,
                "max_weekly_loss": 0.05,
                "max_open_positions_total": 3,
                "max_open_positions_per_symbol": 1,
                "max_leverage": 5.0,
            },
        )

        self.broker = SimulatedBroker(self.account.current_balance, settings.dict())
        self.learning_engine = SelfLearningEngine(settings.data_dir)
        self.adaptive_limiter = AdaptiveTradeLimiter(settings)

        self.daily_trade_count = {}
        self.closed_trades = []
        self.last_daily_reset = None
        self.last_weekly_reset = None
        
        # Diagnostic counters for signal pipeline analysis
        self._init_diagnostics()

        logger.info(
            f"BacktesterComplete initialized (adaptive_trading={settings.adaptive_trading})"
        )
    
    def _init_diagnostics(self):
        """Initialize diagnostic counters for pipeline analysis."""
        self.diagnostics = {
            # Per-pattern signal counts
            "pattern_signals_emitted": {},
            # Pipeline survival counts
            "signals_after_context_filters": 0,
            "signals_after_npes": 0,
            "blocked_by_confidence_gate": 0,
            "blocked_by_model_confirmation": 0,
            "blocked_by_risk": {},
            "blocked_by_policy": {},
            "executed_trades": 0,
            # Daily tracking
            "days_with_no_trades": [],
            "trades_per_day": {},
            # Per-pattern stats
            "pattern_fire_rate": {},
        }
        
        # Initialize per-pattern counters
        for pattern in self.patterns:
            self.diagnostics["pattern_signals_emitted"][pattern.pattern_id] = 0
            self.diagnostics["pattern_fire_rate"][pattern.pattern_id] = {"fired": 0, "checked": 0}

    def run(
        self,
        instruments: List[str],
        timeframes:  List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Run a full backtest."""

        run_id = str(uuid.uuid4())[: 8]
        run_dir = Path(self.settings.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting backtest run_id={run_id}")

        try:
            logger.info("Loading historical data...")
            data_cache = self._load_data(instruments, timeframes, start_date, end_date)

            logger.info("Running event loop...")
            trades = self._run_event_loop(data_cache, instruments, timeframes)

            logger.info("Saving results...")
            self._save_results(run_id, run_dir, trades, data_cache)
            
            # Flush any pending learning engine updates
            if self.settings.enable_learning_updates:
                self.learning_engine.flush()

            logger.info(
                f"Backtest complete: {len(trades)} trades, "
                f"P&L: ${self.broker.balance - 5000:+.2f}"
            )

            return run_id

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            raise

    def _load_data(
        self,
        instruments: List[str],
        timeframes:   List[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Dict:
        """Load historical candles from cache and compute features."""

        logger.info(f"Loading data for {instruments} {timeframes}")

        data_cache = {}

        if start_date:
            start_dt = datetime.fromisoformat(start_date).date()
        else:
            start_dt = datetime.utcnow().date() - timedelta(
                days=365 * self.settings.backtest_years
            )

        if end_date:
            end_dt = datetime.fromisoformat(end_date).date()
        else:
            end_dt = datetime.utcnow().date()

        # Handle both OneDrive and local paths
        data_dir = Path(self.settings.data_dir).resolve()
        logger.info(f"Data directory: {data_dir}")

        # Check if directory exists
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            raise ValueError(f"Data directory not found: {data_dir}")

        for instrument in instruments:
            for timeframe in timeframes:
                try:
                    cache_file = data_dir / f"{instrument}_{timeframe}.parquet"

                    logger.debug(f"Looking for: {cache_file}")

                    if not cache_file.exists():
                        logger.warning(f"Cache not found: {cache_file}")
                        continue

                    # Load raw OHLCV data
                    df = pd.read_parquet(str(cache_file))
                    df["time"] = pd.to_datetime(df["time"])

                    # ✅ COMPUTE FEATURES
                    df = self.feature_engine.compute_features(instrument, timeframe, df)

                    if len(df) == 0:
                        logger.warning(
                            f"No features computed for {instrument} {timeframe}"
                        )
                        continue

                    # Filter by date range
                    df = df[
                        (df["time"].dt.date >= start_dt)
                        & (df["time"].dt.date <= end_dt)
                    ]

                    if len(df) == 0:
                        logger.warning(f"No data in range for {instrument} {timeframe}")
                        continue

                    data_cache[(instrument, timeframe)] = df
                    logger.info(
                        f"  ✓ {instrument} {timeframe}: {len(df)} candles"
                    )

                except Exception as e:
                    logger.error(f"  ✗ {instrument} {timeframe}: {e}")

        if not data_cache:
            raise ValueError(f"No data loaded - check data directory: {data_dir}")

        logger.info(f"Total loaded: {len(data_cache)} instrument-timeframe pairs")
        return data_cache

    def _run_event_loop(
        self,
        data_cache: Dict,
        instruments: List[str],
        timeframes: List[str],
    ) -> List[TradeRecord]:
        """Main backtest event loop with adaptive trade limiting."""

        base_tf = "M5"
        if (instruments[0], base_tf) not in data_cache:
            base_tf = list(data_cache.keys())[0][1]

        base_data = data_cache[(instruments[0], base_tf)]
        logger.info(f"Event loop: {len(base_data)} candles")

        warmup_bars = 200
        
        # Pre-extract time arrays for faster lookup
        time_cache = {}
        index_cache = {}
        for key, df in data_cache.items():
            time_cache[key] = df["time"].values
            index_cache[key] = 0

        def advance_index(key, current_time):
            """Advance cached index to latest position at or before current_time."""
            time_arr = time_cache.get(key)
            if time_arr is None or len(time_arr) == 0:
                return -1, time_arr

            idx = index_cache.get(key, 0)
            if idx >= len(time_arr) or idx < 0:
                idx = 0

            if time_arr[idx] > current_time:
                idx = np.searchsorted(time_arr, current_time, side="right") - 1
                if idx < 0:
                    return -1, time_arr

            while idx + 1 < len(time_arr) and time_arr[idx + 1] <= current_time:
                idx += 1

            index_cache[key] = idx
            return idx, time_arr

        for idx in range(warmup_bars, len(base_data)):

            current_time = base_data.iloc[idx]["time"]
            current_date = current_time.date()

            if self.last_daily_reset is None or current_date > self.last_daily_reset:
                self.account.reset_daily(current_time)
                self.last_daily_reset = current_date
                self.daily_trade_count[current_date] = 0

                max_today = self.adaptive_limiter.calculate_max_trades(
                    self.account, self.closed_trades
                )
                logger.info(f"Daily reset: Max trades today = {max_today}")

            if (
                self.last_weekly_reset is None
                or (current_time - self.last_weekly_reset).days >= 7
            ):
                self.account.reset_weekly(current_time)
                self.last_weekly_reset = current_time

            max_trades_today = self.adaptive_limiter.calculate_max_trades(
                self.account, self.closed_trades
            )

            for instrument in instruments: 

                try:
                    features_dict = {}
                    for tf in timeframes:
                        key = (instrument, tf)
                        if key in data_cache:
                            # Use pre-computed time array for faster filtering
                            df = data_cache[key]
                            last_idx, time_arr = advance_index(key, current_time)
                            if last_idx == -1:
                                continue

                            tf_subset = df.iloc[: last_idx + 1]
                            if len(tf_subset) > 0:
                                features_dict[tf] = tf_subset

                    if not features_dict:
                        continue

                except Exception as e:
                    logger.warning(f"Error fetching features for {instrument}: {e}")
                    continue

                execution_tfs = ["M5", "M15", "H1"]
                for exec_tf in execution_tfs: 

                    if exec_tf not in features_dict: 
                        continue

                    features = features_dict[exec_tf]

                    if len(features) < 100:
                        continue

                    context = self._build_context(instrument, features, features_dict)

                    signals = []
                    for pattern in self.patterns:
                        # Track pattern fire rate
                        self.diagnostics["pattern_fire_rate"][pattern.pattern_id]["checked"] += 1
                        try:
                            sig = pattern.recognize(instrument, exec_tf, features, context)
                            if sig is not None:
                                signals.append(sig)
                                # Track per-pattern signal counts
                                self.diagnostics["pattern_signals_emitted"][pattern.pattern_id] += 1
                                self.diagnostics["pattern_fire_rate"][pattern.pattern_id]["fired"] += 1
                        except Exception as e:
                            logger.debug(f"Pattern {pattern.pattern_id} error: {e}")

                    if not signals:
                        continue
                    
                    # Track signals after context filters
                    self.diagnostics["signals_after_context_filters"] += 1

                    agg_result = self.aggregator.aggregate(signals, context)

                    if not agg_result.trade_allowed or agg_result.direction == 0:
                        continue
                    
                    # Track signals after NPES filter
                    self.diagnostics["signals_after_npes"] += 1

                    trades_today = self.daily_trade_count.get(current_date, 0)
                    allowed, reason = self.adaptive_limiter.should_take_trade(
                        trades_today, agg_result.confidence, max_trades_today
                    )

                    if not allowed: 
                        logger.debug(f"Trade rejected (adaptive): {reason}")
                        # Track confidence gate blocks
                        if "Confidence" in reason:
                            self.diagnostics["blocked_by_confidence_gate"] += 1
                        continue

                    try:
                        entry_price = float(features["close"].iloc[-1])

                        # Check if atr_14 exists
                        if "atr_14" in features.columns:
                            atr_14 = float(features["atr_14"].iloc[-1])
                        elif "atr" in features.columns:
                            atr_14 = float(features["atr"].iloc[-1])
                        else:
                            # Fallback: calculate simple ATR from high-low
                            atr_14 = float(
                                (features["high"].iloc[-1] - features["low"].iloc[-1]) * 1.5
                            )

                        if atr_14 == 0 or atr_14 < 0.0001:
                            continue

                        sl_price, tp_price, sl_atr, tp_atr = self._calculate_sl_tp(
                            instrument,
                            exec_tf,
                            agg_result.direction,
                            entry_price,
                            atr_14,
                            agg_result.contributing_patterns[0]
                            if agg_result.contributing_patterns
                            else "unknown",
                        )

                        units = self._calculate_position_size(entry_price, sl_price)
                        if units == 0:
                            continue

                        if agg_result.direction < 0:
                            units = -units

                        allowed, reason = self.risk_manager.validate_trade(
                            instrument, units, entry_price, sl_price
                        )

                        if not allowed:
                            logger.debug(f"Trade rejected: {reason}")
                            # Track risk blocks by reason
                            risk_reason = reason.split(":")[0] if ":" in reason else reason
                            self.diagnostics["blocked_by_risk"][risk_reason] = \
                                self.diagnostics["blocked_by_risk"].get(risk_reason, 0) + 1
                            continue

                        trade_id = self.broker.place_order(
                            instrument=instrument,
                            direction=agg_result.direction,
                            units=units,
                            entry_price=entry_price,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            atr_14=atr_14,
                            sl_atr=sl_atr,
                            tp_atr=tp_atr,
                            patterns=agg_result.contributing_patterns,
                            npes=agg_result.npes,
                            model_confidence=agg_result.confidence,
                            expected_value_R=agg_result.expected_value_R,
                            regime=context["volatility_regime"],
                            session=context["session"],
                            timeframe=exec_tf,
                            timestamp=current_time,
                            spread_pips=context["spread_pips"],
                            slippage_bps=self.settings.slippage_bps,
                        )

                        if trade_id:
                            self.account.add_position(
                                trade_id, instrument, units, entry_price
                            )
                            self.daily_trade_count[current_date] = (
                                self.daily_trade_count.get(current_date, 0) + 1
                            )
                            # Track executed trades
                            self.diagnostics["executed_trades"] += 1
                            # Track trades per day
                            date_str = str(current_date)
                            self.diagnostics["trades_per_day"][date_str] = \
                                self.diagnostics["trades_per_day"].get(date_str, 0) + 1
                            
                            logger.info(
                                f"Trade #{self.daily_trade_count[current_date]}/{max_trades_today}: "
                                f"{instrument} {agg_result.direction:+d} @ {entry_price:.5f}"
                            )

                    except Exception as e:
                        logger.warning(f"Trade execution error: {e}")

            # Update open positions with current prices - optimized
            for instrument in instruments:
                # Use the base timeframe for price updates (not hardcoded M5)
                key = (instrument, base_tf)
                if key not in data_cache:
                    continue

                price_idx, time_arr = advance_index(key, current_time)
                if price_idx == -1 or time_arr is None:
                    continue

                inst_data = data_cache[key]
                current_price = float(inst_data["close"].iloc[price_idx])

                # Only process trades for this instrument
                for trade in list(self.broker.open_trades.values()):
                    if trade.instrument == instrument:
                        result = self.broker.update_price(
                            trade.trade_id, current_price, current_time
                        )

                        if result:
                            _, trade_record = result

                            self.account.remove_position(
                                trade_record.trade_id,
                                trade_record.exit_price,
                                trade_record.pnl_usd,
                            )

                            self.account.update_equity(self.broker.balance, current_time)
                            self.closed_trades.append(trade_record)

                            if self.settings.enable_learning_updates:
                                self.learning_engine.record_trade(
                                    trade_record,
                                    trade_record.patterns_triggered,
                                    context.get("volatility_regime", "unknown"),
                                    context.get("session", "unknown"),
                                )

        return self.broker.closed_trades

    def _build_context(
        self, instrument: str, features: pd.DataFrame, features_dict: Dict
    ) -> Dict:
        """Build market context snapshot."""

        session = detect_session(features.iloc[-1]["time"])

        htf_trend = 0
        if "H4" in features_dict: 
            h4_features = features_dict["H4"]
            if len(h4_features) > 0:
                # Check if ema columns exist
                if "ema_20" in h4_features.columns and "ema_50" in h4_features.columns:
                    ema_20 = h4_features["ema_20"].iloc[-1]
                    ema_50 = h4_features["ema_50"].iloc[-1]
                    htf_trend = (
                        1 if ema_20 > ema_50 else -1 if ema_20 < ema_50 else 0
                    )
                else:
                    # Fallback: use close price with SMA
                    close_prices = h4_features["close"]
                    sma_20 = close_prices.tail(20).mean()
                    sma_50 = close_prices.tail(50).mean()
                    htf_trend = 1 if sma_20 > sma_50 else -1 if sma_20 < sma_50 else 0

        atr_pct = (
            features["atr_pctl_100"].iloc[-1]
            if "atr_pctl_100" in features.columns
            else 50
        )
        if atr_pct < 33:
            vol_regime = "low"
        elif atr_pct < 67:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        htf_range_pos = 0.5
        if "D" in features_dict or "H4" in features_dict: 
            key = "D" if "D" in features_dict else "H4"
            d1_features = features_dict[key]
            if d1_features is not None and len(d1_features) > 0:
                close = features.iloc[-1]["close"]
                d1_high = d1_features["high"].iloc[-1]
                d1_low = d1_features["low"].iloc[-1]
                d1_range = d1_high - d1_low
                if d1_range > 0:
                    htf_range_pos = (close - d1_low) / d1_range

        return {
            "session": session,
            "htf_trend": htf_trend,
            "volatility_regime": vol_regime,
            "current_drawdown": self.account.current_drawdown,
            "spread_pips": 1.5,
            "liquidity_state": "normal",
            "htf_range_position": htf_range_pos,
        }

    def _calculate_sl_tp(
        self,
        instrument: str,
        timeframe: str,
        direction:  int,
        entry_price: float,
        atr_14: float,
        setup_type: str,
    ) -> Tuple[float, float, float, float]:
        """Calculate SL and TP."""

        sl_atr = 1.5
        tp_atr = 2.5

        sl_atr = np.clip(sl_atr, 0.8, 2.8)
        tp_atr = np.clip(tp_atr, 1.0, 4.5)

        sl_distance = sl_atr * atr_14
        tp_distance = tp_atr * atr_14

        if direction > 0:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else: 
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return sl_price, tp_price, sl_atr, tp_atr

    def _calculate_position_size(self, entry_price: float, sl_price: float) -> int:
        """Calculate position size with leverage cap.
        
        Position size is determined by:
        1. Risk per trade (0.5% of account balance)
        2. Stop distance (entry - SL)
        3. Maximum leverage constraint (from risk_manager config)
        
        The final units is the minimum of risk-based sizing and leverage-capped sizing.
        """
        stop_distance = abs(entry_price - sl_price)
        if stop_distance == 0:
            return 0

        # Calculate units based on risk per trade
        risk_usd = self.account.current_balance * 0.005
        units_by_risk = int(risk_usd / stop_distance)

        # Cap units by maximum leverage
        max_leverage = self.risk_manager.max_leverage
        max_notional = self.account.current_balance * max_leverage
        max_units_by_leverage = int(max_notional / entry_price) if entry_price > 0 else 0

        # Use the smaller of the two to ensure both risk and leverage constraints are met
        units = min(units_by_risk, max_units_by_leverage)

        return max(units, 1)

    def _save_results(
        self,
        run_id: str,
        run_dir: Path,
        trades: List[TradeRecord],
        data_cache: Dict,
    ) -> None:
        """Save backtest results."""

        if trades:
            trades_data = [t.to_dict() for t in trades]
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_parquet(run_dir / "trades.parquet", index=False)
            logger.info(f"Saved {len(trades)} trades")

        with open(run_dir / "account_state.json", "w") as f:
            json.dump(self.account.to_dict(), f, indent=2)

        with open(run_dir / "settings.json", "w") as f:
            json.dump(self.settings.dict(), f, indent=2, default=str)

        metrics = self._compute_metrics(trades)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save diagnostic data (signal pipeline analysis)
        self._save_diagnostics(run_dir)

        logger.info(f"Results saved to {run_dir}")
    
    def _save_diagnostics(self, run_dir: Path) -> None:
        """Save diagnostic counters and signal funnel analysis."""
        
        # Calculate pattern fire rates
        pattern_fire_rates = {}
        for pid, stats in self.diagnostics["pattern_fire_rate"].items():
            checked = stats["checked"]
            fired = stats["fired"]
            rate = (fired / checked * 100) if checked > 0 else 0
            pattern_fire_rates[pid] = {
                "checked": checked,
                "fired": fired,
                "fire_rate_pct": round(rate, 2)
            }
        
        # Build signal funnel
        signal_funnel = {
            "pattern_signals_total": sum(self.diagnostics["pattern_signals_emitted"].values()),
            "signals_after_context_filters": self.diagnostics["signals_after_context_filters"],
            "signals_after_npes": self.diagnostics["signals_after_npes"],
            "blocked_by_confidence_gate": self.diagnostics["blocked_by_confidence_gate"],
            "blocked_by_risk_total": sum(self.diagnostics["blocked_by_risk"].values()),
            "blocked_by_risk_breakdown": self.diagnostics["blocked_by_risk"],
            "executed_trades": self.diagnostics["executed_trades"],
        }
        
        # Calculate survival rates
        total_signals = signal_funnel["pattern_signals_total"]
        if total_signals > 0:
            signal_funnel["survival_rate_npes"] = round(
                signal_funnel["signals_after_npes"] / total_signals * 100, 2
            )
            signal_funnel["survival_rate_final"] = round(
                signal_funnel["executed_trades"] / total_signals * 100, 2
            )
        
        diagnostics_output = {
            "signal_funnel": signal_funnel,
            "pattern_signals_emitted": self.diagnostics["pattern_signals_emitted"],
            "pattern_fire_rates": pattern_fire_rates,
            "blocked_by_risk": self.diagnostics["blocked_by_risk"],
            "trades_per_day": self.diagnostics["trades_per_day"],
        }
        
        with open(run_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics_output, f, indent=2)
        
        # Log summary
        logger.info(f"=== Signal Funnel Summary ===")
        logger.info(f"Pattern signals emitted: {signal_funnel['pattern_signals_total']}")
        logger.info(f"Signals after NPES: {signal_funnel['signals_after_npes']}")
        logger.info(f"Blocked by risk: {signal_funnel['blocked_by_risk_total']}")
        logger.info(f"Executed trades: {signal_funnel['executed_trades']}")

    def _compute_metrics(self, trades: List[TradeRecord]) -> Dict:
        """Compute backtest metrics."""

        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "expectancy_r": 0.0,
            }

        trades_df = pd.DataFrame([t.to_dict() for t in trades])

        wins = (trades_df["pnl_usd"] > 0).sum()
        losses = (trades_df["pnl_usd"] < 0).sum()
        win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0

        total_pnl = trades_df["pnl_usd"].sum()
        avg_pnl = total_pnl / len(trades_df) if len(trades_df) > 0 else 0
        expectancy_r = trades_df["pnl_atr"].mean()

        return {
            "total_trades": len(trades_df),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "total_pnl_usd": float(total_pnl),
            "avg_pnl_per_trade": float(avg_pnl),
            "expectancy_r": float(expectancy_r),
            "max_drawdown": float(self.account.max_drawdown),
            "final_balance": float(self.broker.balance),
        }
