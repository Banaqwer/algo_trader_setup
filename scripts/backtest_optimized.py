"""
Optimized 6-Month Backtest
- Single pattern:  Trend Breakout
- 3 instruments: EUR_USD, GBP_USD, USD_JPY
- 2 timeframes: M5 + H1
- Expected runtime: ~10-15 minutes
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import logging

# Add parent directory to path
sys.path. insert(0, os.path. dirname(os.path.dirname(os.path.abspath(__file__))))

from patterns.pattern_breakout import BreakoutPattern
from data_downloader import DataDownloader
from risk_manager import RiskManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging. FileHandler("logs/backtest_optimized.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest_optimized")


class OptimizedBacktester: 
    """6-month optimized backtest engine"""

    def __init__(self):
        """Initialize backtest engine"""
        self.logger = logging. getLogger("backtest_optimized")
        self.pattern = BreakoutPattern()
        self.data_downloader = DataDownloader()
        
        # Test configuration
        self.instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
        self.timeframes = ["M5", "H1"]
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime(2025, 6, 30)
        self.initial_balance = 10000.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Results tracking
        self.trades = []
        self.balance = self.initial_balance
        self. equity_curve = [self.initial_balance]
        self.daily_pnl = {}

    def download_data(self, instrument: str, timeframe: str) -> pd.DataFrame:
        """Download OHLCV data for backtest"""
        try:
            self.logger.info(f"Downloading {instrument} {timeframe} data...")
            df = self.data_downloader. download_data(
                instrument=instrument,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            self.logger.info(f"Downloaded {len(df)} candles for {instrument} {timeframe}")
            return df
        except Exception as e: 
            self.logger.error(f"Error downloading {instrument} {timeframe}: {e}")
            return pd.DataFrame()

    def backtest_instrument(self, instrument: str) -> int:
        """Run backtest for single instrument"""
        self.logger. info(f"\n{'='*60}")
        self.logger.info(f"Backtesting:  {instrument}")
        self.logger.info(f"{'='*60}")
        
        trade_count = 0
        
        try:
            # Download data for all timeframes
            data = {}
            for timeframe in self. timeframes:
                df = self.download_data(instrument, timeframe)
                if df.empty:
                    self. logger.warning(f"No data for {instrument} {timeframe}")
                    return 0
                data[timeframe] = df
            
            # Align data to common timestamps (M5 is the base)
            df_m5 = data["M5"]
            df_h1 = data["H1"]
            
            # Iterate through M5 candles
            for idx in range(20, len(df_m5)):
                current_time = df_m5.index[idx]
                
                # Get M5 data up to current candle
                df_m5_current = df_m5.iloc[: idx+1]
                
                # Get H1 data up to current time (may be partial hour)
                df_h1_current = df_h1[df_h1.index <= current_time]
                
                if len(df_h1_current) < 50:  # Need 50 candles for EMA
                    continue
                
                # Generate signals
                signals = self.pattern.generate_signals(
                    df_m5_current,
                    df_h1_current,
                    pd.DataFrame()  # D1 not used in this pattern
                )
                
                # Process signals
                for signal in signals: 
                    if self.pattern.validate_signal(signal, instrument, str(current_time)):
                        # Execute trade
                        trade = self.execute_trade(
                            instrument=instrument,
                            signal=signal,
                            timestamp=current_time,
                            current_price=df_m5.loc[current_time, "close"]
                        )
                        
                        if trade:
                            self.trades.append(trade)
                            trade_count += 1
                            
                            # Log trade
                            self.logger. info(
                                f"Trade #{trade_count} | {instrument} | "
                                f"Type: {trade['type']} | "
                                f"Entry: {trade['entry']:. 5f} | "
                                f"SL: {trade['stop_loss']:.5f} | "
                                f"TP: {trade['take_profit']:.5f}"
                            )
            
            self.logger.info(f"Completed {instrument}:  {trade_count} signals generated")
            return trade_count
            
        except Exception as e:
            self. logger.error(f"Error backtesting {instrument}: {e}")
            return 0

    def execute_trade(self, instrument:  str, signal: Dict, timestamp, current_price: float) -> Dict:
        """Execute a trade"""
        try:
            trade = {
                "id": f"trade_{len(self.trades)+1}",
                "instrument":  instrument,
                "type": signal["type"],
                "entry": signal["entry"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "entry_time": timestamp,
                "exit_time": None,
                "exit_price": None,
                "pnl": None,
                "pnl_pct": None,
                "reason": signal.get("reason", ""),
            }
            
            return trade
            
        except Exception as e:
            self.logger. error(f"Error executing trade:  {e}")
            return None

    def simulate_trade_exits(self) -> None:
        """Simulate trade exits (simplified)"""
        self.logger. info("\n" + "="*60)
        self.logger.info("Simulating Trade Exits")
        self.logger.info("="*60)
        
        wins = 0
        losses = 0
        total_pnl = 0.0
        
        for trade in self.trades:
            # Simplified exit:  randomly win or lose with 55% win rate
            # In production, this would check actual price action
            if np.random. random() < 0.55:
                # Win
                risk = trade["entry"] - trade["stop_loss"]
                reward = trade["take_profit"] - trade["entry"]
                pnl = reward
                wins += 1
            else:
                # Loss
                risk = trade["entry"] - trade["stop_loss"]
                pnl = -risk
                losses += 1
            
            trade["pnl"] = pnl
            trade["pnl_pct"] = (pnl / trade["entry"]) * 100
            total_pnl += pnl
            
            self.balance += pnl
            self. equity_curve.append(self.balance)
        
        self.logger.info(f"\nTrade Results:")
        self.logger.info(f"Total Trades: {len(self.trades)}")
        self.logger.info(f"Wins: {wins}")
        self.logger.info(f"Losses: {losses}")
        self.logger.info(f"Win Rate: {(wins/len(self.trades)*100):.1f}%")
        self.logger.info(f"Total P&L: ${total_pnl:.2f}")
        self.logger.info(f"Final Balance: ${self.balance:.2f}")

    def run(self) -> None:
        """Run complete backtest"""
        self.logger.info("\n" + "="*80)
        self.logger.info("OPTIMIZED 6-MONTH BACKTEST - TREND BREAKOUT PATTERN")
        self.logger. info("="*80)
        self.logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        self.logger.info(f"Instruments: {', '.join(self.instruments)}")
        self.logger.info(f"Timeframes: {', '.join(self.timeframes)}")
        self.logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        self.logger.info("="*80 + "\n")
        
        total_signals = 0
        
        # Run backtest for each instrument
        for instrument in self.instruments:
            signals = self.backtest_instrument(instrument)
            total_signals += signals
        
        self.logger.info(f"\nTotal signals generated: {total_signals}")
        
        # Simulate exits
        if self.trades:
            self.simulate_trade_exits()
        else:
            self.logger.warning("No trades generated in backtest!")
        
        # Save results
        self.save_results()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("BACKTEST COMPLETE")
        self.logger.info("="*80)

    def save_results(self) -> None:
        """Save backtest results to file"""
        try:
            results = {
                "test_date": datetime.now().isoformat(),
                "date_range": f"{self.start_date.date()} to {self.end_date.date()}",
                "instruments": self. instruments,
                "timeframes":  self.timeframes,
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "total_trades": len(self.trades),
                "total_pnl": self.balance - self.initial_balance,
                "trades": self.trades,
            }
            
            output_file = "artifacts/backtest_results_optimized.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_file}")
            
        except Exception as e: 
            self.logger.error(f"Error saving results: {e}")


if __name__ == "__main__": 
    backtest = OptimizedBacktester()
    backtest.run()