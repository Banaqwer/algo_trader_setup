"""
Multi-Timeframe Trend Breakout Pattern
- Enters on breakout above 20-period high (M5)
- Confirms with uptrend on H1 (price > EMA 50)
- Risk/Reward ratio: 1:2
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class BreakoutPattern:
    """Multi-timeframe trend breakout pattern"""

    def __init__(self, config: Dict = None):
        """Initialize pattern with configuration"""
        self. name = "TREND_BREAKOUT"
        self.config = config or {}
        
        # Pattern parameters
        self.m5_period = 20  # M5 breakout period
        self.h1_ema_period = 50  # H1 trend confirmation
        self.min_atr_multiplier = 1.0  # Minimum volatility
        self.sl_atr_multiplier = 2.0  # SL distance = 2 x ATR
        self.tp_atr_multiplier = 4.0  # TP distance = 4 x ATR (1: 2 R/R)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close. shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()

    def analyze_m5(self, df_m5: pd.DataFrame) -> Dict:
        """Analyze M5 timeframe for breakout signals"""
        signals = {
            "breakout_up": False,
            "breakout_down": False,
            "high":  None,
            "low": None,
        }

        if len(df_m5) < self.m5_period:
            return signals

        # Calculate 20-period highs and lows
        high_20 = df_m5["high"].rolling(window=self. m5_period).max()
        low_20 = df_m5["low"].rolling(window=self.m5_period).min()

        current_high = high_20.iloc[-1]
        current_low = low_20.iloc[-1]
        current_close = df_m5["close"].iloc[-1]
        previous_close = df_m5["close"].iloc[-2]

        # Breakout above 20-period high
        if previous_close <= current_high and current_close > current_high:
            signals["breakout_up"] = True
            signals["high"] = current_high

        # Breakout below 20-period low
        if previous_close >= current_low and current_close < current_low:
            signals["breakout_down"] = True
            signals["low"] = current_low

        return signals

    def analyze_h1(self, df_h1: pd. DataFrame) -> Dict:
        """Analyze H1 timeframe for trend confirmation"""
        trend = {
            "uptrend": False,
            "downtrend":  False,
            "ema_50": None,
        }

        if len(df_h1) < self.h1_ema_period:
            return trend

        # Calculate EMA 50
        ema_50 = self.calculate_ema(df_h1["close"], self.h1_ema_period)
        current_ema = ema_50.iloc[-1]
        current_price = df_h1["close"]. iloc[-1]

        # Uptrend:  price above EMA 50
        if current_price > current_ema: 
            trend["uptrend"] = True

        # Downtrend: price below EMA 50
        if current_price < current_ema:
            trend["downtrend"] = True

        trend["ema_50"] = current_ema

        return trend

    def generate_signals(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> List[Dict]:
        """Generate buy/sell signals based on multi-timeframe analysis"""
        signals = []

        # Analyze each timeframe
        m5_signal = self.analyze_m5(df_m5)
        h1_trend = self.analyze_h1(df_h1)

        # Calculate ATR for risk management
        atr = self. calculate_atr(df_m5["high"], df_m5["low"], df_m5["close"])
        current_atr = atr.iloc[-1]

        if pd.isna(current_atr) or current_atr < 0.0001:
            return signals

        current_price = df_m5["close"].iloc[-1]

        # BUY SIGNAL:  Breakout up + H1 uptrend
        if m5_signal["breakout_up"] and h1_trend["uptrend"]:
            sl = current_price - (self.sl_atr_multiplier * current_atr)
            tp = current_price + (self. tp_atr_multiplier * current_atr)

            signals.append({
                "type": "BUY",
                "entry":  current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": self.tp_atr_multiplier / self.sl_atr_multiplier,
                "confidence": 0.75,
                "reason": "M5 breakout up + H1 uptrend",
            })

        # SELL SIGNAL: Breakout down + H1 downtrend
        if m5_signal["breakout_down"] and h1_trend["downtrend"]:
            sl = current_price + (self. sl_atr_multiplier * current_atr)
            tp = current_price - (self. tp_atr_multiplier * current_atr)

            signals.append({
                "type":  "SELL",
                "entry": current_price,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": self. tp_atr_multiplier / self.sl_atr_multiplier,
                "confidence": 0.75,
                "reason": "M5 breakout down + H1 downtrend",
            })

        return signals

    def validate_signal(self, signal: Dict, instrument: str, current_time: str) -> bool:
        """Validate signal before trading"""
        # Basic validation
        if signal. get("confidence", 0) < 0.7:
            return False

        if signal.get("risk_reward", 0) < 1.5:
            return False

        return True