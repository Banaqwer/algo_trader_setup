"""data_ingestion.py"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CandleBuffer:
    """Per-timeframe candle buffer."""
    
    def __init__(self, instrument: str, timeframe: str, max_bars: int = 2500):
        self.instrument = instrument
        self.timeframe = timeframe
        self.max_bars = max_bars
        self. data = pd.DataFrame()
    
    def add_candle(self, candle: dict):
        """Add a candle to buffer."""
        if self.data. empty:
            self.data = pd.DataFrame([candle])
        else:
            new_df = pd.DataFrame([candle])
            self.data = pd.concat([self. data, new_df], ignore_index=True)
        
        if len(self.data) > self.max_bars:
            self.data = self. data.iloc[-self.max_bars:].reset_index(drop=True)
    
    def get_data(self, last_n:  int = None) -> pd.DataFrame:
        """Retrieve data."""
        if last_n: 
            return self.data.tail(last_n).copy()
        return self.data. copy()
    
    def get_bar_count(self) -> int:
        """Number of bars in buffer."""
        return len(self.data)

class DataManager:
    """Central data store for all instruments and timeframes."""
    
    def __init__(self, instruments: list, timeframes: list):
        self.instruments = instruments
        self.timeframes = timeframes
        self.buffers = {}
        
        for inst in instruments:
            for tf in timeframes:
                key = (inst, tf)
                self.buffers[key] = CandleBuffer(inst, tf)
        
        logger.info(f"DataManager initialized:  {len(self.buffers)} buffers")
    
    def get_data(self, instrument: str, timeframe: str, last_n:  int = None) -> pd.DataFrame:
        """Retrieve candle data."""
        key = (instrument, timeframe)
        if key not in self.buffers:
            raise ValueError(f"Unknown instrument/timeframe: {key}")
        return self.buffers[key].get_data(last_n=last_n)
    
    def get_latest_bar(self, instrument: str, timeframe: str):
        """Get the latest candle."""
        data = self.get_data(instrument, timeframe, last_n=1)
        return data.iloc[0] if len(data) > 0 else None
    
    def get_bar_count(self, instrument:  str, timeframe: str) -> int:
        """Number of bars available."""
        key = (instrument, timeframe)
        return self. buffers[key].get_bar_count()
    
    def on_new_candle(self, instrument:  str, timeframe: str, candle: dict):
        """Called when a new candle arrives."""
        key = (instrument, timeframe)
        if key in self.buffers:
            self.buffers[key].add_candle(candle)