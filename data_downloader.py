import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class DataDownloader:
    """Download historical OHLC data from OANDA."""
    
    def __init__(self, settings):
        self.settings = settings
        self.api_key = os.getenv("OANDA_ACCESS_TOKEN")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.base_url = "https://api-fxpractice.oanda.com"  # Practice server
        self.data_dir = Path(settings.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        api_status = "✓ SET" if self.api_key else "✗ NOT SET"
        logger.info(
            f"DataDownloader initialized: data_dir={self.data_dir}, API_KEY {api_status}"
        )
    
    def download(self, instrument: str, timeframe: str, years: int = 5):
        """Download historical candles from OANDA."""
        
        if not self.api_key:
            logger.warning("OANDA_ACCESS_TOKEN not set.   Skipping download.")
            return False
        
        try:
            logger.info(f"Downloading {instrument} {timeframe} ({years} years)...")
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365 * years)
            
            candles = self._fetch_candles(instrument, timeframe, start_date, end_date)
            
            if not candles: 
                logger.warning(f"No candles received for {instrument} {timeframe}")
                return False
            
            df = pd.DataFrame(candles)
            
            # Save to parquet
            cache_file = self._get_cache_path(instrument, timeframe)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)
            
            logger.info(f"Saved {len(df)} candles to {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {instrument} {timeframe}:  {e}")
            return False
    
    def _fetch_candles(
        self,
        instrument: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """Fetch candles from OANDA API."""
        
        candles = []
        current = start_date
        
        # OANDA limits to 5000 candles per request
        tf_minutes = self._timeframe_to_minutes(timeframe)
        max_candles_per_request = 5000
        chunk_days = (max_candles_per_request * tf_minutes) // (24 * 60)
        
        while current < end_date:
            chunk_end = min(current + timedelta(days=chunk_days), end_date)
            
            try:
                url = f"{self.base_url}/v3/instruments/{instrument}/candles"
                params = {
                    "price":   "MBA",
                    "granularity": timeframe,
                    "from":  current.isoformat() + "Z",
                    "to": chunk_end.isoformat() + "Z",
                }
                headers = {
                    "Authorization":  f"Bearer {self.api_key}",
                    "Accept-Datetime-Format": "UNIX",
                }
                
                response = requests. get(
                    url, params=params, headers=headers, timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "candles" in data: 
                        for candle in data["candles"]:
                            candles.append({
                                "time": pd.to_datetime(float(candle["time"]), unit='s'),
                                "open":  float(candle["mid"]["o"]),
                                "high":  float(candle["mid"]["h"]),
                                "low":  float(candle["mid"]["l"]),
                                "close":  float(candle["mid"]["c"]),
                                "volume":  int(candle. get("volume", 0)),
                            })
                        logger. debug(
                            f"  {instrument} {timeframe} {current.date()}: "
                            f"{len(candles)} total"
                        )
                elif response.status_code == 401:
                    logger.error(
                        "OANDA authentication failed. "
                        "Check API_KEY and ACCOUNT_ID."
                    )
                    return []
                else: 
                    logger.warning(
                        f"API returned {response. status_code}:  {response.text}"
                    )
                    
            except Exception as e:
                logger.warning(f"Chunk fetch failed ({current. date()}): {e}")
            
            current = chunk_end
        
        return candles
    
    def _timeframe_to_minutes(self, timeframe:  str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "M1": 1,
            "M2": 2,
            "M4": 4,
            "M5": 5,
            "M10": 10,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H2": 120,
            "H3":  180,
            "H4": 240,
            "H6": 360,
            "H8": 480,
            "H12": 720,
            "D":  1440,
            "W": 10080,
            "M":  43200,
        }
        return mapping.get(timeframe, 60)
    
    def _get_cache_path(self, instrument: str, timeframe: str) -> Path:
        """Get cache file path for instrument/timeframe."""
        return self.data_dir / f"{instrument}_{timeframe}.parquet"
    
    @staticmethod
    def load_candles(
        instrument:  str,
        timeframe: str,
        start_date,
        end_date,
        data_dir: str,
    ) -> pd.DataFrame:
        """Load cached candles."""
        
        cache_file = Path(data_dir).resolve() / f"{instrument}_{timeframe}.parquet"
        
        if not cache_file.exists():
            logger.warning(f"Cache not found:  {cache_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(cache_file)
            df["time"] = pd.to_datetime(df["time"])
            
            # Convert string dates to datetime. date if provided
            if start_date:
                start_dt = datetime.fromisoformat(start_date).date() if isinstance(start_date, str) else start_date
            else:
                start_dt = None
                
            if end_date: 
                end_dt = datetime.fromisoformat(end_date).date() if isinstance(end_date, str) else end_date
            else:
                end_dt = None
            
            # Filter by date range
            if start_dt and end_dt: 
                df = df[
                    (df["time"]. dt.date >= start_dt) & 
                    (df["time"].dt.date <= end_dt)
                ]
            elif start_dt:
                df = df[df["time"].dt. date >= start_dt]
            elif end_dt:
                df = df[df["time"].dt.date <= end_dt]
            
            logger.debug(
                f"Loaded {len(df)} candles for {instrument} {timeframe}"
            )
            return df
            
        except Exception as e: 
            logger.error(f"Load failed for {instrument} {timeframe}: {e}")
            return pd.DataFrame()