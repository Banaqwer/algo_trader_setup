"""self_learning.py"""

import logging
import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging. getLogger(__name__)

class PatternStatBucket:
    """Statistics for a pattern in a specific context."""
    
    def __init__(self, instrument: str, timeframe: str, pattern_id: str,
                 regime: str, session:  str):
        self.instrument = instrument
        self.timeframe = timeframe
        self.pattern_id = pattern_id
        self.regime = regime
        self.session = session
        
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl_usd = 0.0
        self.total_pnl_atr = 0.0
        
        self.recent_trades = []
        self.recent_window = 30
        
        self.win_rate = 0.5
        self.expectancy_usd = 0.0
        self.expectancy_r = 0.0
        self.confidence = 0.5
        self.enabled = True
    
    def add_trade(self, pnl_usd: float, pnl_atr: float):
        """Record a trade outcome."""
        
        self.trade_count += 1
        if pnl_usd > 0:
            self.win_count += 1
        elif pnl_usd < 0:
            self.loss_count += 1
        
        self. total_pnl_usd += pnl_usd
        self.total_pnl_atr += pnl_atr
        
        self. recent_trades.append(pnl_usd)
        if len(self.recent_trades) > self.recent_window:
            self.recent_trades.pop(0)
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Recompute derived metrics."""
        
        if self.trade_count == 0:
            self.win_rate = 0.5
            self.expectancy_usd = 0.0
            self.expectancy_r = 0.0
            return
        
        self.win_rate = self.win_count / self.trade_count
        self.expectancy_usd = self.total_pnl_usd / self.trade_count
        self.expectancy_r = self.total_pnl_atr / self. trade_count
        
        confidence = min(0.9, 0.5 + (self.win_rate - 0.5) * 0.6)
        confidence *= min(1.0, self.trade_count / 50.0)
        self.confidence = max(0.3, confidence)
        
        if len(self.recent_trades) >= 10:
            recent_loss_count = sum(1 for p in self.recent_trades[-10:] if p < 0)
            if recent_loss_count >= 9:
                self.enabled = False
                logger.warning(f"Pattern {self.pattern_id} disabled (9/10 losses)")
    
    def to_dict(self) -> Dict:
        """Export as dict."""
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "pattern_id": self.pattern_id,
            "regime": self.regime,
            "session": self.session,
            "trade_count":  self.trade_count,
            "win_rate": self.win_rate,
            "expectancy_usd":  self.expectancy_usd,
            "expectancy_r":  self.expectancy_r,
            "confidence": self.confidence,
            "enabled": self.enabled,
        }

class SelfLearningEngine:
    """Updates pattern statistics from closed trades."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "pattern_stats.db")
        
        self.buckets:  Dict[Tuple, PatternStatBucket] = {}
        
        self._init_db()
        self._load_buckets()
        
        logger.info(f"SelfLearningEngine initialized:   {len(self.buckets)} buckets loaded")
    
    def _init_db(self):
        """Initialize SQLite database."""
        
        conn = sqlite3.connect(self. db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_stats (
                id INTEGER PRIMARY KEY,
                instrument TEXT,
                timeframe TEXT,
                pattern_id TEXT,
                regime TEXT,
                session TEXT,
                trade_count INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                total_pnl_usd REAL,
                total_pnl_atr REAL,
                win_rate REAL,
                expectancy_usd REAL,
                expectancy_r REAL,
                confidence REAL,
                enabled INTEGER,
                updated_at TEXT,
                UNIQUE(instrument, timeframe, pattern_id, regime, session)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_buckets(self):
        """Load buckets from database."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM pattern_stats")
            rows = cursor.fetchall()
            
            for row in rows:
                (_, instrument, timeframe, pattern_id, regime, session,
                 trade_count, win_count, loss_count, total_pnl_usd, total_pnl_atr,
                 win_rate, expectancy_usd, expectancy_r, confidence, enabled, _) = row
                
                bucket = PatternStatBucket(instrument, timeframe, pattern_id, regime, session)
                bucket.trade_count = trade_count
                bucket.win_count = win_count
                bucket.loss_count = loss_count
                bucket. total_pnl_usd = total_pnl_usd
                bucket.total_pnl_atr = total_pnl_atr
                bucket.win_rate = win_rate
                bucket.expectancy_usd = expectancy_usd
                bucket.expectancy_r = expectancy_r
                bucket.confidence = confidence
                bucket. enabled = bool(enabled)
                
                key = (instrument, timeframe, pattern_id, regime, session)
                self. buckets[key] = bucket
            
            conn.close()
        
        except Exception as e:
            logger.warning(f"Could not load buckets: {e}")
    
    def record_trade(self, trade_record, patterns_triggered: List[str],
                    regime: str, session: str):
        """Record a closed trade and update pattern stats."""
        
        instrument = trade_record. instrument
        timeframe = trade_record.entry_timeframe
        pnl_usd = trade_record.pnl_usd
        pnl_atr = trade_record.pnl_atr
        
        for pattern_id in patterns_triggered: 
            key = (instrument, timeframe, pattern_id, regime, session)
            
            if key not in self.buckets:
                bucket = PatternStatBucket(instrument, timeframe, pattern_id, regime, session)
                self. buckets[key] = bucket
            else:
                bucket = self.buckets[key]
            
            bucket.add_trade(pnl_usd, pnl_atr)
        
        self._save_buckets()
    
    def _save_buckets(self):
        """Persist buckets to database."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for bucket in self.buckets.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO pattern_stats 
                    (instrument, timeframe, pattern_id, regime, session,
                     trade_count, win_count, loss_count, total_pnl_usd, total_pnl_atr,
                     win_rate, expectancy_usd, expectancy_r, confidence, enabled, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bucket.instrument, bucket.timeframe, bucket.pattern_id,
                    bucket. regime, bucket.session,
                    bucket.trade_count, bucket.win_count, bucket.loss_count,
                    bucket. total_pnl_usd, bucket.total_pnl_atr,
                    bucket.win_rate, bucket.expectancy_usd, bucket.expectancy_r,
                    bucket.confidence, int(bucket.enabled),
                    datetime. utcnow().isoformat()
                ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Failed to save buckets: {e}")
    
    def get_expectancy(self, instrument: str, timeframe: str, pattern_id: str,
                      regime: str, session: str) -> Tuple[float, float, int]:
        """Get cached expectancy for a pattern in context."""
        
        key = (instrument, timeframe, pattern_id, regime, session)
        
        if key in self.buckets:
            bucket = self.buckets[key]
            return bucket.expectancy_r, bucket.win_rate, bucket.trade_count
        
        return 0.0, 0.5, 0
    
    def is_pattern_enabled(self, instrument: str, timeframe: str, pattern_id: str,
                          regime: str, session: str) -> bool:
        """Check if pattern is enabled in this context."""
        
        key = (instrument, timeframe, pattern_id, regime, session)
        
        if key in self.buckets:
            return self.buckets[key].enabled
        
        return True