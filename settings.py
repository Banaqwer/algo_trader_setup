"""settings.py - Central configuration for algo trader with adaptive trading support"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator, root_validator
import yaml
from copy import deepcopy

# Load .env file
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Central configuration for algo trader."""
    
    # Broker/Data
    oanda_access_token: str = Field(default="", env="OANDA_ACCESS_TOKEN")
    oanda_account_id: str = Field(default="", env="OANDA_ACCOUNT_ID")
    oanda_env: str = Field(default="practice", env="OANDA_ENV")
    universe: str = Field(
        default="EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,USD_CHF,NZD_USD",
        env="UNIVERSE"
    )
    price_type: str = Field(default="M", env="PRICE_TYPE")
    
    # Backtest Scope
    backtest_years: int = Field(default=5, env="BACKTEST_YEARS")
    start_date: Optional[str] = Field(default=None, env="START_DATE")
    end_date: Optional[str] = Field(default=None, env="END_DATE")
    
    # Directories - use local ./data folder
    data_dir: str = Field(default="./data", env="DATA_DIR")
    artifact_dir: str = Field(default="./artifacts", env="ARTIFACT_DIR")
    log_dir: str = Field(default="./logs", env="LOG_DIR")
    
    # Execution Mode & Safety
    mode: str = Field(default="backtest", env="MODE")
    execution_enabled: bool = Field(default=False, env="EXECUTION_ENABLED")
    kill_switch: bool = Field(default=True, env="KILL_SWITCH")
    
    # Backtest Realism
    apply_spread: bool = Field(default=True, env="APPLY_SPREAD")
    apply_slippage: bool = Field(default=True, env="APPLY_SLIPPAGE")
    slippage_bps: float = Field(default=1.5, env="SLIPPAGE_BPS")
    base_spread_pips: float = Field(default=1.5, env="BASE_SPREAD_PIPS")
    cost_multiplier: float = Field(default=1.0, env="COST_MULTIPLIER")
    spread_filter_pctl: int = Field(default=80, env="SPREAD_FILTER_PCTL")
    
    # Adaptive Trading Policy
    adaptive_trading: bool = Field(default=True, env="ADAPTIVE_TRADING")
    base_max_trades_per_day: int = Field(default=10, env="BASE_MAX_TRADES_PER_DAY")
    min_trades_per_day: int = Field(default=2, env="MIN_TRADES_PER_DAY")
    max_trades_per_day:  int = Field(default=20, env="MAX_TRADES_PER_DAY")
    
    # Confidence thresholds
    min_confidence_to_trade: float = Field(default=0.55, env="MIN_CONFIDENCE_TO_TRADE")
    confidence_multiplier: float = Field(default=0.5, env="CONFIDENCE_MULTIPLIER")
    
    # Fallback if adaptive disabled
    max_trades_per_day_fixed: int = Field(default=10, env="MAX_TRADES_PER_DAY_FIXED")

    # Pattern toggles (used for diagnostics of non-firing patterns)
    enable_session_bias_pattern: bool = Field(default=False, env="ENABLE_SESSION_BIAS_PATTERN")
    enable_correlation_divergence_pattern: bool = Field(default=False, env="ENABLE_CORRELATION_DIVERGENCE_PATTERN")
    enable_htf_range_pattern: bool = Field(default=True, env="ENABLE_HTF_RANGE_PATTERN")
    
    # Classic trade policy
    block_rollover:  bool = Field(default=True, env="BLOCK_ROLLOVER")
    rollover_window_utc: str = Field(default="21:55-22:10", env="ROLLOVER_WINDOW_UTC")
    block_weekend: bool = Field(default=True, env="BLOCK_WEEKEND")
    weekend_cutoff_utc: str = Field(default="Fri 20:00", env="WEEKEND_CUTOFF_UTC")
    
    # Learning
    enable_learning_updates: bool = Field(default=True, env="ENABLE_LEARNING_UPDATES")
    min_bucket_trades_for_update: int = Field(default=30, env="MIN_BUCKET_TRADES_FOR_UPDATE")
    
    # Walk-forward
    walkforward_train_years: int = Field(default=4, env="WALKFORWARD_TRAIN_YEARS")
    walkforward_test_months: int = Field(default=6, env="WALKFORWARD_TEST_MONTHS")
    walkforward_step_months: int = Field(default=3, env="WALKFORWARD_STEP_MONTHS")
    walkforward_purge_days: int = Field(default=5, env="WALKFORWARD_PURGE_DAYS")
    walkforward_embargo_days: int = Field(default=2, env="WALKFORWARD_EMBARGO_DAYS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create directories
        for d in [self.data_dir, self.artifact_dir, self. log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # Log API key status
        api_key_status = "✓ SET" if self.oanda_access_token else "✗ NOT SET"
        account_status = "✓ SET" if self.oanda_account_id else "✗ NOT SET"
        
        logger.info(
            f"Settings loaded: mode={self.mode}, backtest_years={self.backtest_years}, "
            f"adaptive_trading={self.adaptive_trading}, "
            f"data_dir={self.data_dir}, "
            f"OANDA_API_KEY {api_key_status}, OANDA_ACCOUNT_ID {account_status}"
        )
    
    @root_validator(pre=True)
    def clean_null_values(cls, values):
        """Remove any null/None values and convert to proper types."""
        for key, value in list(values.items()):
            # Handle null strings and None
            if value == 'null' or value == 'None' or value is None:
                if key in values:
                    del values[key]
                continue
            
            # Skip empty strings
            if value == '':  
                if key in values:
                    del values[key]
                continue
            
            # Convert integer fields
            if key in [
                'backtest_years', 'base_max_trades_per_day', 'min_trades_per_day',
                'max_trades_per_day', 'spread_filter_pctl', 'min_bucket_trades_for_update',
                'walkforward_train_years', 'walkforward_test_months', 'walkforward_step_months',
                'max_trades_per_day_fixed'
            ]:
                try:
                    values[key] = int(value)
                except (ValueError, TypeError):
                    if key in values:
                        del values[key]
            
            # Convert float fields
            elif key in ['slippage_bps', 'min_confidence_to_trade', 'confidence_multiplier']:
                try:
                    values[key] = float(value)
                except (ValueError, TypeError):
                    if key in values:
                        del values[key]
            elif key in ['base_spread_pips', 'cost_multiplier']:
                try:
                    values[key] = float(value)
                except (ValueError, TypeError):
                    if key in values:
                        del values[key]
        
        return values
    
    @validator("mode")
    def validate_mode(cls, v):
        if v not in ["backtest", "paper", "live"]:
            raise ValueError(f"mode must be backtest, paper, or live, got {v}")
        return v
    
    @validator("oanda_env")
    def validate_oanda_env(cls, v):
        if v not in ["practice", "live"]:  
            raise ValueError(f"oanda_env must be practice or live, got {v}")
        return v
    
    @validator("min_confidence_to_trade")
    def validate_confidence(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"min_confidence_to_trade must be 0.0-1.0, got {v}")
        return v
    
    @validator("start_date", "end_date", pre=True, always=True)
    def validate_dates(cls, v):
        """Convert null/None strings to None."""
        if v is None or v == 'null' or v == '':
            return None
        return v
    
    @property
    def instruments(self) -> List[str]:
        """Parse universe into list of instruments."""
        return [s.strip() for s in self.universe.split(",") if s.strip()]
    
    @property
    def can_execute(self) -> bool:
        """Check if actual execution is allowed."""
        return (
            self.mode == "live" and
            self.execution_enabled and
            not self.kill_switch
        )
    
    @property
    def data_processed_dir(self) -> str:
        return os.path.join(self.data_dir, "processed")
    
    @property
    def data_coverage_dir(self) -> str:
        return os.path.join(self.data_dir, "coverage")
    
    @property
    def runs_dir(self) -> str:
        return os.path.join(self.artifact_dir, "runs")
    
    def dict(self):
        """Export settings as dict (masking sensitive info)."""
        return {
            "oanda_access_token":   "***REDACTED***" if self.oanda_access_token else "NOT SET",
            "oanda_account_id":   self.oanda_account_id[: 8] + "..." if self.oanda_account_id else "NOT SET",
            "oanda_env":  self. oanda_env,
            "universe": self.universe,
            "backtest_years":   self.backtest_years,
            "mode": self.mode,
            "execution_enabled":  self.execution_enabled,
            "kill_switch": self.kill_switch,
            "apply_spread": self.apply_spread,
            "apply_slippage":   self.apply_slippage,
            "slippage_bps": self.slippage_bps,
            "base_spread_pips": self.base_spread_pips,
            "cost_multiplier": self.cost_multiplier,
            "adaptive_trading": self.adaptive_trading,
            "base_max_trades_per_day": self. base_max_trades_per_day,
            "min_trades_per_day": self.min_trades_per_day,
            "max_trades_per_day": self.max_trades_per_day,
            "min_confidence_to_trade": self.min_confidence_to_trade,
            "log_level":   self.log_level,
            "enable_session_bias_pattern": self.enable_session_bias_pattern,
            "enable_correlation_divergence_pattern": self.enable_correlation_divergence_pattern,
            "enable_htf_range_pattern": self.enable_htf_range_pattern,
            "walkforward": {
                "train_years": self.walkforward_train_years,
                "test_months": self.walkforward_test_months,
                "step_months": self.walkforward_step_months,
                "purge_days": self.walkforward_purge_days,
                "embargo_days": self.walkforward_embargo_days,
            },
        }
    
    class Config: 
        env_file = ".env"
        case_sensitive = False


def load_config(config_file: Optional[str] = None, **overrides) -> Settings:
    """Load config from env vars, YAML file, and overrides."""
    settings_dict = {}
    
    yaml_path = config_file or "./config.yaml"
    if os.path.exists(yaml_path):
        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            settings_dict.update(yaml_config)
    
    settings_dict.update(overrides)
    return Settings(**settings_dict)


settings = load_config()
