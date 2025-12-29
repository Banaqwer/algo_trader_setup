"""Test settings loading."""

import pytest
from settings import settings

def test_settings_loaded():
    """Test settings are loaded correctly."""
    assert settings is not None
    assert settings.mode == "backtest"
    assert settings.execution_enabled == False
    assert settings.kill_switch == True

def test_instruments_parsed():
    """Test universe parsing."""
    instruments = settings.instruments
    assert isinstance(instruments, list)
    assert len(instruments) > 0
    assert "EUR_USD" in instruments

def test_can_execute_blocked():
    """Test execution is blocked for backtest."""
    assert not settings.can_execute

def test_directories_exist():
    """Test directories are created."""
    from pathlib import Path
    assert Path(settings.data_dir).exists()
    assert Path(settings.artifact_dir).exists()
    assert Path(settings.log_dir).exists()