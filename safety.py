"""safety.py"""

import logging

logger = logging.getLogger(__name__)

class ExecutionBlocked(Exception):
    """Raised when execution is blocked by safety rules."""
    pass

def check_execution_allowed(settings) -> bool:
    """Check if trade execution is allowed.  HARD RULE:  must pass ALL checks."""
    
    checks = [
        (settings.mode == "live", "MODE must be 'live'"),
        (settings.execution_enabled, "EXECUTION_ENABLED must be true"),
        (not settings.kill_switch, "KILL_SWITCH must be false"),
    ]
    
    for check, msg in checks:
        if not check: 
            logger.error(f"Execution blocked: {msg}")
            raise ExecutionBlocked(msg)
    
    return True

def block_execution_for_backtest(settings):
    """Verify backtest cannot execute trades."""
    if settings.can_execute:
        raise ExecutionBlocked(
            "FATAL: Backtester detected execution enabled. "
            "Set EXECUTION_ENABLED=false, MODE=backtest, KILL_SWITCH=true"
        )
    logger.info("Backtest execution gating:  OK (all orders will be simulated)")