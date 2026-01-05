#!/usr/bin/env python3
"""smoke_test.py - Quick 90-day sanity check"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "smoke_test. log"),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def run_smoke_test():
    """Run 90-day smoke test."""
    
    logger.info("="*70)
    logger.info("ALGO TRADER - SMOKE TEST (90 DAYS)")
    logger.info("="*70)
    
    try:
        from settings import settings
        from backtester_complete import BacktesterComplete
        
        logger.info("Initializing backtester...")
        backtester = BacktesterComplete(settings)
        
        logger.info("Running 90-day backtest...")
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=90)
        
        run_id = backtester.run(
            instruments=["EUR_USD"],
            timeframes=["M15"],
            start_date=start_date. isoformat(),
            end_date=end_date.isoformat()
        )

        manifest_path = Path(settings.runs_dir) / run_id / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not created: {manifest_path}")
        
        logger.info(f"✓ Smoke test complete! Run ID: {run_id}")
        logger.info(f"Results:  artifacts/runs/{run_id}/")
        logger.info("="*70)
        return True
        
    except Exception as e:
        logger.exception(f"✗ Smoke test failed: {e}")
        return False

if __name__ == "__main__": 
    success = run_smoke_test()
    sys.exit(0 if success else 1)
