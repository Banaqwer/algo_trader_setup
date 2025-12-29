"""main.py"""

import sys
import logging
from pathlib import Path
from datetime import datetime

from cli import app
from settings import settings

log_dir = Path(settings.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "algo_trader.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("="*70)
    logger.info("ALGO TRADER - PRODUCTION ALGORITHMIC TRADING SYSTEM")
    logger.info(f"Version: 1.0.0 | Started: {datetime.utcnow().isoformat()}")
    logger.info("="*70)
    
    try:
        app()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e: 
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__": 
    main()