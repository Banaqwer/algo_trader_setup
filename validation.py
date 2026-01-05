"""validation.py - utilities for validation harnesses (e.g., cost sensitivity)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backtester_complete import BacktesterComplete
from settings import settings


def run_cost_sensitivity(
    instruments: List[str],
    timeframes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    multipliers: Optional[List[float]] = None,
) -> List[dict]:
    """
    Execute a cost sensitivity sweep at different spread/slippage multipliers.

    Returns list of dictionaries containing multiplier, run_id, and metrics file path.
    """
    sweep = []
    multipliers = multipliers or [1.0, 1.25, 1.50]

    for factor in multipliers:
        run_settings = settings.copy(deep=True)
        run_settings.cost_multiplier = factor
        backtester = BacktesterComplete(run_settings)
        run_id = backtester.run(
            instruments=instruments,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
        )
        run_dir = Path(run_settings.runs_dir) / run_id
        metrics_path = run_dir / "metrics.json"
        sweep.append(
            {
                "cost_multiplier": factor,
                "run_id": run_id,
                "metrics_file": str(metrics_path),
            }
        )

    summary_path = (
        Path(settings.runs_dir)
        / f"cost_sensitivity_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(sweep, f, indent=2)

    return sweep
