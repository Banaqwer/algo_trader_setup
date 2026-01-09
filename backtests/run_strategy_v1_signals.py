"""Run Strategy V1 over EUR/USD and write outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_loader import load_eurusd_timeframes
from strategies import StrategyV1Config, run_strategy_v1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_eurusd_timeframes(m5_csv_path="data/EUR_USD_M5.csv")
    h1 = data["H1"]
    h4 = data["H4"]
    d1 = data["D1"]

    trades, summary = run_strategy_v1(h1, h4, d1, StrategyV1Config())

    output_dir = Path("backtests/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_path = output_dir / "trades_v1.csv"
    summary_path = output_dir / "summary_v1.json"

    trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Strategy V1 Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
