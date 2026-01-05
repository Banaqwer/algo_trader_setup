import typer
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd

from settings import settings
from backtester_complete import BacktesterComplete
from reporting_complete import ReportGenerator
from data_downloader import DataDownloader
from walkforward import generate_walkforward_windows
from validation import run_cost_sensitivity

app = typer.Typer(help="Algo Trader CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


@app.command()
def print_config():
    """Print current configuration."""
    logger.info("ALGO TRADER - CURRENT CONFIGURATION")
    logger.info("")
    for key, value in settings.dict().items():
        logger.info(f"{key: <40} {value}")


@app.command()
def smoke_test():
    """Run quick 90-day smoke test."""
    from scripts.smoke_test import run_smoke_test
    success = run_smoke_test()
    if not success:
        raise typer.Exit(code=1)


@app.command()
def backtest(
    years: int = typer.Option(5, help="Number of years to backtest"),
    universe: Optional[str] = typer.Option(
        None, help="Instruments (comma-separated)"
    ),
    tfs: Optional[str] = typer.Option(
        "M5,M15,H1", help="Timeframes (comma-separated)"
    ),
):
    """Run backtest."""
    logger.info("Starting backtest...")
    
    try:
        backtester = BacktesterComplete(settings)
        
        instruments = universe.split(",") if universe else settings.instruments
        timeframes = tfs.split(",") if tfs else ["M5", "M15", "H1"]
        
        logger.info(f"Instruments: {instruments}")
        logger.info(f"Timeframes:  {timeframes}")
        logger.info(f"Years: {years}")
        
        run_id = backtester.run(
            instruments=instruments,
            timeframes=timeframes,
        )
        
        logger.info(f"Backtest complete!  Run ID: {run_id}")
        logger.info(f"Results:  artifacts/runs/{run_id}/")
        
    except Exception as e:
        logger.exception(f"Backtest failed:  {e}")
        raise typer.Exit(code=1)


@app.command()
def report(
    run_id: str = typer.Option(... , help="Run ID to generate report for"),
):
    """Generate report for a backtest run."""
    logger.info(f"Generating report for run_id={run_id}")
    
    try:
        reporter = ReportGenerator(settings)
        reporter.generate_report(run_id)
        
        report_path = Path(settings.runs_dir) / run_id / "report.md"
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.exception(f"Report generation failed: {e}")
        raise typer. Exit(code=1)


@app.command()
def download_data(
    years: int = typer.Option(5, help="Number of years to download"),
    universe: Optional[str] = typer.Option(
        None, help="Instruments (comma-separated)"
    ),
    tfs: Optional[str] = typer. Option(
        "M5,M15,H1,H4,D", help="Timeframes (comma-separated)"
    ),
):
    """Download historical data from OANDA."""
    logger. info("Starting data download...")
    
    try:
        downloader = DataDownloader(settings)
        
        instruments = universe.split(",") if universe else settings.instruments
        timeframes = tfs. split(",") if tfs else ["M5", "M15", "H1", "H4", "D"]
        
        logger. info(f"Instruments: {instruments}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Years: {years}")
        
        for instrument in instruments:
            for tf in timeframes:
                logger.info(f"Downloading {instrument} {tf}...")
                downloader.download(instrument, tf, years)
        
        logger. info("Download complete!")
        
    except Exception as e:
        logger.exception(f"Download failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def cost_sensitivity(
    universe: Optional[str] = typer.Option(None, help="Instruments (comma-separated)"),
    tfs: Optional[str] = typer.Option("M15", help="Timeframes (comma-separated)"),
    start_date: Optional[str] = typer.Option(None, help="Start date ISO"),
    end_date: Optional[str] = typer.Option(None, help="End date ISO"),
):
    """Run cost sensitivity sweep at 1.0x / 1.25x / 1.50x costs."""
    instruments = universe.split(",") if universe else settings.instruments
    timeframes = tfs.split(",") if tfs else ["M15"]

    logger.info(
        f"Running cost sensitivity sweep for {instruments} {timeframes} | "
        f"{start_date or 'default'} → {end_date or 'default'}"
    )
    try:
        results = run_cost_sensitivity(
            instruments=instruments,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
        )
        for row in results:
            logger.info(
                f"Multiplier {row['cost_multiplier']:.2f} | "
                f"run_id={row['run_id']} | metrics={row['metrics_file']}"
            )
    except Exception as e:
        logger.exception(f"Cost sensitivity sweep failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def walkforward(
    train_years: int = typer.Option(
        4, help="Training period in years"
    ),
    test_months: int = typer.Option(
        6, help="Test period in months"
    ),
    step_months: int = typer.Option(
        3, help="Step forward in months"
    ),
):
    """Run walk-forward validation."""
    logger.info("Starting walk-forward validation...")
    
    try:
        backtester = BacktesterComplete(settings)

        today = pd.Timestamp(datetime.utcnow().date())
        dataset_end = pd.Timestamp(settings.end_date) if settings.end_date else today
        dataset_start = pd.Timestamp(settings.start_date) if settings.start_date else dataset_end - pd.DateOffset(years=settings.backtest_years)

        logger.info(
            f"Train: {train_years}y, Test: {test_months}m, Step: {step_months}m, "
            f"Purge: {settings.walkforward_purge_days}d, Embargo: {settings.walkforward_embargo_days}d"
        )

        windows = generate_walkforward_windows(
            start_date=dataset_start,
            end_date=dataset_end,
            train_years=train_years,
            test_months=test_months,
            step_months=step_months,
            purge_days=settings.walkforward_purge_days,
            embargo_days=settings.walkforward_embargo_days,
        )

        if not windows:
            logger.warning("No walk-forward windows generated for the given date range.")
            return

        for idx, window in enumerate(windows, start=1):
            logger.info(
                f"Iteration {idx}/{len(windows)} | "
                f"Train {window.train_start.date()} → {window.train_end.date()} | "
                f"Purge to {window.purge_end.date()} | Embargo to {window.embargo_end.date()} | "
                f"Test {window.test_start.date()} → {window.test_end.date()}"
            )
            run_id = backtester.run(
                instruments=settings.instruments,
                timeframes=["M5", "M15", "H1"],
                start_date=window.test_start.date().isoformat(),
                end_date=window.test_end.date().isoformat(),
            )
            logger.info(f"  Iteration {idx} complete:  {run_id}")

        logger.info("Walk-forward complete!")
        
    except Exception as e:
        logger.exception(f"Walk-forward failed: {e}")
        raise typer. Exit(code=1)


if __name__ == "__main__": 
    app()
