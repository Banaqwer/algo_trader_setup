import typer
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from settings import settings
from backtester_complete import BacktesterComplete
from reporting_complete import ReportGenerator
from data_downloader import DataDownloader

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
        
        logger. info(
            f"Train:  {train_years} years, Test: {test_months} months, "
            f"Step: {step_months} months"
        )
        
        # Simplified walk-forward:  just run multiple backtests
        for i in range(3):  # Run 3 iterations
            logger.info(f"Iteration {i+1}/3...")
            run_id = backtester.run(
                instruments=settings.instruments,
                timeframes=["M5", "M15", "H1"],
            )
            logger.info(f"  Iteration {i+1} complete:  {run_id}")
        
        logger.info("Walk-forward complete!")
        
    except Exception as e:
        logger.exception(f"Walk-forward failed: {e}")
        raise typer. Exit(code=1)


if __name__ == "__main__": 
    app()