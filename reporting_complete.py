import logging
import json
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from settings import settings

logger = logging.getLogger(__name__)

class MetricsComputer:
    """Compute backtest metrics from trade journal."""
    
    @staticmethod
    def compute_all(trades_df: pd.DataFrame, initial_capital: float = 5000.0) -> Dict:
        """Compute comprehensive metrics."""
        
        if trades_df.empty:
            return MetricsComputer._empty_metrics()
        
        if isinstance(trades_df["entry_time"].iloc[0], str):
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        
        trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
        
        total_trades = len(trades_df)
        wins = (trades_df["pnl_usd"] > 0).sum()
        losses = (trades_df["pnl_usd"] < 0).sum()
        breakeven = (trades_df["pnl_usd"] == 0).sum()
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df["pnl_usd"].sum()
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        gross_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum()
        gross_loss = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        expectancy_r = trades_df["pnl_atr"].mean()
        expectancy_usd = avg_pnl_per_trade
        
        equity_curve = initial_capital + trades_df["pnl_usd"].cumsum()
        max_equity = equity_curve.max()
        final_equity = equity_curve.iloc[-1]
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        trading_period_years = (trades_df["exit_time"]. max() - trades_df["entry_time"].min()).days / 365.25
        if trading_period_years > 0:
            cagr = (final_equity / initial_capital) ** (1 / trading_period_years) - 1
        else:
            cagr = 0
        
        if len(trades_df) > 1:
            returns = np.diff(equity_curve) / initial_capital
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        if len(trades_df) > 1:
            returns = np.diff(equity_curve) / initial_capital
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino = 0
        else:
            sortino = 0
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        avg_mae_atr = trades_df["mae_atr"].mean()
        avg_mfe_atr = trades_df["mfe_atr"].mean()
        avg_mae_pct = trades_df["mae_pct"].mean()
        avg_mfe_pct = trades_df["mfe_pct"]. mean()
        
        per_tf = MetricsComputer._breakdown_by_timeframe(trades_df)
        per_pattern = MetricsComputer._breakdown_by_pattern(trades_df)
        
        return {
            "backtest_summary": {
                "total_trades": int(total_trades),
                "wins": int(wins),
                "losses": int(losses),
                "breakeven": int(breakeven),
                "win_rate": float(win_rate),
                "total_pnl_usd": float(total_pnl),
                "avg_pnl_per_trade":  float(avg_pnl_per_trade),
                "profit_factor": float(profit_factor),
                "initial_capital": float(initial_capital),
                "final_balance": float(final_equity),
            },
            "risk_metrics": {
                "max_drawdown": float(max_drawdown),
                "max_drawdown_pct": float(max_drawdown * 100),
                "avg_mae_atr": float(avg_mae_atr),
                "avg_mfe_atr": float(avg_mfe_atr),
                "avg_mae_pct": float(avg_mae_pct),
                "avg_mfe_pct": float(avg_mfe_pct),
            },
            "performance_metrics": {
                "cagr": float(cagr),
                "cagr_pct": float(cagr * 100),
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino),
                "calmar_ratio": float(calmar),
                "expectancy_r": float(expectancy_r),
                "expectancy_usd": float(expectancy_usd),
            },
            "per_timeframe": per_tf,
            "per_pattern": per_pattern,
        }
    
    @staticmethod
    def _breakdown_by_timeframe(trades_df: pd.DataFrame) -> Dict: 
        """Breakdown metrics by timeframe."""
        
        result = {}
        for tf in trades_df["entry_timeframe"].unique():
            tf_trades = trades_df[trades_df["entry_timeframe"] == tf]
            
            wins = (tf_trades["pnl_usd"] > 0).sum()
            total = len(tf_trades)
            
            result[str(tf)] = {
                "trades": int(total),
                "wins": int(wins),
                "win_rate": float(wins / total) if total > 0 else 0,
                "total_pnl":  float(tf_trades["pnl_usd"].sum()),
                "avg_pnl":  float(tf_trades["pnl_usd"].mean()),
                "expectancy_r": float(tf_trades["pnl_atr"].mean()),
            }
        
        return result
    
    @staticmethod
    def _breakdown_by_pattern(trades_df: pd.DataFrame) -> Dict:
        """Breakdown metrics by pattern."""
        
        result = {}
        all_patterns = set()
        for patterns_str in trades_df["patterns_triggered"]:
            if isinstance(patterns_str, str):
                patterns = [p.strip() for p in patterns_str.split(",")]
                all_patterns.update(patterns)
        
        for pattern in all_patterns:
            pattern_trades = trades_df[trades_df["patterns_triggered"]. str.contains(pattern, na=False)]
            
            if len(pattern_trades) > 0:
                wins = (pattern_trades["pnl_usd"] > 0).sum()
                total = len(pattern_trades)
                
                result[str(pattern)] = {
                    "trades": int(total),
                    "wins": int(wins),
                    "win_rate": float(wins / total),
                    "total_pnl": float(pattern_trades["pnl_usd"].sum()),
                    "expectancy_r": float(pattern_trades["pnl_atr"].mean()),
                }
        
        return result
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics dict."""
        return {
            "backtest_summary": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "profit_factor": 0.0,
            },
            "risk_metrics":  {
                "max_drawdown": 0.0,
            },
            "performance_metrics":  {
                "cagr":  0.0,
                "sharpe_ratio": 0.0,
            },
        }

class ReportGenerator:
    """Generate comprehensive backtest reports."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def generate_report(self, run_id: str) -> None:
        """Generate full report from backtest run."""
        
        run_dir = Path(self.settings. runs_dir) / run_id
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        logger.info(f"Generating report for run_id={run_id}")
        
        trades_file = run_dir / "trades. parquet"
        if trades_file.exists():
            trades_df = pd.read_parquet(trades_file)
        else:
            trades_df = None
        
        if trades_df is not None:
            metrics = MetricsComputer.compute_all(trades_df, initial_capital=5000.0)
            
            with open(run_dir / "metrics_full.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            self._generate_markdown_report(run_dir, metrics, trades_df)
            self._generate_charts(run_dir, trades_df)
        
        logger.info(f"Report complete:  {run_dir}")
    
    def _generate_markdown_report(self, run_dir: Path, metrics: Dict, trades_df: pd.DataFrame) -> None:
        """Generate markdown summary."""
        
        md = f"""# Backtest Report

**Run ID:** {run_dir.name}
**Generated:** {datetime.utcnow().isoformat()}

---

## Performance Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics["backtest_summary"]["total_trades"]} |
| Win Rate | {metrics["backtest_summary"]["win_rate"]:.1%} |
| Total P&L | ${metrics["backtest_summary"]["total_pnl_usd"]:+.2f} |
| Final Balance | ${metrics["backtest_summary"]["final_balance"]:+.2f} |
| CAGR | {metrics["performance_metrics"]["cagr"]:. 2%} |
| Sharpe Ratio | {metrics["performance_metrics"]["sharpe_ratio"]:.2f} |
| Max Drawdown | {metrics["risk_metrics"]["max_drawdown"]:.2%} |
| Profit Factor | {metrics["backtest_summary"]["profit_factor"]:.2f} |
| Expectancy (R) | {metrics["performance_metrics"]["expectancy_r"]: +.2f} |

### Risk Metrics

- **Max Adverse Excursion (Avg):** {metrics["risk_metrics"]["avg_mae_atr"]:.2f} ATR ({metrics["risk_metrics"]["avg_mae_pct"]:.1f}%)
- **Max Favorable Excursion (Avg):** {metrics["risk_metrics"]["avg_mfe_atr"]:.2f} ATR ({metrics["risk_metrics"]["avg_mfe_pct"]:.1f}%)

---

## Performance by Timeframe

"""
        
        for tf, stats in metrics.get("per_timeframe", {}).items():
            md += f"""
### {tf}

- Trades: {stats["trades"]}
- Win Rate: {stats["win_rate"]:.1%}
- Total P&L: ${stats["total_pnl"]:+.2f}
- Expectancy (R): {stats["expectancy_r"]:+.2f}

"""
        
        md += f"""
---

## Pattern Performance

"""
        
        for pattern, stats in metrics.get("per_pattern", {}).items():
            md += f"""
### {pattern}

- Trades: {stats["trades"]}
- Win Rate: {stats["win_rate"]:.1%}
- Total P&L: ${stats["total_pnl"]:+. 2f}
- Expectancy (R): {stats["expectancy_r"]:+.2f}

"""
        
        md += f"""
---

## Trade Summary

Total trades: {len(trades_df)}

### Recent Trades (Last 20)

"""
        
        if len(trades_df) > 0:
            recent = trades_df.tail(20)[["instrument", "entry_timeframe", "direction", "pnl_usd", "pnl_atr", "exit_type"]]
            md += recent.to_markdown(index=False)
        
        report_file = run_dir / "report. md"
        with open(report_file, "w") as f:
            f.write(md)
        
        logger.info(f"Markdown report:  {report_file}")
    
    def _generate_charts(self, run_dir: Path, trades_df: pd.DataFrame) -> None:
        """Generate backtest charts."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Backtest Results:  {run_dir.name}")
        
        if len(trades_df) > 0:
            trades_sorted = trades_df.sort_values("exit_time")
            equity = 5000 + trades_sorted["pnl_usd"].cumsum()
            
            axes[0, 0].plot(equity.values, linewidth=2, color="blue")
            axes[0, 0].set_title("Equity Curve")
            axes[0, 0].set_ylabel("Balance ($)")
            axes[0, 0].grid(True, alpha=0.3)
        
        if len(trades_df) > 0:
            trades_sorted = trades_df.sort_values("exit_time")
            equity = 5000 + trades_sorted["pnl_usd"].cumsum()
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            
            axes[0, 1].fill_between(range(len(drawdown)), drawdown. values, 0, alpha=0.3, color="red")
            axes[0, 1]. plot(drawdown.values, linewidth=1, color="red")
            axes[0, 1].set_title("Drawdown %")
            axes[0, 1].set_ylabel("Drawdown (%)")
            axes[0, 1].grid(True, alpha=0.3)
        
        if len(trades_df) > 0:
            axes[1, 0].hist(trades_df["pnl_usd"], bins=30, edgecolor="black", alpha=0.7)
            axes[1, 0].axvline(trades_df["pnl_usd"].mean(), color="red", linestyle="--", label=f"Mean: {trades_df['pnl_usd'].mean():.2f}")
            axes[1, 0].set_title("P&L Distribution")
            axes[1, 0].set_xlabel("P&L ($)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if len(trades_df) > 0:
            tf_counts = trades_df["entry_timeframe"].value_counts()
            axes[1, 1].bar(tf_counts.index, tf_counts.values, color="steelblue", edgecolor="black")
            axes[1, 1].set_title("Trades by Timeframe")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_file = run_dir / "charts.png"
        fig.savefig(chart_file, dpi=100)
        plt.close(fig)
        
        logger. info(f"Charts:  {chart_file}")