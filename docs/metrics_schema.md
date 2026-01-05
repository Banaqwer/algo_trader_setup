# Metrics & Diagnostics Schema

> Canonical definitions for all run artifacts produced during Phase 1–2.  
> All counters map directly to code paths in `backtester_complete.py`, `execution_engine_complete.py`, and `reporting_complete.py`.

## 1) Core Backtest Metrics (`metrics.json`)

| Field | Dimensions | Description | Source |
|-------|------------|-------------|--------|
| `total_trades` | scalar | Count of closed trades | `_compute_metrics` |
| `wins` / `losses` | scalar | Closed trades with positive / negative PnL | `_compute_metrics` |
| `win_rate` | scalar | `wins / total_trades` | `_compute_metrics` |
| `total_pnl_usd` | scalar | Sum of `pnl_usd` across closed trades | `_compute_metrics` |
| `avg_pnl_per_trade` | scalar | `total_pnl_usd / total_trades` | `_compute_metrics` |
| `expectancy_r` | scalar | Mean of `pnl_atr` | `_compute_metrics` |
| `max_drawdown` | scalar | Peak-to-trough drawdown (fraction) | `AccountState.max_drawdown` |
| `final_balance` | scalar | Broker balance after last trade | `SimulatedBroker.balance` |

## 2) Signal Funnel Diagnostics (`diagnostics.json`)

| Field | Dimensions | When it increments | Example |
|-------|------------|--------------------|---------|
| `pattern_signals_emitted[pattern_id].total` | pattern_id | Each `pattern.recognize` returns a signal | `VOL_COMPRESSION=42` |
| `pattern_signals_emitted[pattern_id].timeframes[tf]` | pattern_id, timeframe | Same as above, bucketed by execution timeframe | `FAILED_BREAKOUT.M15=18` |
| `signals_after_context_filters` | scalar | Non-empty signal set after context build per instrument/tf | `27` |
| `signals_after_npes` | scalar | Aggregator passed NPES threshold and direction ≠ 0 | `19` |
| `blocked_by_confidence_gate` | scalar | `AdaptiveTradeLimiter.should_take_trade` rejects for confidence | `4` |
| `blocked_by_model_confirmation` | scalar | Aggregator/confirmation stage rejects (no model gating yet) | `3` |
| `blocked_by_policy[reason]` | reason | Adaptive policy blocks (e.g., `Daily limit`, `Confidence`) | `{"Daily limit 10 reached": 2}` |
| `blocked_by_risk[reason]` | reason | `RiskManager.validate_trade` rejects | `{"Max positions per symbol (1) reached for EUR_USD": 6}` |
| `executed_trades` | scalar | A trade is placed successfully | `12` |
| `trades_per_day[date]` | date | Count of executed trades per calendar day | `{"2025-12-05": 3}` |
| `daily_funnel[date]` | date | Per-day survival counts + block breakdowns | `{pattern_signals_emitted: 5, blocked_by_policy: {"Daily limit 10 reached":1}}` |
| `days_with_no_trades[]` | list of snapshots | A day ended with 0 executions (WHY_NO_TRADE_TODAY) | `{"date":"2025-12-07","signals_after_npes":2,"blocked_by_risk":{"Max open positions (3) reached":2}}` |
| `patterns_disabled` | list | Patterns disabled via config flags (diagnostics clarity) | `["SESSION_BIAS","CORRELATION_DIVERGENCE"]` |
| `pattern_fire_rates[pattern_id]` | pattern_id | Checked vs fired counts and % | `{"checked":120,"fired":6,"fire_rate_pct":5.0}` |

## 3) Run Manifest (`run_manifest.json`)

| Field | Dimensions | Description | Example |
|-------|------------|-------------|---------|
| `run_id` | scalar | UUID fragment for the run | `"9ac3e7b2"` |
| `timestamp_utc` | scalar | Manifest creation time | `"2026-01-05T12:34:56.789Z"` |
| `git_commit` | scalar | `git rev-parse HEAD` at runtime | `"abcd1234..."` |
| `settings_snapshot` | object | Redacted config snapshot (`settings.dict()`) | `{adaptive_trading: true, cost_multiplier: 1.25, ...}` |
| `cost_assumptions` | object | Spread/slippage multipliers and toggles | `{base_spread_pips:1.5, slippage_bps:1.5, cost_multiplier:1.0}` |
| `walkforward_settings` | object | Train/test/step/purge/embargo values | `{train_years:4, test_months:6, purge_days:5, embargo_days:2}` |
| `data_coverage[instrument_tf]` | instrument/timeframe | Rows loaded and start/end timestamps | `{"EUR_USD_M15":{"rows":12345,"start":"2025-01-01","end":"2025-03-31"}}` |

## 4) Cost Sensitivity Sweep (`cost_sensitivity_*.json`)

| Field | Dimensions | Description | Example |
|-------|------------|-------------|---------|
| `cost_multiplier` | scalar | Spread/slippage multiplier applied | `1.50` |
| `run_id` | scalar | Run id for that multiplier | `"e12c4b8f"` |
| `metrics_file` | scalar | Path to metrics for that sweep leg | `"artifacts/runs/e12c4b8f/metrics.json"` |

## 5) Report (`report.md`)

- Embeds signal funnel summary (including top risk/policy blocks) and WHY_NO_TRADE_TODAY snapshots.
- Includes per-timeframe and per-pattern breakdowns from `metrics_full.json`.
