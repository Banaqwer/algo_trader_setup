# Algo Trader Repository Audit Report

**Date:** 2026-01-05
**Auditor:** Jack — Quant Architect & Systems Debugger
**Status:** Phase 1 Complete

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI (cli.py)                                   │
│  Commands: print_config, smoke_test, backtest, download_data, report    │
│            walkforward                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BacktesterComplete (backtester_complete.py)           │
│  Event-driven bar-by-bar backtest loop                                   │
│  - Loads data via DataDownloader                                         │
│  - Computes features via FeatureEngine                                   │
│  - Runs patterns across timeframes                                       │
│  - Aggregates signals via PatternAggregator                              │
│  - Validates via RiskManager + AdaptiveLimiter                          │
│  - Executes via SimulatedBroker                                          │
│  - Records via SelfLearningEngine                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐    ┌─────────────────────────┐    ┌───────────────────┐
│ FeatureEngine │    │    6 Pattern Modules     │    │  PatternAggregator │
│ (31 features) │    │  - VolCompression        │    │  (NPES scoring)    │
│ - ATR/EMA/RSI │    │  - LiquiditySweep        │    │  - Long/Short sum  │
│ - MACD/ADX/BB │    │  - SessionBias           │    │  - Normalized      │
│ - Regime/Wick │    │  - HTFRange              │    │  - Trade gating    │
└───────────────┘    │  - CorrDivergence        │    └───────────────────┘
                     │  - FailedBreakout        │
                     └─────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐    ┌─────────────────┐    ┌───────────────────────┐
│ AdaptiveLimiter   │    │  RiskManager     │    │  SimulatedBroker      │
│ - Conf threshold  │    │  - Leverage cap  │    │  - Spread/Slippage    │
│ - Daily limits    │    │  - Position lims │    │  - SL/TP tracking     │
│ - Win rate adjust │    │  - Daily/Weekly  │    │  - MAE/MFE recording  │
└───────────────────┘    └─────────────────┘    └───────────────────────┘
```

## 2. File-by-File Analysis

| File | Purpose | Issues Found |
|------|---------|--------------|
| `backtester_complete.py` | Main event loop | Position sizing doesn't cap by leverage |
| `pattern_aggregation.py` | NPES scoring | Works correctly, baseline threshold = 0.15 |
| `feature_engineering.py` | Technical indicators | NaN handling OK, 200 bar warmup |
| `risk_manager.py` | Risk validation | Leverage check correctly rejects over-leveraged trades |
| `adaptive_limiter.py` | Confidence gating | min_confidence = 0.55, works correctly |
| `trading_utils.py` | Spread/slippage | Correct implementation |
| `patterns/volatility_compression.py` | Pattern 1 | Fires ~0.6% of bars (good) |
| `patterns/liquidity_sweep.py` | Pattern 2 | Fires ~3% of bars (good) |
| `patterns/session_bias.py` | Pattern 3 | NEVER fires - requires pre-populated stats |
| `patterns/htf_range.py` | Pattern 4 | Context-dependent, fires when htf_range_position at extremes |
| `patterns/correlation_divergence.py` | Pattern 5 | NEVER fires - requires corr pair returns in context |
| `patterns/failed_breakout.py` | Pattern 6 | Fires ~12% of bars (highest) |
| `models/model_manager.py` | ML placeholder | Returns uniform probabilities (0.33, 0.34, 0.33) - NOT BLOCKING |

## 3. Top 10 Failure Modes (Ranked by Impact)

### 🔴 CRITICAL (Blocks ALL Trades)

**#1: Position Sizing Exceeds Leverage Cap**
- **Impact:** BLOCKING 100% of trades
- **Location:** `backtester_complete.py:546-556` (`_calculate_position_size`)
- **Evidence:** 
  ```python
  # Current code calculates units solely based on risk:
  units = int(risk_usd / stop_distance)
  # But doesn't cap by max leverage!
  ```
- **Test Results:** All 47 signals that passed aggregation were rejected with "Leverage (7.56-9.99) exceeds max (5.0)"
- **Root Cause:** With 0.5% risk and small ATR (~0.0005-0.001), calculated units create 7-10x leverage
- **Fix:** Cap units by `min(units_by_risk, max_units_at_leverage_cap)`

### 🟠 HIGH (Significantly Reduces Signal Count)

**#2: SessionBiasPattern Never Fires**
- **Impact:** 0 signals from this pattern (should be 5-10% of total)
- **Location:** `patterns/session_bias.py:30-31`
- **Evidence:** `session_stats: {}` - dictionary is never populated
- **Root Cause:** Pattern requires pre-populated historical session statistics
- **Fix:** Either populate stats from learning engine or remove dependency

**#3: CorrelationDivergencePattern Never Fires**
- **Impact:** 0 signals from this pattern
- **Location:** `patterns/correlation_divergence.py:33-35`
- **Evidence:** Context missing `{corr_pair}_returns` key
- **Root Cause:** Backtester doesn't pass correlated pair returns to context
- **Fix:** Build context with correlated pair returns from multi-timeframe data

**#4: HTFRangePattern Context Dependency**
- **Impact:** Fires only at range extremes (<15% or >85%)
- **Location:** `patterns/htf_range.py:18-20`, `backtester_complete.py:495-505`
- **Evidence:** `htf_range_position` only calculated when D1/H4 data available
- **Status:** Partially working - depends on HTF data availability

### 🟡 MEDIUM (Affects Performance)

**#5: ModelManager Returns Uniform Probabilities**
- **Impact:** No ML-based signal filtering (neutral impact currently)
- **Location:** `models/model_manager.py:36-38`
- **Evidence:** Always returns `(0.33, 0.34, 0.33)` - not used for gating
- **Status:** Not blocking trades, but doesn't add value

**#6: No Signal Pipeline Counters**
- **Impact:** Cannot diagnose where signals die
- **Location:** Throughout `backtester_complete.py`
- **Fix:** Add comprehensive counters per stage

**#7: Spread Calculation for JPY Pairs**
- **Impact:** JPY pip value may be wrong
- **Location:** `trading_utils.py:24`
- **Evidence:** Uses `spread_pips * 0.0001` for JPY (should be `0.001`?)
- **Status:** Needs verification

### 🟢 LOW (Minor Issues)

**#8: Walking Timestamp Comparison Bug in Adaptive Limiter**
- **Location:** `adaptive_limiter.py:105`
- **Evidence:** `t.exit_time > cutoff` compares Timestamp to datetime
- **Impact:** May cause type mismatch warnings

**#9: Missing Purge/Embargo in Walk-Forward**
- **Location:** `cli.py:154-160`
- **Evidence:** Walk-forward is simplified, just runs 3 sequential backtests
- **Impact:** Potential data leakage in walk-forward validation

**#10: No Cost Sensitivity Analysis**
- **Impact:** Cannot verify robustness to transaction cost changes
- **Status:** Feature gap, not a bug

## 4. Signal Pipeline Analysis (30-day EUR_USD M15)

| Stage | Count | Survival Rate |
|-------|-------|---------------|
| Pattern signals emitted | 60 | 100% |
| Passed aggregation (NPES > 0.15) | 47 | 78% |
| Passed confidence gate (≥0.55) | 47 | 100% |
| **Blocked by leverage** | 47 | **0%** |
| Executed trades | **0** | **0%** |

## 5. Recommended Fix Priority

1. **[CRITICAL]** Fix position sizing to cap by leverage (Phase 3)
2. **[HIGH]** Add diagnostic instrumentation (Phase 2)
3. **[MEDIUM]** Fix SessionBias and CorrelationDivergence patterns
4. **[MEDIUM]** Improve context building in backtester
5. **[LOW]** Add cost sensitivity testing

## 6. Verification Plan

After fixes:
1. Run `python -m pytest tests/ -v` - all tests should pass
2. Run 30-day smoke test - should produce >0 trades
3. Run 1-year backtest - should have reasonable trade count (100-500/year)
4. Compare metrics: win rate, expectancy R, max DD, Sharpe

---

**Next Steps:** Proceed to Phase 2 (Instrumentation) then Phase 3 (Critical Fixes)
