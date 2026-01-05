# Validation Results

**Date:** 2026-01-05
**Status:** Phase 4 Complete - System Functional

---

## Test Commands

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run 30-day smoke test
python -c "from scripts.smoke_test import run_smoke_test; run_smoke_test()"

# Run backtest via CLI
python -m cli backtest --years 1 --universe EUR_USD --tfs M15
```

---

## Baseline vs Improved Results

### Before Fixes
| Metric | Value |
|--------|-------|
| Trades (2-month) | 0 |
| P&L | $0.00 |
| System Status | **BROKEN** - 100% trades blocked |
| Root Cause | Position sizing → Leverage exceeded 5x cap |

### After Fixes (2-month: Nov-Dec 2025)
| Metric | Value |
|--------|-------|
| Trades | 157 |
| Win Rate | 47.1% |
| P&L | -$585.97 |
| Final Balance | $4,414.03 |
| System Status | **FUNCTIONAL** |

### 6-Month Validation (Jun-Dec 2025, EUR_USD + GBP_USD)
| Metric | Value |
|--------|-------|
| Trades | 841 |
| Win Rate | 39.4% |
| Trades/Month | ~140 |
| P&L | -$3,374.51 |
| Final Balance | $1,625.49 |
| Profit Factor | 0.57 |

---

## Unit Test Results

```
21 passed, 1 warning

Tests:
- test_position_size_respects_leverage_cap ✓
- test_position_size_non_zero_for_valid_input ✓
- test_position_size_zero_for_zero_stop ✓
- test_validate_trade_passes_with_leverage_cap ✓
- test_validate_trade_rejects_overleveraged ✓
- test_volatility_compression_direction_bounds ✓
- test_liquidity_sweep_pattern_id ✓
- test_failed_breakout_pattern_id ✓
- test_aggregation_empty_signals ✓
- test_aggregation_npes_bounds ✓
- test_backtester_initialization ✓
- test_backtester_context_building ✓
- test_initial_state ✓
- test_drawdown_tracking ✓
- test_position_tracking ✓
+ 6 pre-existing tests
```

---

## Signal Funnel Analysis

| Stage | Count | Survival % |
|-------|-------|------------|
| Pattern signals | 532 | 100% |
| After NPES | 440 | 83% |
| Blocked by risk | 264 | - |
| Executed | 158 | 30% |

### Risk Block Breakdown
- Max positions per symbol: 256 (97%)
- Daily loss limit: 8 (3%)

---

## Security Scan Results

CodeQL Analysis: **0 alerts** (Python)

---

## Known Limitations

1. **Negative Expectancy:** Current patterns have profit factor ~0.57 (lose money)
2. **Non-Firing Patterns:** SessionBias, CorrelationDivergence, HTFRange never fire
3. **ModelManager:** Returns uniform probabilities (placeholder)
4. **Walk-Forward:** Simplified (no purge/embargo)

---

## Recommendations for Future Work

1. **Pattern Edge Investigation:**
   - Analyze why patterns have negative expectancy
   - Consider different exit strategies (tighter SL, wider TP)
   - Investigate pattern conditions for higher-quality signals

2. **Enable More Patterns:**
   - Populate session_stats for SessionBiasPattern
   - Add correlated pair returns to context
   - Ensure HTF data is always available

3. **Reduce Position Blocking:**
   - Consider allowing 2 positions per symbol
   - Or implement pyramiding logic

4. **Cost Sensitivity:**
   - Test with +25% and +50% spread to assess robustness
