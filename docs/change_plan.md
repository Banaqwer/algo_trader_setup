# Change Plan for Algo Trader Improvements

**Status:** In Progress
**Last Updated:** 2026-01-05

---

## Summary of Issues Found

### Critical (Fixed)
1. **Position sizing exceeds leverage cap** - All trades blocked
2. **Price update loop using wrong timeframe** - Trades never close (use M5 when M15 requested)

### High Priority (Pending)
3. SessionBiasPattern never fires - needs session_stats population
4. CorrelationDivergencePattern never fires - needs correlated pair returns
5. HTFRangePattern underutilized - context often missing htf_range_position

### Medium Priority (Pending)
6. ModelManager returns uniform probabilities (placeholder)
7. Walk-forward is simplified (no purge/embargo)

---

## Changes Made

### Phase 1-2: Critical Fixes

#### 1. `backtester_complete.py` - Position Sizing Fix

**Location:** `_calculate_position_size()` method (lines 603-626)

**Before:**
```python
def _calculate_position_size(self, entry_price: float, sl_price: float) -> int:
    stop_distance = abs(entry_price - sl_price)
    if stop_distance == 0:
        return 0
    risk_usd = self.account.current_balance * 0.005
    units = int(risk_usd / stop_distance)
    return max(units, 1)
```

**After:**
```python
def _calculate_position_size(self, entry_price: float, sl_price: float) -> int:
    stop_distance = abs(entry_price - sl_price)
    if stop_distance == 0:
        return 0
    
    # Calculate units based on risk per trade
    risk_usd = self.account.current_balance * 0.005
    units_by_risk = int(risk_usd / stop_distance)
    
    # Cap units by maximum leverage
    max_leverage = self.risk_manager.max_leverage
    max_notional = self.account.current_balance * max_leverage
    max_units_by_leverage = int(max_notional / entry_price) if entry_price > 0 else 0
    
    # Use the smaller of the two
    units = min(units_by_risk, max_units_by_leverage)
    return max(units, 1)
```

**Verification:** Trades now execute (0 → 157 in 2-month test)

#### 2. `backtester_complete.py` - Price Update Loop Fix

**Location:** Event loop price update section (lines 475-482)

**Before:**
```python
key = (instrument, "M5")  # Hardcoded M5
```

**After:**
```python
key = (instrument, base_tf)  # Use selected base timeframe
```

**Verification:** Trades now close properly (SL/TP hit)

#### 3. `backtester_complete.py` - Diagnostic Instrumentation

**Added:**
- `_init_diagnostics()` method
- Per-pattern signal counters
- Signal funnel tracking
- Risk block reason tracking
- `_save_diagnostics()` method for JSON output

**Output:** `diagnostics.json` in run directory with:
- Signal funnel summary
- Per-pattern fire rates
- Risk block breakdown
- Trades per day

---

## Files Modified

| File | Changes |
|------|---------|
| `backtester_complete.py` | Position sizing fix, price update fix, diagnostics |
| `tests/test_backtester.py` | New - 15 unit tests for core functionality |
| `docs/audit_report.md` | New - Comprehensive audit documentation |
| `docs/change_plan.md` | New - This file |

---

## Validation Results

### Baseline (Before Fixes)
- Trades: 0
- P&L: $0.00
- Issue: All trades blocked by leverage check

### After Fixes (2-month test: Nov-Dec 2025)
- Trades: 157
- Win Rate: 47.1%
- P&L: -$585.97
- Final Balance: $4,414.03

### Signal Funnel (After Fixes)
- Pattern signals: 532
- Passed NPES: 440 (83%)
- Blocked by risk: 264 (mostly max positions per symbol)
- Executed: 158 (30% of signals)

---

## Remaining Work

### Next Steps (Priority Order)
1. **Reduce position limit blocks** - Consider increasing max_open_positions_per_symbol or allowing pyramiding
2. **Fix non-firing patterns** - SessionBias, CorrelationDivergence, HTFRange
3. **Improve pattern edge** - Current profit factor ~0.60 indicates negative expectancy
4. **Add cost sensitivity analysis** - Test with +25% and +50% spread

### Out of Scope (Future Work)
- ML model implementation (ModelManager)
- Walk-forward with purge/embargo
- Multi-timeframe correlation context
