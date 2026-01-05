# Validation Guide (Phase 1–2)

**Date:** 2026-01-05  
**Status:** Instrumentation & diagnostics complete — strategy untouched

---

## Commands

```bash
# Run all unit tests (includes purge/embargo + manifest checks)
python -m pytest tests -v

# 90-day smoke test (fails if manifest missing)
python -m cli smoke-test

# Cost sensitivity sweep (1.0x / 1.25x / 1.50x spread+slippage)
python -m cli cost-sensitivity --universe EUR_USD --tfs M15

# Walk-forward with purge+embargo windows
python -m cli walkforward --train-years 4 --test-months 6 --step-months 3

# Generate report for a run
python -m cli report --run-id <RUN_ID>
```

---

## Expected Artifacts Per Run
- `artifacts/runs/<run_id>/run_manifest.json` (git hash, settings snapshot, cost assumptions, data coverage, walk-forward settings)
- `artifacts/runs/<run_id>/diagnostics.json` (full signal funnel + WHY_NO_TRADE_TODAY)
- `artifacts/runs/<run_id>/metrics.json` (core backtest metrics)
- `artifacts/runs/<run_id>/report.md` (funnel summary, top risk/policy blocks, per-pattern stats)

---

## Cost Sensitivity Sweep Output
- Command writes a summary file: `artifacts/runs/cost_sensitivity_<timestamp>.json`
- Each entry: `{cost_multiplier, run_id, metrics_file}`
- Use `metrics_file` paths to build comparison tables; expected trend is monotonic degradation as costs rise (diagnostic only, not profitability tuning).

---

## Walk-Forward Hygiene (Purged + Embargoed)
- Windows generated via `generate_walkforward_windows` (train_years, test_months, step_months, purge_days, embargo_days).
- Guarantee: `train_end < purge_end ≤ embargo_end ≤ test_start ≤ test_end` and non-overlapping test segments (see `tests/test_walkforward.py`).

---

## Current Limitations (Tracked)
1. SessionBias and CorrelationDivergence remain **disabled by config** pending context feeds.
2. HTF range still depends on D/H4 data availability.
3. Strategy profitability intentionally unchanged; diagnostics only.

Use `docs/metrics_schema.md` for metric definitions and `docs/next_iteration_loop.md` for monitoring/rollback rules.
