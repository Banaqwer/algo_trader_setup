# Change Plan (Phase 1–2 Completion)

**Status:** Completed instrumentation/validation scope — no strategy changes  
**Last Updated:** 2026-01-05

---

## Ordered File-by-File Changes (Phase 2)

| File | What it does | Phase-2 Fixes |
|------|--------------|---------------|
| `settings.py` | Central config | Added cost multiplier + base spread, pattern enable flags, purge/embargo fields, expanded `dict()` for manifests |
| `backtester_complete.py` | Event-driven backtester | Full funnel counters, per-day WHY_NO_TRADE snapshots, manifest writer, cost-aware context, pattern disable diagnostics |
| `execution_engine_complete.py` | Broker realism | Exit spread/slippage now respect cost multiplier/base spread |
| `cli.py` | Entry points | Added `cost-sensitivity`, walk-forward now uses purged/embargoed windows |
| `walkforward.py` | Split generator | New purged + embargoed window builder |
| `validation.py` | Harnesses | Cost sensitivity sweep (1.0x/1.25x/1.50x) |
| `reporting_complete.py` | Reporting | Embedded funnel summary, top block reasons, WHY_NO_TRADE_TODAY |
| `scripts/smoke_test.py` | Smoke harness | Asserts `run_manifest.json` is produced |
| `tests/test_walkforward.py` | Tests | Ensures purge/embargo splits, no overlap |
| `tests/test_manifest.py` | Tests | Verifies manifest presence + required fields |
| `docs/audit_report.md` | Audit artifact | Updated architecture + failure modes |
| `docs/metrics_schema.md` | Reference | Canonical definitions for metrics/diagnostics/manifests |
| `docs/next_iteration_loop.md` | Ops playbook | Monitoring, rollback, reproducibility, validation steps |

---

## What Remains (post-Phase 2, without touching strategy)
- **Enable context-dependent patterns:** Wire session stats + correlated returns so SessionBias/CorrelationDivergence can be re-enabled safely.
- **HTF data completeness:** Ensure D/H4 availability to avoid HTF range gaps.
- **Economic realism tests:** Automate cost sensitivity runs in CI once data volume concerns are addressed.
- **Data coverage checks:** Add automated coverage completeness assertions before backtests start.

---

## How to Run Phase-2 Validations
- Unit tests: `python -m pytest tests -v`
- Smoke test (creates manifest): `python -m cli smoke-test`
- Cost sweep: `python -m cli cost-sensitivity --universe EUR_USD --tfs M15`
- Walk-forward (purged/embargoed): `python -m cli walkforward --train-years 4 --test-months 6 --step-months 3`
