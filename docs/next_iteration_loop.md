# Next Iteration Loop (Operational Playbook)

## Monitoring Plan (Phase 1–2 scope)
- **Artifacts to check per run:** `run_manifest.json`, `diagnostics.json`, `metrics.json`, `report.md`.
- **Signal funnel health:** Watch survival rates (`pattern_signals_total → executed_trades`) and top block reasons (risk/policy). Investigate if execution rate < 5% or a single risk reason > 70%.
- **Pattern activity:** Ensure no enabled pattern shows `fire_rate_pct = 0`. Disabled patterns must remain explicitly flagged via `patterns_disabled`.
- **Cost sensitivity:** Run `python -m cli cost-sensitivity --universe EUR_USD --tfs M15` weekly; compare metrics across 1.0x/1.25x/1.50x multipliers.
- **Walk-forward hygiene:** Validate `generate_walkforward_windows` outputs (train/test + purge/embargo) before any new strategy release.

## Rollback Triggers
- Missing or malformed `run_manifest.json` or `diagnostics.json` in any run.
- Signal funnel collapse: `signals_after_npes / pattern_signals_total < 5%` for two consecutive runs.
- Policy/risk saturation: any single block reason > 85% of denials.
- Unexpected pattern silence: enabled pattern shows `fire_rate_pct == 0` over a full week of data.
- Data coverage gaps: `data_coverage` shows < 90% of expected candles for any instrument/timeframe.

## Reproducibility Rules
- Every run must be tied to a git commit hash (stored in `run_manifest.json`).
- Preserve full config snapshots via `settings_snapshot`; never run with ad-hoc overrides without recording them.
- Keep raw parquet inputs immutable; reference coverage via `data_coverage` hashes.
- Record manual commands used (see validation section) when re-running smoke/backtests.

## Validation & Promotion Steps
1. **Smoke test:** `python -m cli smoke-test` (ensures manifest creation).
2. **Unit tests:** `python -m pytest tests -v` (includes purge/embargo + manifest coverage).
3. **Cost sensitivity sweep:** `python -m cli cost-sensitivity --universe EUR_USD --tfs M15`.
4. **Walk-forward dry run:** `python -m cli walkforward --train-years 4 --test-months 6 --step-months 3` (confirms purge/embargoed splits).
5. **Report review:** Inspect `report.md` for funnel summary and WHY_NO_TRADE_TODAY snapshots.

## Change Acceptance
- Only promote changes when all above checks are green and funnel survival is consistent with prior baselines.
- Document any temporary pattern disables in `patterns_disabled` and `docs/change_plan.md`.
