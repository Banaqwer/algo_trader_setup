import json
from pathlib import Path

import pandas as pd

from backtester_complete import BacktesterComplete
from settings import settings


def test_run_manifest_created(tmp_path):
    """run_manifest.json should include git hash, settings, coverage, and cost assumptions."""
    bt = BacktesterComplete.__new__(BacktesterComplete)
    bt.settings = settings.copy(deep=True)
    bt.settings.artifact_dir = str(tmp_path / "artifacts")
    # ensure runs_dir property points to temp artifact dir
    Path(bt.settings.runs_dir).mkdir(parents=True, exist_ok=True)

    run_dir = Path(bt.settings.runs_dir) / "testrun"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cache = {
        ("EUR_USD", "M15"): pd.DataFrame(
            {"time": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]}
        )
    }

    # call internal helper directly
    bt._save_run_manifest("testrun", run_dir, data_cache)

    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists(), "Manifest file not written"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["run_id"] == "testrun"
    assert "git_commit" in manifest
    assert "settings_snapshot" in manifest
    assert "cost_assumptions" in manifest
    assert "data_coverage" in manifest


def test_smoke_test_writes_manifest(monkeypatch, tmp_path):
    """Smoke test should fail if manifest missing; simulate run to confirm check."""
    from scripts import smoke_test as smoke_mod

    temp_settings = settings.copy(deep=True)
    temp_settings.artifact_dir = str(tmp_path / "artifacts")

    class DummyBacktester:
        def __init__(self, *_args, **_kwargs):
            self.settings = temp_settings

        def run(self, instruments, timeframes, start_date=None, end_date=None):
            run_id = "dummy1234"
            run_dir = Path(self.settings.runs_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_manifest.json").write_text("{}")
            return run_id

    monkeypatch.setattr(smoke_mod, "BacktesterComplete", DummyBacktester, raising=False)
    monkeypatch.setattr(smoke_mod, "settings", temp_settings, raising=False)

    assert smoke_mod.run_smoke_test() is True
