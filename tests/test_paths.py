from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from scene_recon.paths import (
    latest_run,
    list_runs,
    odm_results_dir,
    resolve_run,
    run_dir,
    runs_dir,
    scored_candidates_path,
    scoring_manifest_path,
    slug_dir,
    stamp_run_ts,
)


def test_slug_layout(tmp_path: Path) -> None:
    slug = "0088_test"
    slug_path = slug_dir(tmp_path, slug)
    assert slug_path == odm_results_dir(tmp_path) / slug
    assert scored_candidates_path(slug_path).name == "candidates_scored.csv"
    assert scoring_manifest_path(slug_path).name == "scoring.json"


def test_runs_resolution(tmp_path: Path) -> None:
    slug_path = slug_dir(tmp_path, "0088_test")
    ts = stamp_run_ts(datetime(2026, 6, 16, 14, 0, 0, tzinfo=timezone.utc))
    run_path = run_dir(slug_path, ts)
    run_path.mkdir(parents=True)

    assert list_runs(slug_path) == [run_path]
    assert latest_run(slug_path) == run_path
    assert resolve_run(slug_path) == run_path
    assert resolve_run(slug_path, ts) == run_path


def test_resolve_run_missing(tmp_path: Path) -> None:
    slug_path = slug_dir(tmp_path, "0088_test")
    with pytest.raises(FileNotFoundError):
        latest_run(slug_path)
