from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def stamp_run_ts(when: datetime | None = None) -> str:
    return (when or datetime.now(timezone.utc)).strftime("%Y%m%d%H%M%S")


def odm_results_dir(data_root: Path) -> Path:
    return data_root / "odm-results"


def slug_dir(data_root: Path, slug: str) -> Path:
    return odm_results_dir(data_root) / slug


def scored_candidates_path(slug_dir_path: Path) -> Path:
    return slug_dir_path / "candidates_scored.csv"


def scoring_manifest_path(slug_dir_path: Path) -> Path:
    return slug_dir_path / "scoring.json"


def runs_dir(slug_dir_path: Path) -> Path:
    return slug_dir_path / "runs"


def run_dir(slug_dir_path: Path, run_ts: str) -> Path:
    return runs_dir(slug_dir_path) / run_ts


def odm_dataset_name() -> str:
    return "odm_input"


def odm_project_path(run_dir_path: Path) -> Path:
    """ODM --project-path: parent of the dataset folder (contains odm_input/)."""
    return run_dir_path


def list_runs(slug_dir_path: Path) -> list[Path]:
    base = runs_dir(slug_dir_path)
    if not base.is_dir():
        return []
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 14]
    return sorted(runs, key=lambda p: p.name)


def latest_run(slug_dir_path: Path) -> Path:
    runs = list_runs(slug_dir_path)
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_dir(slug_dir_path)}")
    return runs[-1]


def resolve_run(slug_dir_path: Path, run_ts: str | None = None) -> Path:
    if run_ts:
        path = run_dir(slug_dir_path, run_ts)
        if not path.is_dir():
            raise FileNotFoundError(f"Run not found: {path}")
        return path
    return latest_run(slug_dir_path)
