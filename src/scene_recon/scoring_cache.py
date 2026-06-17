from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scene_recon.candidates import init_candidates
from scene_recon.paths import scored_candidates_path, scoring_manifest_path
from scene_recon.poses import load_poses
from scene_recon.record import Record
from scene_recon.schema import SELECTION_COLUMNS
from scene_recon.video import score_all_frames


def _video_fingerprint(record: Record) -> dict:
    stat = record.video.stat()
    return {
        "video": str(record.video),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "poses_path": str(record.poses_path),
    }


def scoring_is_current(slug_dir_path: Path, record: Record) -> bool:
    manifest_path = scoring_manifest_path(slug_dir_path)
    scored_path = scored_candidates_path(slug_dir_path)
    if not manifest_path.is_file() or not scored_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("fingerprint") == _video_fingerprint(record)


def load_scored_candidates(slug_dir_path: Path) -> pd.DataFrame:
    scored_path = scored_candidates_path(slug_dir_path)
    if not scored_path.is_file():
        raise FileNotFoundError(f"missing scored cache: {scored_path}")
    df = pd.read_csv(scored_path, index_col="FrameNumber")
    for col in SELECTION_COLUMNS:
        if col == "selected":
            df[col] = False
        else:
            df[col] = pd.NA
    return df


def save_scored_candidates(slug_dir_path: Path, candidates: pd.DataFrame, record: Record) -> None:
    slug_dir_path.mkdir(parents=True, exist_ok=True)
    score_cols = [c for c in candidates.columns if c not in SELECTION_COLUMNS]
    candidates[score_cols].to_csv(scored_candidates_path(slug_dir_path), index=True)
    manifest = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(candidates),
        "fingerprint": _video_fingerprint(record),
    }
    scoring_manifest_path(slug_dir_path).write_text(json.dumps(manifest, indent=2) + "\n")


def score_record(record: Record, slug_dir_path: Path) -> pd.DataFrame:
    poses = load_poses(record)
    candidates = init_candidates(poses)
    candidates = score_all_frames(record, candidates)
    save_scored_candidates(slug_dir_path, candidates, record)
    return candidates


def load_or_score_record(
    record: Record,
    slug_dir_path: Path,
    *,
    rescore: bool = False,
) -> pd.DataFrame:
    if rescore or not scoring_is_current(slug_dir_path, record):
        return score_record(record, slug_dir_path)
    return load_scored_candidates(slug_dir_path)
