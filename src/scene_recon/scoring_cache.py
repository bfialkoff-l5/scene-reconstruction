from __future__ import annotations

import json
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scene_recon.candidates import init_candidates
from scene_recon.paths import scored_candidates_path, scoring_manifest_path
from scene_recon.poses import load_poses
from scene_recon.record import Record
from scene_recon.schema import SELECTION_COLUMNS
from scene_recon.video import score_all_frames


def file_fingerprint(path: Path) -> dict:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def record_fingerprint(record: Record) -> dict:
    video_stat = record.video.stat()
    return {
        "video": str(record.video),
        "video_size": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "pose_source": record.pose_source,
        "poses": file_fingerprint(record.poses_path),
    }


def scoring_is_current(slug_dir_path: Path, record: Record) -> bool:
    manifest_path = scoring_manifest_path(slug_dir_path, record.cache_key)
    scored_path = scored_candidates_path(slug_dir_path, record.cache_key)
    if not manifest_path.is_file() or not scored_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("fingerprint") == record_fingerprint(record)


def load_scored_candidates(slug_dir_path: Path, record: Record) -> pd.DataFrame:
    scored_path = scored_candidates_path(slug_dir_path, record.cache_key)
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
    candidates[score_cols].to_csv(
        scored_candidates_path(slug_dir_path, record.cache_key), index=True
    )
    manifest = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(candidates),
        "fingerprint": record_fingerprint(record),
    }
    scoring_manifest_path(slug_dir_path, record.cache_key).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


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
    return load_scored_candidates(slug_dir_path, record)
