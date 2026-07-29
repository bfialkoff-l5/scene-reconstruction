from __future__ import annotations

from pathlib import Path

import pandas as pd

from scene_recon.record import Record
from scene_recon.scoring_cache import save_scored_candidates, scoring_is_current


def _record(tmp_path: Path) -> Record:
    record_path = tmp_path / "raw" / "flight"
    poses_dir = record_path / "_derived_v2"
    poses_dir.mkdir(parents=True)
    video = record_path / "AvatarS0093.mp4"
    poses = poses_dir / "gt_AvatarS0093.csv"
    intrinsics = record_path / "intrinsicK.csv"
    video.write_bytes(b"video")
    poses.write_text("FrameNumber,easting\n0,1\n")
    intrinsics.write_text("camera")
    return Record(
        path=record_path,
        slug="flight",
        video=video,
        poses_path=poses,
        intrinsics=intrinsics,
        stream_id="AvatarS0093",
        pose_source="_derived_v2",
    )


def test_pose_content_change_invalidates_scoring_cache(tmp_path: Path) -> None:
    record = _record(tmp_path)
    cache_dir = tmp_path / "odm-results" / record.slug
    candidates = pd.DataFrame(
        {
            "easting": [1.0],
            "feature_count": [100],
            "sharpness": [2.0],
            "quality_score": [0.5],
            "cell_x": [pd.NA],
            "cell_y": [pd.NA],
            "selected": [False],
            "reject_reason": [pd.NA],
        },
        index=pd.Index([0], name="FrameNumber"),
    )

    save_scored_candidates(cache_dir, candidates, record)
    assert scoring_is_current(cache_dir, record)

    record.poses_path.write_text("FrameNumber,easting\n0,2\n")
    assert not scoring_is_current(cache_dir, record)
