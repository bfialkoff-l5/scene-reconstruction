from __future__ import annotations

from pathlib import Path

import pytest

from scene_recon.record import Record

RECORD_PATH = Path("/home/bfialkoff/s3/raw/0088_20260122_eitan_1")


def _write_record_layout(path: Path, poses_dir: str) -> None:
    (path / poses_dir).mkdir(parents=True)
    (path / "AvatarS0093.mp4").touch()
    (path / "intrinsicK.csv").touch()
    (path / poses_dir / "gt_AvatarS0093.csv").touch()


def test_record_requires_derived_v2(tmp_path: Path) -> None:
    record_path = tmp_path / "raw" / "flight"
    _write_record_layout(record_path, "_derived")

    with pytest.raises(FileNotFoundError, match="_derived_v2"):
        Record.from_path(record_path)


def test_record_uses_derived_v2(tmp_path: Path) -> None:
    record_path = tmp_path / "raw" / "flight"
    _write_record_layout(record_path, "_derived_v2")
    record = Record.from_path(record_path)

    assert record.pose_source == "_derived_v2"
    assert record.cache_key == "derived_v2"
    assert record.poses_path == record_path / "_derived_v2" / "gt_AvatarS0093.csv"


@pytest.mark.skipif(not RECORD_PATH.is_dir(), reason="example record not on disk")
def test_record_from_path_example() -> None:
    record = Record.from_path(RECORD_PATH)
    assert record.slug == "0088_20260122_eitan_1"
    assert record.stream_id == "AvatarS0093"
    assert record.video.name == "AvatarS0093.mp4"
    assert record.poses_path.name == "gt_AvatarS0093.csv"
    assert record.poses_path.parent.name == "_derived_v2"
    assert record.intrinsics.name == "intrinsicK.csv"
    assert record.data_root == RECORD_PATH.parent.parent
