from __future__ import annotations

from pathlib import Path

import pytest

from scene_recon.record import Record

RECORD_PATH = Path("/home/bfialkoff/s3/raw/0088_20260122_eitan_1")


@pytest.mark.skipif(not RECORD_PATH.is_dir(), reason="example record not on disk")
def test_record_from_path_example() -> None:
    record = Record.from_path(RECORD_PATH)
    assert record.slug == "0088_20260122_eitan_1"
    assert record.stream_id == "AvatarS0093"
    assert record.video.name == "AvatarS0093.mp4"
    assert record.poses_path.name == "gt_AvatarS0093.csv"
    assert record.intrinsics.name == "intrinsicK.csv"
    assert record.data_root == RECORD_PATH.parent.parent
