from __future__ import annotations

from pathlib import Path

import pytest

from scene_recon.intrinsics import (
    CameraIntrinsics,
    cameras_json_for_record,
    load_intrinsic_k,
)


def _write_intrinsic_k(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2274.31688,0,945.863866,,",
                "0,2276.21216,493.567277,,",
                "0,0,1,,",
                "-0.01915783,0.12972421,-0.01080202,-0.00100774,0.59895634",
            ]
        )
        + "\n"
    )


def test_load_intrinsic_k(tmp_path: Path) -> None:
    intrinsic_path = tmp_path / "intrinsicK.csv"
    _write_intrinsic_k(intrinsic_path)

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = load_intrinsic_k(intrinsic_path)
    assert fx == pytest.approx(2274.31688)
    assert fy == pytest.approx(2276.21216)
    assert cx == pytest.approx(945.863866)
    assert cy == pytest.approx(493.567277)
    assert k1 == pytest.approx(-0.01915783)
    assert k3 == pytest.approx(0.59895634)


def test_camera_intrinsics_normalization() -> None:
    intrinsics = CameraIntrinsics(
        fx=2274.31688,
        fy=2276.21216,
        cx=945.863866,
        cy=493.567277,
        k1=-0.01915783,
        k2=0.12972421,
        p1=-0.01080202,
        p2=-0.00100774,
        k3=0.59895634,
        width=1920,
        height=1080,
    )

    assert intrinsics.focal_x == pytest.approx(2274.31688 / 1920)
    assert intrinsics.focal_y == pytest.approx(2276.21216 / 1920)
    assert intrinsics.c_x == pytest.approx((945.863866 - 960) / 1920)
    assert intrinsics.c_y == pytest.approx((493.567277 - 540) / 1920)
    assert "1920 1080 brown" in intrinsics.camera_id()


@pytest.mark.skipif(
    not Path("/home/bfialkoff/s3/raw/0088_20260122_eitan_1").is_dir(),
    reason="example record not on disk",
)
def test_cameras_json_for_example_record() -> None:
    from scene_recon.record import Record

    record = Record.from_path("/home/bfialkoff/s3/raw/0088_20260122_eitan_1")
    payload = cameras_json_for_record(record)
    assert len(payload) == 1
    camera = next(iter(payload.values()))
    assert camera["projection_type"] == "brown"
    assert camera["width"] == 1920
    assert camera["height"] == 1080
    assert camera["focal_x"] == pytest.approx(2274.31688 / 1920, rel=1e-4)
