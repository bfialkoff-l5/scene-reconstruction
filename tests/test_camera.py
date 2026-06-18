from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scene_recon.camera import Camera


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


def _example_camera() -> Camera:
    return Camera(
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


def test_from_csv(tmp_path: Path) -> None:
    intrinsic_path = tmp_path / "intrinsicK.csv"
    _write_intrinsic_k(intrinsic_path)

    cam = Camera.from_csv(intrinsic_path, width=1920, height=1080)
    assert cam.fx == pytest.approx(2274.31688)
    assert cam.fy == pytest.approx(2276.21216)
    assert cam.cx == pytest.approx(945.863866)
    assert cam.cy == pytest.approx(493.567277)
    assert cam.k1 == pytest.approx(-0.01915783)
    assert cam.k3 == pytest.approx(0.59895634)
    assert cam.width == 1920
    assert cam.height == 1080


def test_odm_normalization() -> None:
    cam = _example_camera()
    assert cam.focal_x == pytest.approx(2274.31688 / 1920)
    assert cam.focal_y == pytest.approx(2276.21216 / 1920)
    assert cam.c_x == pytest.approx((945.863866 - 960) / 1920)
    assert cam.c_y == pytest.approx((493.567277 - 540) / 1920)
    assert "1920 1080 brown" in cam.camera_id()
    entry = cam.to_odm_entry()
    assert entry["projection_type"] == "brown"
    assert entry["k1"] == pytest.approx(-0.01915783)


def test_pixels_to_rays() -> None:
    cam = _example_camera()
    pixels = np.array(
        [[cam.cx, cam.cy], [100.0, 200.0], [1500.0, 900.0]], dtype=np.float64
    )
    rays = cam.pixels_to_rays(pixels)
    assert rays.shape == (3, 3)
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(rays[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_project_roundtrip() -> None:
    cam = _example_camera()
    pixels = np.array([[100.0, 200.0], [1500.0, 900.0], [800.0, 400.0]], dtype=np.float64)
    rays = cam.pixels_to_rays(pixels)
    back = cam.project(rays)
    assert np.allclose(back, pixels, atol=1e-3)


def test_sample_grid() -> None:
    cam = _example_camera()
    grid = cam.sample_grid((48, 27))
    assert grid.shape == (48 * 27, 2)
    assert grid[:, 0].min() >= 0.0 and grid[:, 0].max() <= cam.width - 1
    assert grid[:, 1].min() >= 0.0 and grid[:, 1].max() <= cam.height - 1
