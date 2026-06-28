from __future__ import annotations

import json
from math import radians

import numpy as np
import pandas as pd

from scene_recon.export import write_geo_txt, write_rotation_priors
from scene_recon.geometry.extrinsics import CameraPose


def _rotation_from_opk(omega: float, phi: float, kappa: float) -> np.ndarray:
    """World->camera rotation from omega/phi/kappa (radians).

    Independent reimplementation of opensfm.geometry.rotation_from_opk, so this test
    fails if our encoder (or the convention assumed by the patched bundle adjustment)
    ever drifts. The whole rotation-prior fix depends on this being the exact inverse
    of the encoder in export._opk_degrees_from_cam_to_enu.
    """

    def aa(v: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(v)
        if theta < 1e-12:
            return np.eye(3)
        k = v / theta
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)

    rw = aa(np.array([-omega, 0.0, 0.0]))
    rp = aa(np.array([0.0, -phi, 0.0]))
    rk = aa(np.array([0.0, 0.0, -kappa]))
    rc = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    return rc @ rk @ rp @ rw


def test_rotation_priors_round_trip_R_cam_to_enu(tmp_path) -> None:
    # The fix hinges on omega/phi/kappa being a *lossless* encoding of camera attitude:
    # rotation_from_opk(opk) must reproduce world->cam (= R_cam_to_enu.T) exactly, for
    # oblique/strapdown poses where ODM's geo YPR -> compute_opk path loses ~40 deg.
    selected = pd.DataFrame(
        {
            "utm_zone": ["36N", "36N", "36N"],
            "easting": [700000.0, 700010.0, 700020.0],
            "northing": [3500000.0, 3500020.0, 3500040.0],
            "altamsl": [120.0, 121.0, 122.0],
            "yaw_rad": [radians(90.0), radians(-45.0), radians(170.0)],
            "pitch_rad": [radians(35.0), radians(-25.0), radians(50.0)],
            "roll_rad": [radians(12.0), radians(-8.0), radians(3.0)],
        },
        index=[1, 2, 3],
    )

    out = tmp_path / "rotation_priors.json"
    write_rotation_priors(selected, out)
    priors = json.loads(out.read_text())

    assert set(priors) == {"000001.png", "000002.png", "000003.png"}
    for frame_number, row in selected.iterrows():
        opk = priors[f"{int(frame_number):06d}.png"]
        r_recovered = _rotation_from_opk(
            radians(opk["omega"]), radians(opk["phi"]), radians(opk["kappa"])
        )
        world_to_cam = CameraPose.from_row(row).R_cam_to_enu().T
        assert np.allclose(r_recovered, world_to_cam, atol=1e-9)


def test_write_geo_txt_is_position_only(tmp_path) -> None:
    # geo.txt is intentionally position-only: ODM's geo YPR -> opk conversion is
    # lossy for our strapdown camera and opk is never a BA rotation prior, so
    # emitting orientation cannot help and a wrong convention corrupts matching.
    # See docs/FINDINGS.md (2026-06-25).
    selected = pd.DataFrame(
        {
            "utm_zone": ["36N", "36N"],
            "easting": [700000.0, 700010.0],
            "northing": [3500000.0, 3500020.0],
            "altamsl": [120.0, 121.0],
            "yaw_rad": [radians(90.0), radians(-45.0)],
            "pitch_rad": [radians(10.0), radians(-5.0)],
            "roll_rad": [radians(2.0), radians(-3.0)],
        },
        index=[1, 2],
    )

    out = tmp_path / "geo.txt"
    write_geo_txt(selected, out)

    lines = out.read_text().splitlines()
    assert lines[0] == "WGS84 UTM 36N"

    first = lines[1].split()
    assert len(first) == 4  # name + easting/northing/alt, NO orientation
    assert first[0] == "000001.png"
    assert float(first[1]) == 700000.0
    assert float(first[2]) == 3500000.0
    assert float(first[3]) == 120.0
