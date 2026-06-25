from __future__ import annotations

from math import radians

import pandas as pd

from scene_recon.export import write_geo_txt


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
