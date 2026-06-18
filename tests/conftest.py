from __future__ import annotations

import pandas as pd

from scene_recon.geometry.footprint import GroundFootprint
from scene_recon.selection import GroundGrid


def square_footprints(
    candidates: pd.DataFrame, grid: GroundGrid, *, half_cells: int = 1
) -> dict[int, GroundFootprint]:
    """Synthetic nadir footprints: each frame covers a square of cells around its
    pose. Lets selector/coverage tests run without the DTM ray-march engine."""
    fps: dict[int, GroundFootprint] = {}
    for idx, row in candidates.iterrows():
        cx, cy = grid.cell_id(float(row["easting"]), float(row["northing"]))
        cells = frozenset(
            (cx + dx, cy + dy)
            for dx in range(-half_cells, half_cells + 1)
            for dy in range(-half_cells, half_cells + 1)
        )
        fps[int(idx)] = GroundFootprint(
            frame_number=int(idx),
            cells=cells,
            valid=True,
            valid_frac=1.0,
            area_m2=len(cells) * grid.bin_size_m**2,
            centroid_e=float(row["easting"]),
            centroid_n=float(row["northing"]),
            hull_wkt="",
            reject_detail=None,
        )
    return fps
