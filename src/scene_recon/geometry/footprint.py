from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from scene_recon.camera import Camera
from scene_recon.geometry.extrinsics import CameraPose, world_rays
from scene_recon.geometry.raymarch import raymarch_first_hit
from scene_recon.geometry.terrain import TerrainModel
from scene_recon.selection.grid import GroundGrid


@dataclass(frozen=True)
class GroundFootprint:
    """The ground a frame actually sees: cells hit by first-intersection rays."""

    frame_number: int
    cells: frozenset[tuple[int, int]]
    valid: bool
    valid_frac: float
    area_m2: float
    centroid_e: float
    centroid_n: float
    hull_wkt: str
    reject_detail: str | None


def _convex_hull_wkt(points: np.ndarray) -> str:
    pts = sorted({(round(float(e), 2), round(float(n), 2)) for e, n in points})
    if len(pts) < 3:
        ring = pts
    else:

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper: list[tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        ring = lower[:-1] + upper[:-1]
    if not ring:
        return "POLYGON EMPTY"
    closed = ring + [ring[0]]
    inner = ", ".join(f"{e:.2f} {n:.2f}" for e, n in closed)
    return f"POLYGON (({inner}))"


def ground_footprint(
    camera: Camera,
    pose: CameraPose,
    terrain: TerrainModel,
    grid: GroundGrid,
    *,
    frame_number: int,
    ray_grid: tuple[int, int] = (48, 27),
    max_range_m: float = 2000.0,
    step_m: float = 10.0,
    min_valid_ray_frac: float = 0.25,
) -> GroundFootprint:
    pixels = camera.sample_grid(ray_grid)
    origins, dirs = world_rays(camera, pose, pixels)
    hits, valid = raymarch_first_hit(
        origins, dirs, terrain, step_m=step_m, max_range_m=max_range_m
    )
    valid_frac = float(valid.mean()) if valid.size else 0.0
    ground = hits[valid]
    cells = grid.cells_for_points(ground)

    if valid_frac < min_valid_ray_frac:
        reject: str | None = "too_few_rays"
    elif not cells:
        reject = "no_cells"
    else:
        reject = None
    is_valid = reject is None

    if ground.size:
        centroid_e = float(ground[:, 0].mean())
        centroid_n = float(ground[:, 1].mean())
        hull_wkt = _convex_hull_wkt(ground[:, :2])
    else:
        centroid_e = centroid_n = float("nan")
        hull_wkt = "POLYGON EMPTY"

    return GroundFootprint(
        frame_number=frame_number,
        cells=cells,
        valid=is_valid,
        valid_frac=valid_frac,
        area_m2=len(cells) * grid.bin_size_m**2,
        centroid_e=centroid_e,
        centroid_n=centroid_n,
        hull_wkt=hull_wkt,
        reject_detail=reject,
    )


def compute_footprints(
    candidates: pd.DataFrame,
    camera: Camera,
    terrain: TerrainModel,
    grid: GroundGrid,
    *,
    ray_grid: tuple[int, int] = (48, 27),
    max_range_m: float = 2000.0,
    step_m: float = 10.0,
    min_valid_ray_frac: float = 0.25,
    show_progress: bool = True,
) -> dict[int, GroundFootprint]:
    """Ray-march every candidate once. Drives both selection scoring and the
    mission-region coverage denominator."""
    rows = candidates.iterrows()
    if show_progress:
        rows = tqdm(rows, total=len(candidates), desc="Footprints", unit="frame")
    out: dict[int, GroundFootprint] = {}
    for idx, row in rows:
        pose = CameraPose.from_row(row)
        out[int(idx)] = ground_footprint(
            camera,
            pose,
            terrain,
            grid,
            frame_number=int(idx),
            ray_grid=ray_grid,
            max_range_m=max_range_m,
            step_m=step_m,
            min_valid_ray_frac=min_valid_ray_frac,
        )
    return out
