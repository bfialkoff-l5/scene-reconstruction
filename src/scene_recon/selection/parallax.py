"""Parallax-aware coverage: 3D convergence angle between viewing directions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scene_recon.selection.footprint import FootprintCache
from scene_recon.selection.grid import GroundGrid
from scene_recon.selection.params import SelectionParams

CellViewers = dict[tuple[int, int], list[int]]
CellGroundZ = dict[tuple[int, int], float]


def max_convergence_deg(directions: np.ndarray) -> float:
    if directions.shape[0] < 2:
        return 0.0
    g = directions @ directions.T
    mincos = float(np.clip(g.min(), -1.0, 1.0))
    return math.degrees(math.acos(mincos))


@dataclass
class ParallaxContext:
    """Precomputed camera positions — built once, no pandas in the hot path."""

    cam_pos: dict[int, np.ndarray]
    _cell_centers: dict[tuple[int, int], tuple[float, float, float]] = field(
        default_factory=dict
    )

    def cell_center_3d(
        self, cell: tuple[int, int], grid: GroundGrid, cell_ground_z: CellGroundZ, frame_idx: int
    ) -> np.ndarray:
        key = cell
        if key not in self._cell_centers:
            ce, cn = grid.cell_center(cell)
            cz = cell_ground_z.get(cell, float("nan"))
            if not math.isfinite(cz):
                cz = float(self.cam_pos[frame_idx][2])
            self._cell_centers[key] = (ce, cn, cz)
        ce, cn, cz = self._cell_centers[key]
        return np.array([ce, cn, cz], dtype=float)


def build_parallax_context(out: pd.DataFrame) -> ParallaxContext:
    cam_pos = {
        int(idx): np.array(
            [float(row["easting"]), float(row["northing"]), float(row["altamsl"])],
            dtype=float,
        )
        for idx, row in out.iterrows()
    }
    return ParallaxContext(cam_pos)


def cell_direction(
    frame_idx: int,
    cell: tuple[int, int],
    ctx: ParallaxContext,
    grid: GroundGrid,
    cell_ground_z: CellGroundZ,
) -> np.ndarray:
    pos = ctx.cam_pos[frame_idx]
    d = pos - ctx.cell_center_3d(cell, grid, cell_ground_z, frame_idx)
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else d


# --- reporting / verification helpers (convergence measurement) ---


def _view_directions(
    viewer_indices: list[int],
    cell: tuple[int, int],
    ctx: ParallaxContext,
    grid: GroundGrid,
    cell_ground_z: CellGroundZ,
) -> np.ndarray:
    if not viewer_indices:
        return np.empty((0, 3))
    return np.stack(
        [cell_direction(int(f), cell, ctx, grid, cell_ground_z) for f in viewer_indices]
    )


def convergence_by_cell(
    cell_viewers: CellViewers,
    ctx: ParallaxContext,
    grid: GroundGrid,
    cell_ground_z: CellGroundZ,
) -> dict[tuple[int, int], float]:
    """Max pairwise convergence angle (deg) per covered cell."""
    out: dict[tuple[int, int], float] = {}
    for cell, frames in cell_viewers.items():
        if len(frames) < 2:
            out[cell] = 0.0
        else:
            out[cell] = max_convergence_deg(
                _view_directions(frames, cell, ctx, grid, cell_ground_z)
            )
    return out


def parallax_satisfied(
    viewer_indices: list[int],
    cell: tuple[int, int],
    ctx: ParallaxContext,
    grid: GroundGrid,
    cell_ground_z: CellGroundZ,
    params: SelectionParams,
) -> bool:
    if len(viewer_indices) < params.parallax_min_views:
        return False
    return (
        max_convergence_deg(_view_directions(viewer_indices, cell, ctx, grid, cell_ground_z))
        >= params.parallax_min_convergence_deg
    )


def mission_cell_ground_z(
    grid: GroundGrid,
    mission_cells: frozenset[tuple[int, int]],
    out: pd.DataFrame,
    terrain,
) -> CellGroundZ:
    """Sample DTM elevation at mission cell centers. ``terrain`` is a TerrainModel."""
    if not mission_cells:
        return {}
    cells = list(mission_cells)
    ce = np.array([grid.cell_center(c)[0] for c in cells], dtype=float)
    cn = np.array([grid.cell_center(c)[1] for c in cells], dtype=float)
    zs = terrain.elevation_at(ce, cn)
    return {cells[i]: float(zs[i]) for i in range(len(cells))}


def approx_cell_ground_z(
    grid: GroundGrid,
    mission_cells: frozenset[tuple[int, int]],
    out: pd.DataFrame,
) -> CellGroundZ:
    """ponytail: flat-ground fallback when DTM is unavailable (unit tests)."""
    z = float(out["altamsl"].median())
    return {c: z for c in mission_cells}


def cell_viewers_from_selection(
    candidates: pd.DataFrame, footprints: FootprintCache
) -> CellViewers:
    viewers: CellViewers = {}
    for idx in candidates[candidates["selected"]].index:
        fp = footprints.get(int(idx))
        if fp is None:
            continue
        for cell in fp.cells:
            viewers.setdefault(cell, []).append(int(idx))
    return viewers


@dataclass(frozen=True)
class ParallaxMetrics:
    n_cells_covered: int
    n_parallax_satisfied: int
    pct_covered_parallax_satisfied: float
    median_max_convergence_deg: float
    convergence_quantiles_deg: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "n_cells_covered": self.n_cells_covered,
            "n_parallax_satisfied": self.n_parallax_satisfied,
            "pct_covered_parallax_satisfied": round(self.pct_covered_parallax_satisfied, 4),
            "median_max_convergence_deg": round(self.median_max_convergence_deg, 1),
            "convergence_quantiles_deg": self.convergence_quantiles_deg,
        }


def parallax_metrics(
    cell_viewers: CellViewers,
    ctx: ParallaxContext,
    grid: GroundGrid,
    cell_ground_z: CellGroundZ,
    params: SelectionParams,
) -> ParallaxMetrics:
    conv_deg: list[float] = []
    n_sat = 0
    for cell, frames in cell_viewers.items():
        if parallax_satisfied(frames, cell, ctx, grid, cell_ground_z, params):
            n_sat += 1
        if len(frames) >= 2:
            conv_deg.append(
                max_convergence_deg(_view_directions(frames, cell, ctx, grid, cell_ground_z))
            )
    n_cov = len(cell_viewers)
    pct = n_sat / max(n_cov, 1)
    quantiles = (
        {
            q: round(float(np.quantile(conv_deg, p)), 1)
            for q, p in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
        }
        if conv_deg
        else {}
    )
    return ParallaxMetrics(
        n_cells_covered=n_cov,
        n_parallax_satisfied=n_sat,
        pct_covered_parallax_satisfied=pct,
        median_max_convergence_deg=float(np.median(conv_deg)) if conv_deg else 0.0,
        convergence_quantiles_deg=quantiles,
    )
