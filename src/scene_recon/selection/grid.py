from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroundGrid:
    bin_size_m: float
    origin_e: float
    origin_n: float

    @classmethod
    def from_poses(
        cls,
        candidates: pd.DataFrame,
        *,
        bin_size_m: float,
        margin_m: float = 0.0,
    ) -> "GroundGrid":
        min_e = float(candidates["easting"].min())
        min_n = float(candidates["northing"].min())
        e0 = np.floor((min_e - margin_m) / bin_size_m) * bin_size_m
        n0 = np.floor((min_n - margin_m) / bin_size_m) * bin_size_m
        return cls(bin_size_m=bin_size_m, origin_e=float(e0), origin_n=float(n0))

    def cell_id(self, easting: float, northing: float) -> tuple[int, int]:
        cx = int(np.floor((easting - self.origin_e) / self.bin_size_m))
        cy = int(np.floor((northing - self.origin_n) / self.bin_size_m))
        return (cx, cy)

    def cells_for_points(self, points: np.ndarray) -> frozenset[tuple[int, int]]:
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return frozenset()
        en = pts[:, :2]
        en = en[~np.isnan(en).any(axis=1)]
        if en.size == 0:
            return frozenset()
        cells = np.floor(
            (en - (self.origin_e, self.origin_n)) / self.bin_size_m
        ).astype(int)
        return frozenset(map(tuple, np.unique(cells, axis=0)))

    def mission_cells(self, footprints: Iterable) -> frozenset[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for fp in footprints:
            if fp.valid:
                cells |= fp.cells
        return frozenset(cells)

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        cx, cy = cell
        return (
            (cx + 0.5) * self.bin_size_m + self.origin_e,
            (cy + 0.5) * self.bin_size_m + self.origin_n,
        )
