from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scene_recon.selection.params import (
    MAX_NADIR_ANGLE_DEG,
    MIN_NADIR_ANGLE_DEG,
    SelectionParams,
)

ViewCounts = dict[tuple[int, int], int]
FootprintCache = dict[int, tuple[set[tuple[int, int]], float, float, int]]


@dataclass(frozen=True)
class Footprint:
    center_easting: float
    center_northing: float
    forward_e: float
    forward_n: float
    right_e: float
    right_n: float
    half_width_m: float
    half_depth_m: float
    agl_m: float

    @property
    def area_m2(self) -> float:
        return 4.0 * self.half_width_m * self.half_depth_m

    def corners(self) -> tuple[tuple[float, float], ...]:
        out: list[tuple[float, float]] = []
        for along, across in (
            (-self.half_depth_m, -self.half_width_m),
            (self.half_depth_m, -self.half_width_m),
            (self.half_depth_m, self.half_width_m),
            (-self.half_depth_m, self.half_width_m),
        ):
            easting = self.center_easting + self.forward_e * along + self.right_e * across
            northing = self.center_northing + self.forward_n * along + self.right_n * across
            out.append((easting, northing))
        return tuple(out)


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def translation_m(a: pd.Series, b: pd.Series) -> float:
    de = float(a["easting"] - b["easting"])
    dn = float(a["northing"] - b["northing"])
    return math.hypot(de, dn)


def rotation_deg(a: pd.Series, b: pd.Series) -> float:
    dr = _wrap_angle(float(a.get("roll_rad", 0.0) - b.get("roll_rad", 0.0)))
    dp = _wrap_angle(float(a.get("pitch_rad", 0.0) - b.get("pitch_rad", 0.0)))
    dy = _wrap_angle(float(a.get("yaw_rad", 0.0) - b.get("yaw_rad", 0.0)))
    return math.degrees(math.sqrt(dr * dr + dp * dp + dy * dy))


def assign_bins(candidates: pd.DataFrame, params: SelectionParams) -> pd.DataFrame:
    out = candidates.copy()
    origin_e = out["easting"].min()
    origin_n = out["northing"].min()
    out["cell_x"] = np.floor((out["easting"] - origin_e) / params.bin_size_m).astype("Int64")
    out["cell_y"] = np.floor((out["northing"] - origin_n) / params.bin_size_m).astype("Int64")
    return out


def infer_ground_altitude_m(candidates: pd.DataFrame) -> float:
    altitudes = candidates["altamsl"].dropna().astype(float)
    if altitudes.empty:
        return 0.0
    return float(altitudes.quantile(0.02))


def footprint_for_row(
    row: pd.Series,
    params: SelectionParams,
    *,
    ground_altitude_m: float,
) -> Footprint:
    agl_m = max(params.min_agl_m, _safe_float(row["altamsl"]) - ground_altitude_m)
    yaw = _safe_float(row.get("yaw_rad", 0.0))
    pitch = _safe_float(row.get("pitch_rad", 0.0))

    nadir_angle = math.pi / 2.0 + pitch
    nadir_angle = max(
        math.radians(MIN_NADIR_ANGLE_DEG),
        min(math.radians(MAX_NADIR_ANGLE_DEG), nadir_angle),
    )

    slant_range = min(agl_m / math.cos(nadir_angle), params.max_slant_m)
    forward_distance = slant_range * math.sin(nadir_angle)

    forward_e = math.sin(yaw)
    forward_n = math.cos(yaw)
    right_e = math.cos(yaw)
    right_n = -math.sin(yaw)

    center_easting = _safe_float(row["easting"]) + forward_e * forward_distance
    center_northing = _safe_float(row["northing"]) + forward_n * forward_distance
    half_width_m = max(params.bin_size_m, slant_range * params.footprint_half_width_scale)
    half_depth_m = max(params.bin_size_m, slant_range * params.footprint_half_depth_scale)

    return Footprint(
        center_easting=center_easting,
        center_northing=center_northing,
        forward_e=forward_e,
        forward_n=forward_n,
        right_e=right_e,
        right_n=right_n,
        half_width_m=half_width_m,
        half_depth_m=half_depth_m,
        agl_m=agl_m,
    )


def footprint_cells(footprint: Footprint, *, cell_size_m: float) -> set[tuple[int, int]]:
    corners = footprint.corners()
    min_e = min(e for e, _ in corners)
    max_e = max(e for e, _ in corners)
    min_n = min(n for _, n in corners)
    max_n = max(n for _, n in corners)

    min_x = math.floor(min_e / cell_size_m)
    max_x = math.floor(max_e / cell_size_m)
    min_y = math.floor(min_n / cell_size_m)
    max_y = math.floor(max_n / cell_size_m)

    cells: set[tuple[int, int]] = set()
    for cell_x in range(min_x, max_x + 1):
        center_e = (cell_x + 0.5) * cell_size_m
        for cell_y in range(min_y, max_y + 1):
            center_n = (cell_y + 0.5) * cell_size_m
            de = center_e - footprint.center_easting
            dn = center_n - footprint.center_northing
            along = de * footprint.forward_e + dn * footprint.forward_n
            across = de * footprint.right_e + dn * footprint.right_n
            if abs(along) <= footprint.half_depth_m and abs(across) <= footprint.half_width_m:
                cells.add((cell_x, cell_y))
    return cells


def recount_views(
    selected: set[int],
    out: pd.DataFrame,
    params: SelectionParams,
    ground_altitude_m: float,
    footprint_cache: FootprintCache,
) -> ViewCounts:
    counts: ViewCounts = {}
    for idx in selected:
        if idx in footprint_cache:
            cells, _agl, _area, _n = footprint_cache[idx]
        else:
            row = out.loc[idx]
            footprint = footprint_for_row(row, params, ground_altitude_m=ground_altitude_m)
            cells = footprint_cells(footprint, cell_size_m=params.bin_size_m)
        for c in cells:
            counts[c] = counts.get(c, 0) + 1
    return counts


def compute_view_counts(candidates: pd.DataFrame, params: SelectionParams) -> ViewCounts:
    selected = candidates[candidates["selected"]]
    if selected.empty:
        return {}
    ground_altitude_m = infer_ground_altitude_m(selected)
    counts: ViewCounts = {}
    for idx in selected.index:
        row = selected.loc[idx]
        footprint = footprint_for_row(row, params, ground_altitude_m=ground_altitude_m)
        cells = footprint_cells(footprint, cell_size_m=params.bin_size_m)
        for c in cells:
            counts[c] = counts.get(c, 0) + 1
    return counts
