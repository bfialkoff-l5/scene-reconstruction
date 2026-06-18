from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path

SELECTION_POLICY = "frustum_view_count_dtm"

BIN_SIZE_M = 5.0
MIN_ALTITUDE_M = 0.0
# Grounded-frame trim: a frame is airborne once its altitude has risen this far
# above the flight's resting level. Measured as a *rise above resting*, so the
# absolute datum cancels (robust to DTM/datum error). 2 m clears pre-takeoff jitter.
GROUND_TRIM_RISE_M = 2.0
CLUSTER_RADIUS_M = 10.0
COVERAGE_WARN_M = 15.0
MAX_MOTION_GAP_M = 60.0
# Eval / uncapped default: ~all eligible frames (0088 superset ~3810 at T=0.5, so this
# is effectively uncapped). ponytail: real production cap (ODM image-count tolerance) is
# the one open dial; lower it once the first end-to-end ODM run sets the ceiling.
MAX_KEYFRAMES = 15000

# Stage-1 overlap spacing: keep the next frame once its footprint Jaccard with the
# last kept frame drops to <= this. 0.5 ~= high overlap (master-plan probe value).
OVERLAP_JACCARD_TARGET = 0.5

TARGET_VIEWS_PER_CELL = 5
# Parallax/convergence target (verification metric, not an objective).
PARALLAX_MIN_VIEWS = 3
PARALLAX_MIN_CONVERGENCE_DEG = 10.0
# Honest coverage gate: fraction of mission cells that must be covered (>=1 view) by
# the final selection. Stage-1 (no cap) reaches ~96% on 0088; capped degrades
# predictably. 0.5 leaves margin while still flagging a too-small budget.
MIN_PCT_MISSION_COVERED = 0.5

# Footprint engine (DTM ray-march)
RAY_GRID = (48, 27)
MAX_RANGE_M = 2000.0
RAY_STEP_M = 10.0
MIN_VALID_RAY_FRAC = 0.25
DATUM_OFFSET_M = 0.0
TERRAIN_MARGIN_M = 500.0


@dataclass(frozen=True)
class SelectionParams:
    bin_size_m: float = BIN_SIZE_M
    terrain_gpkg: Path | None = None
    datum_offset_m: float = DATUM_OFFSET_M
    ray_grid: tuple[int, int] = RAY_GRID
    ray_step_m: float = RAY_STEP_M
    max_range_m: float = MAX_RANGE_M
    min_valid_ray_frac: float = MIN_VALID_RAY_FRAC
    terrain_margin_m: float = TERRAIN_MARGIN_M
    min_altitude_m: float = MIN_ALTITUDE_M
    ground_trim_rise_m: float = GROUND_TRIM_RISE_M
    overlap_jaccard_target: float = OVERLAP_JACCARD_TARGET
    cluster_radius_m: float = CLUSTER_RADIUS_M
    coverage_warn_m: float = COVERAGE_WARN_M
    max_motion_gap_m: float = MAX_MOTION_GAP_M
    max_keyframes: int = MAX_KEYFRAMES
    target_views_per_cell: int = TARGET_VIEWS_PER_CELL
    parallax_min_views: int = PARALLAX_MIN_VIEWS
    parallax_min_convergence_deg: float = PARALLAX_MIN_CONVERGENCE_DEG
    min_pct_mission_covered: float = MIN_PCT_MISSION_COVERED

    def as_constants(self) -> dict:
        constants = {"policy": SELECTION_POLICY, **asdict(self)}
        if constants.get("terrain_gpkg") is not None:
            constants["terrain_gpkg"] = str(constants["terrain_gpkg"])
        constants["ray_grid"] = list(self.ray_grid)
        return constants


DEFAULT_SELECTION_PARAMS = SelectionParams()


def params_from_constants(constants: dict) -> SelectionParams:
    valid = {f.name for f in dataclass_fields(SelectionParams)}
    kwargs = {k: v for k, v in constants.items() if k in valid}
    if isinstance(kwargs.get("terrain_gpkg"), str):
        kwargs["terrain_gpkg"] = Path(kwargs["terrain_gpkg"])
    if isinstance(kwargs.get("ray_grid"), list):
        kwargs["ray_grid"] = tuple(kwargs["ray_grid"])
    return SelectionParams(**kwargs)
