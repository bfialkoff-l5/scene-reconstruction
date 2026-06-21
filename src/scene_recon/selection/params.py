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

# Stage-1 keyframe spacing: keep the next frame once the camera has moved this many
# metres from the last kept frame. This is the *baseline* between consecutive keyframes
# and is what triangulation actually needs -- at survey depth D the baseline/depth ratio
# is spacing/D, and SfM wants ~0.05-0.15 (Snavely/Goesele discard tiny-baseline pairs).
#
# 4 m is the empirical optimum on 0088 (~80 m AGL): it yields 475 keyframes whose nearest
# neighbour is ~4 m and 16th-nearest (the --matcher-neighbors 16 reach) is ~33 m, and
# that run is our best ODM result -- 33.5 ha mapped, 68k sparse points, track length 7.3,
# 89% of core cells at >=3 views. Pushing coverage further is a spacing knob, not a
# separate objective: 3 m -> 624 frames / 91% but a tighter 25 m matcher reach.
#
# Do NOT add a per-cell view-count cull on top: tested at 943 frames (target 3 views,
# incidence-bounded core) it packed near-duplicates 0.42 m apart, collapsed the matcher
# reach 33 m -> 7.7 m, and regressed the ortho to 25.2 ha / 60k points / track length 16.5
# (deep-but-narrow degenerate triangulation). View count is parallax-blind; spacing is the
# control signal that protects the baseline. See git history for the reverted cull.
#
# Why distance and not footprint-Jaccard: a Jaccard trigger measured the footprint polygon,
# which at oblique pitch + DTM ray-march swings ~80x faster than the camera (sub-degree
# attitude noise -> tens of metres of footprint jitter). It fired on noise, not motion, and
# piled up bursts of ~0.2 m-apart near-duplicates with zero parallax. Camera position is
# GPS-stable and not orientation-amplified, so distance is the robust control signal.
KEYFRAME_SPACING_M = 4.0

# GSD-consistency floor (Goesele 2007 view selection: keep matched views within a
# small resolution ratio, ~1<=r<2). GSD scales with AGL, so we drop frames flying
# lower than the flight median AGL by more than this ratio. Kills near-ground frames
# whose sub-cm GSD poisons ODM's DSM sizing (a 1 cm DSM over a 410 m site -> OOM)
# without trimming legitimate altitude variation. Self-calibrating to the flight's
# own median; a large value disables the gate (floor -> 0). Acts on the per-frame
# agl_m column (build.py samples it from the DTM); a footprint-area proxy was tried
# and rejected -- pitched low frames see distant ground, so their footprint is near
# median and the real sub-cm-GSD outliers slip through.
GSD_RATIO_MAX = 3.0

# Over-coverage cull (ORB-SLAM "survival of the fittest" / COLMAP redundant-image
# pruning): drop a keyframe when every cell in its footprint still has > this many
# views from other keepers, worst-quality-first. Tames hover/orbit redundancy
# (0088 hit 45 views/cell) without opening holes. 0 disables.
# ponytail: coverage-count only, ignores parallax — a wide-baseline frame can be
# culled if its cells are otherwise saturated. Upgrade path: skip culling a frame
# that uniquely raises an under-target cell's convergence angle.
MAX_VIEWS_PER_CELL = 0

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
# Max ray slant range. On 0088 no ray ever hits past ~1113 m (p99.9 = 914 m), and
# cells seen only beyond ~800 m are far near-horizon fringe (huge GSD, grazing,
# low-value for reconstruction). 800 keeps 99.4% of cells and stops shallow rays
# from marching the old 2000 m ceiling (~200 steps) on oblique/takeoff frames,
# which was the cause of the throughput decay. Raise toward 1200 for bit-identical
# footprints if a flight flies higher / more oblique.
MAX_RANGE_M = 800.0
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
    keyframe_spacing_m: float = KEYFRAME_SPACING_M
    gsd_ratio_max: float = GSD_RATIO_MAX
    max_views_per_cell: int = MAX_VIEWS_PER_CELL
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
