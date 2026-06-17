from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields

SELECTION_POLICY = "frustum_view_count_v3"

BIN_SIZE_M = 5.0
MIN_ALTITUDE_M = 0.0
MIN_TRANSLATION_M = 2.0
MIN_ROTATION_DEG = 5.0
MAX_FRAME_GAP = 90
CLUSTER_RADIUS_M = 10.0
MAX_PER_CLUSTER = 8
COVERAGE_WARN_M = 15.0
MAX_MOTION_GAP_M = 60.0
MAX_KEYFRAMES = 500

TARGET_VIEWS_PER_CELL = 5
MIN_PCT_CELLS_AT_TARGET = 0.7

MIN_COVERAGE_GAIN_RATIO = 0.08
MIN_COVERAGE_GAIN_CELLS = 8
FOOTPRINT_HALF_WIDTH_SCALE = 0.45
FOOTPRINT_HALF_DEPTH_SCALE = 0.75
MIN_AGL_M = 2.0
MAX_SLANT_M = 60.0
MIN_NADIR_ANGLE_DEG = 3.0
MAX_NADIR_ANGLE_DEG = 85.0

SCORE_QUALITY_WEIGHT = 0.30
SCORE_COVERAGE_WEIGHT = 0.50
SCORE_NOVELTY_WEIGHT = 0.20

CONNECTION_RADIUS_M = 30.0
MAIN_COMPONENT_RATIO = 0.7


@dataclass(frozen=True)
class SelectionParams:
    bin_size_m: float = BIN_SIZE_M
    min_altitude_m: float = MIN_ALTITUDE_M
    min_translation_m: float = MIN_TRANSLATION_M
    min_rotation_deg: float = MIN_ROTATION_DEG
    max_frame_gap: int = MAX_FRAME_GAP
    cluster_radius_m: float = CLUSTER_RADIUS_M
    max_per_cluster: int = MAX_PER_CLUSTER
    coverage_warn_m: float = COVERAGE_WARN_M
    max_motion_gap_m: float = MAX_MOTION_GAP_M
    max_keyframes: int = MAX_KEYFRAMES
    target_views_per_cell: int = TARGET_VIEWS_PER_CELL
    min_pct_cells_at_target: float = MIN_PCT_CELLS_AT_TARGET
    min_coverage_gain_ratio: float = MIN_COVERAGE_GAIN_RATIO
    min_coverage_gain_cells: int = MIN_COVERAGE_GAIN_CELLS
    footprint_half_width_scale: float = FOOTPRINT_HALF_WIDTH_SCALE
    footprint_half_depth_scale: float = FOOTPRINT_HALF_DEPTH_SCALE
    min_agl_m: float = MIN_AGL_M
    max_slant_m: float = MAX_SLANT_M
    score_quality_weight: float = SCORE_QUALITY_WEIGHT
    score_coverage_weight: float = SCORE_COVERAGE_WEIGHT
    score_novelty_weight: float = SCORE_NOVELTY_WEIGHT
    connection_radius_m: float = CONNECTION_RADIUS_M
    main_component_ratio: float = MAIN_COMPONENT_RATIO

    def as_constants(self) -> dict:
        return {
            "policy": SELECTION_POLICY,
            **asdict(self),
        }


DEFAULT_SELECTION_PARAMS = SelectionParams()


def params_from_constants(constants: dict) -> SelectionParams:
    valid = {f.name for f in dataclass_fields(SelectionParams)}
    kwargs = {k: v for k, v in constants.items() if k in valid}
    return SelectionParams(**kwargs)
