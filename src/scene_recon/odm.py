from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Target matching baseline reach. --matcher-neighbors is a fixed *count*, so the physical
# baseline it spans shrinks as keyframe spacing packs frames denser (reach ~ sqrt(k*s)).
# We instead pin the reach: pick k so each frame matches every neighbour within this many
# metres. 33 m is the reach of our best run (4 m spacing, k=16); holding it keeps
# triangulation healthy at any spacing instead of silently degrading (3 m -> 25 m -> noisy
# depth; the 943-frame cull -> 7.7 m -> catastrophic).
MATCHER_REACH_M = 33.0

# Documented baseline only — run_odm.sh passes --fast-orthophoto when no args are given.
# matcher_neighbors is computed per selection (see recommend_matcher_neighbors) so the
# matching graph spans MATCHER_REACH_M regardless of spacing; run_odm.sh reads it from
# odm_options.json. run_odm.sh always binds cameras.json via --cameras
# --use-fixed-camera-params so ODM uses our lab calibration instead of self-calibrating.
DEFAULT_ODM_OPTIONS: dict[str, object] = {
    "fast_orthophoto": True,
    "matcher_neighbors": 0,
    "gps_accuracy": 3,
    "orthophoto_resolution": 5,
    "use_fixed_camera_params": True,
}


def recommend_matcher_neighbors(
    eastings, northings, *, reach_m: float = MATCHER_REACH_M
) -> int:
    """The k for --matcher-neighbors that makes the matching graph span ~reach_m on this
    selection's actual geometry: the median number of other keyframes within reach_m of a
    keyframe. Self-calibrating to flight density, so a denser spacing automatically asks
    ODM for more neighbours and the effective baseline stays put.
    ponytail: O(n^2) over selected frames (~hundreds); fine well past any real keyframe count."""
    p = np.column_stack([np.asarray(eastings, float), np.asarray(northings, float)])
    if len(p) < 2:
        return 0
    r2 = reach_m * reach_m
    within = np.array([int(((p - p[i]) ** 2).sum(1).__le__(r2).sum()) - 1 for i in range(len(p))])
    return max(int(np.median(within)), 1)


def default_odm_cli_args() -> list[str]:
    return ["--fast-orthophoto"]


def write_odm_options(
    odm_input: Path, *, cameras_path: Path, matcher_neighbors: int = 0
) -> None:
    payload = {
        **DEFAULT_ODM_OPTIONS,
        "matcher_neighbors": matcher_neighbors,
        "cameras": str(cameras_path),
        "cli_args_if_empty": default_odm_cli_args(),
        "note": (
            "matcher_neighbors is auto-scaled to hold ~%.0f m matching baseline reach for "
            "this selection's frame density (run_odm.sh passes it unless overridden). "
            "cameras.json is exported from intrinsicK.csv and bound by run_odm.sh via "
            "--cameras --use-fixed-camera-params; its key matches ODM's detected camera id "
            "(see Camera.camera_id) so the override binds instead of ODM self-calibrating"
            % MATCHER_REACH_M
        ),
    }
    (odm_input / "odm_options.json").write_text(json.dumps(payload, indent=2) + "\n")
