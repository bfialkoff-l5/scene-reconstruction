from __future__ import annotations

import json
from pathlib import Path

# Documented baseline only — run_odm.sh passes --fast-orthophoto when no args are given.
# Stock ODM matching (matcher_neighbors=0, gps_accuracy=3) matches our successful runs.
# run_odm.sh always binds cameras.json via --cameras --use-fixed-camera-params so ODM
# uses our lab calibration instead of self-calibrating from a default focal.
DEFAULT_ODM_OPTIONS: dict[str, object] = {
    "fast_orthophoto": True,
    "matcher_neighbors": 0,
    "gps_accuracy": 3,
    "orthophoto_resolution": 5,
    "use_fixed_camera_params": True,
}


def default_odm_cli_args() -> list[str]:
    return ["--fast-orthophoto"]


def write_odm_options(odm_input: Path, *, cameras_path: Path) -> None:
    payload = {
        **DEFAULT_ODM_OPTIONS,
        "cameras": str(cameras_path),
        "cli_args_if_empty": default_odm_cli_args(),
        "note": (
            "cameras.json is exported from intrinsicK.csv and bound by run_odm.sh via "
            "--cameras --use-fixed-camera-params; its key matches ODM's detected camera id "
            "(see Camera.camera_id) so the override binds instead of ODM self-calibrating"
        ),
    }
    (odm_input / "odm_options.json").write_text(json.dumps(payload, indent=2) + "\n")
