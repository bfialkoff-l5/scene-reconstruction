from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import asin, atan2, degrees
from pathlib import Path

import numpy as np
import pandas as pd

from scene_recon.geometry.extrinsics import CameraPose
from scene_recon.record import Record
from scene_recon.scoring_cache import file_fingerprint
from scene_recon.video import frame_filename

# Identifies the rotation-prior sidecar to our patched OpenSfM (read via the
# SCENE_RECON_ROTATION_PRIORS env var that run_odm.sh points at this file).
ROTATION_PRIORS_FILENAME = "rotation_priors.json"


def _opk_degrees_from_cam_to_enu(r_cam_to_enu: np.ndarray) -> tuple[float, float, float]:
    """OpenSfM omega/phi/kappa (degrees) from a camera->ENU rotation matrix.

    Mirrors opensfm.geometry.opk_from_rotation, which takes the *world->camera* matrix
    (= r_cam_to_enu.T, so r_cam_to_enu.T.T @ Rc = r_cam_to_enu @ Rc below). This is the
    exact inverse of OpenSfM's rotation_from_opk, so feeding these angles back reproduces
    r_cam_to_enu.T bit-for-bit -- a lossless 3-DOF encoding of the camera attitude, unlike
    the YPR -> compute_opk path ODM applies to geo.txt.
    """
    rc = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    r = r_cam_to_enu.dot(rc)
    omega = atan2(-r[1, 2], r[2, 2])
    phi = asin(float(np.clip(r[0, 2], -1.0, 1.0)))
    kappa = atan2(-r[0, 1], r[0, 0])
    return degrees(omega), degrees(phi), degrees(kappa)


def write_rotation_priors(selected: pd.DataFrame, output_path: Path) -> None:
    """Write exact per-image IMU camera attitude as OpenSfM omega/phi/kappa (degrees).

    This is the orientation channel that actually fixes the georeferencing variance: our
    patched OpenSfM reads this file in extract_metadata and feeds each angle triplet to
    bundle adjustment as a full 3-DOF absolute rotation prior, pinning the otherwise
    unconstrained rotation and forbidding the catastrophic global-flip basin. The angles
    round-trip R_cam_to_enu losslessly (see _opk_degrees_from_cam_to_enu), so unlike
    geo.txt YPR they survive into BA unmangled. See docs/FINDINGS.md.
    """
    if selected.empty:
        raise ValueError("cannot write rotation priors from empty selection")

    priors: dict[str, dict[str, float]] = {}
    for frame_number, row in selected.sort_index().iterrows():
        pose = CameraPose.from_row(row)
        omega, phi, kappa = _opk_degrees_from_cam_to_enu(pose.R_cam_to_enu())
        priors[frame_filename(int(frame_number))] = {
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
        }

    output_path.write_text(json.dumps(priors, indent=2) + "\n")


def write_geo_txt(selected: pd.DataFrame, output_path: Path) -> None:
    if selected.empty:
        raise ValueError("cannot write geo.txt from empty selection")

    utm_zone = str(selected.iloc[0]["utm_zone"])
    lines = [f"WGS84 UTM {utm_zone}"]

    # NOTE: position only. Per-image orientation is intentionally NOT emitted here:
    # ODM's geo.txt YPR -> compute_opk conversion is lossy for our strapdown/oblique
    # camera (it assumes a gimballed near-nadir camera) and corrupts matching. Orientation
    # instead flows losslessly through rotation_priors.json (see write_rotation_priors),
    # which our patched OpenSfM consumes as a true bundle-adjustment rotation prior.
    for frame_number, row in selected.sort_index().iterrows():
        lines.append(
            f"{frame_filename(int(frame_number))} "
            f"{row['easting']} {row['northing']} {row['altamsl']}"
        )

    output_path.write_text("\n".join(lines) + "\n")


@dataclass
class BuildManifest:
    record_path: str
    slug: str
    stream_id: str
    video: str
    poses_path: str
    pose_source: str
    poses_sha256: str
    intrinsics: str
    run_ts: str
    run_dir: str
    n_candidates: int
    n_selected: int
    selected_frame_numbers: list[int]
    selection_policy: str
    selection_constants: dict

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def write_build_manifest(
    record: Record,
    candidates: pd.DataFrame,
    run_dir_path: Path,
    run_ts: str,
    selection_policy: str,
    selection_constants: dict,
) -> None:
    selected = candidates[candidates["selected"]]
    poses_fingerprint = file_fingerprint(record.poses_path)
    manifest = BuildManifest(
        record_path=str(record.path),
        slug=record.slug,
        stream_id=record.stream_id,
        video=str(record.video),
        poses_path=str(record.poses_path),
        pose_source=record.pose_source,
        poses_sha256=poses_fingerprint["sha256"],
        intrinsics=str(record.intrinsics),
        run_ts=run_ts,
        run_dir=f"runs/{run_ts}",
        n_candidates=len(candidates),
        n_selected=len(selected),
        selected_frame_numbers=[int(n) for n in selected.index.tolist()],
        selection_policy=selection_policy,
        selection_constants=selection_constants,
    )
    manifest.write(run_dir_path / "build.json")
