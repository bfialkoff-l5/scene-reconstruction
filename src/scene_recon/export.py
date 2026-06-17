from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from scene_recon.record import Record
from scene_recon.video import frame_filename


def write_geo_txt(selected: pd.DataFrame, output_path: Path) -> None:
    if selected.empty:
        raise ValueError("cannot write geo.txt from empty selection")

    utm_zone = str(selected.iloc[0]["utm_zone"])
    lines = [f"WGS84 UTM {utm_zone}"]

    for frame_number, row in selected.sort_index().iterrows():
        lines.append(
            f"{frame_filename(int(frame_number))} {row['easting']} {row['northing']} {row['altamsl']}"
        )

    output_path.write_text("\n".join(lines) + "\n")


@dataclass
class BuildManifest:
    record_path: str
    slug: str
    stream_id: str
    video: str
    poses_path: str
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
    manifest = BuildManifest(
        record_path=str(record.path),
        slug=record.slug,
        stream_id=record.stream_id,
        video=str(record.video),
        poses_path=str(record.poses_path),
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
