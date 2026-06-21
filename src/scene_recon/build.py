from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scene_recon.camera import Camera
from scene_recon.candidates import selected_candidates
from scene_recon.export import write_build_manifest, write_geo_txt
from scene_recon.geometry.footprint import compute_footprints
from scene_recon.geometry.terrain import TerrainModel
from scene_recon.intrinsics import write_cameras_json
from scene_recon.odm import MATCHER_REACH_M, recommend_matcher_neighbors, write_odm_options
from scene_recon.selection import (
    DEFAULT_SELECTION_PARAMS,
    SELECTION_POLICY,
    GroundGrid,
    SelectionParams,
    compute_view_counts,
    coverage_metrics,
    load_footprints,
    save_footprints,
    select_keyframes,
)
from scene_recon.selection.parallax import approx_cell_ground_z, mission_cell_ground_z
from scene_recon.paths import run_dir, slug_dir, stamp_run_ts
from scene_recon.record import Record
from scene_recon.scoring import QUALITY_WEIGHT_FEATURES, QUALITY_WEIGHT_SHARPNESS
from scene_recon.scoring_cache import load_or_score_record, load_scored_candidates
from scene_recon.selection_health import SelectionFailed, assess_selection
from scene_recon.selection_report import write_selection_report
from scene_recon.video import extract_frames

log = logging.getLogger(__name__)

FOOTPRINTS_FILENAME = "footprints.pkl"


def _selection_constants(params: SelectionParams) -> dict:
    return {
        **params.as_constants(),
        "quality_weight_features": QUALITY_WEIGHT_FEATURES,
        "quality_weight_sharpness": QUALITY_WEIGHT_SHARPNESS,
    }


def _mission_bbox(candidates: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(candidates["easting"].min()),
        float(candidates["northing"].min()),
        float(candidates["easting"].max()),
        float(candidates["northing"].max()),
    )


def _load_or_compute_footprints(
    record: Record,
    candidates: pd.DataFrame,
    grid: GroundGrid,
    params: SelectionParams,
    cache_path: Path,
    *,
    reuse: bool,
    terrain: TerrainModel | None = None,
):
    if reuse and cache_path.is_file():
        log.info("loading footprint cache %s", cache_path)
        return load_footprints(cache_path)

    if params.terrain_gpkg is None:
        raise ValueError(
            "policy frustum_view_count_dtm requires --terrain-gpkg (DTM GeoPackage)"
        )

    log.info("ray-marching footprints against %s", params.terrain_gpkg)
    camera = Camera.from_record(record)
    if terrain is None:
        terrain = TerrainModel.from_gpkg(
            params.terrain_gpkg,
            bbox_utm=_mission_bbox(candidates),
            margin_m=params.terrain_margin_m,
            datum_offset_m=params.datum_offset_m,
        )
    footprints = compute_footprints(
        candidates,
        camera,
        terrain,
        grid,
        ray_grid=params.ray_grid,
        max_range_m=params.max_range_m,
        step_m=params.ray_step_m,
        min_valid_ray_frac=params.min_valid_ray_frac,
    )
    save_footprints(cache_path, footprints)
    return footprints


def export_run(
    record: Record,
    candidates,
    run_dir_path: Path,
    run_ts: str,
    params: SelectionParams,
) -> None:
    selected = selected_candidates(candidates)
    log.info("selected %d / %d keyframes", len(selected), len(candidates))

    constants = _selection_constants(params)
    odm_input = run_dir_path / "odm_input"
    images_dir = odm_input / "images"
    frame_numbers = [int(n) for n in selected.index.tolist()]

    extract_frames(record, frame_numbers, images_dir)
    write_geo_txt(selected, odm_input / "geo.txt")
    cameras_path = odm_input / "cameras.json"
    write_cameras_json(record, cameras_path)
    matcher_neighbors = recommend_matcher_neighbors(
        selected["easting"], selected["northing"]
    )
    log.info(
        "odm matcher-neighbors auto = %d (holds ~%.0f m baseline reach for %d keyframes)",
        matcher_neighbors,
        MATCHER_REACH_M,
        len(selected),
    )
    write_odm_options(
        odm_input, cameras_path=cameras_path, matcher_neighbors=matcher_neighbors
    )
    write_build_manifest(
        record,
        candidates,
        run_dir_path,
        run_ts=run_ts,
        selection_policy=SELECTION_POLICY,
        selection_constants=constants,
    )


def build_record(
    record_path: str | Path,
    *,
    select_only: bool = False,
    rescore: bool = False,
    params: SelectionParams | None = None,
) -> Path:
    record = Record.from_path(record_path)
    slug_path = slug_dir(record.data_root, record.slug)
    slug_path.mkdir(parents=True, exist_ok=True)

    selection = params or DEFAULT_SELECTION_PARAMS

    # Fail fast (before the expensive scoring pass) on missing/unreadable inputs.
    will_reuse_footprints = (
        select_only and not rescore and (slug_path / FOOTPRINTS_FILENAME).is_file()
    )
    if not will_reuse_footprints:
        if selection.terrain_gpkg is None:
            raise ValueError(
                f"policy {SELECTION_POLICY} requires --terrain-gpkg (DTM GeoPackage) "
                "to compute footprints"
            )
        if not Path(selection.terrain_gpkg).is_file():
            raise ValueError(f"--terrain-gpkg not found: {selection.terrain_gpkg}")

    if select_only:
        log.info("record=%s loading scored cache", record.slug)
        candidates = load_scored_candidates(slug_path)
    else:
        log.info("record=%s scoring candidates", record.slug)
        candidates = load_or_score_record(record, slug_path, rescore=rescore)
        scored = candidates["quality_score"].astype(float)
        log.info(
            "scoring complete  quality_score p50=%.3f p10=%.3f",
            scored.median(),
            scored.quantile(0.1),
        )

    grid = GroundGrid.from_poses(
        candidates, bin_size_m=selection.bin_size_m, margin_m=selection.terrain_margin_m
    )
    terrain = None
    if selection.terrain_gpkg is not None:
        terrain = TerrainModel.from_gpkg(
            selection.terrain_gpkg,
            bbox_utm=_mission_bbox(candidates),
            margin_m=selection.terrain_margin_m,
            datum_offset_m=selection.datum_offset_m,
        )
    footprints = _load_or_compute_footprints(
        record,
        candidates,
        grid,
        selection,
        slug_path / FOOTPRINTS_FILENAME,
        reuse=select_only and not rescore,
        terrain=terrain,
    )
    mission_cells = grid.mission_cells(footprints.values())
    cell_ground_z = (
        mission_cell_ground_z(grid, mission_cells, candidates, terrain)
        if terrain is not None
        else approx_cell_ground_z(grid, mission_cells, candidates)
    )

    if terrain is not None:
        # Per-frame AGL = camera altitude - DTM under the camera. Feeds the selector's
        # GSD-consistency floor and is a useful diagnostic column in the audit CSV.
        candidates["agl_m"] = candidates["altamsl"].to_numpy(float) - terrain.elevation_at(
            candidates["easting"].to_numpy(float), candidates["northing"].to_numpy(float)
        )

    candidates = select_keyframes(candidates, footprints, grid, selection)
    view_counts = compute_view_counts(candidates, footprints)
    coverage = coverage_metrics(
        view_counts,
        selection.target_views_per_cell,
        mission_cells=mission_cells,
        bin_size_m=selection.bin_size_m,
    )
    health = assess_selection(
        candidates,
        selection,
        view_counts=view_counts,
        mission_cells=mission_cells,
        footprints=footprints,
        grid=grid,
        cell_ground_z=cell_ground_z,
    )

    run_ts = stamp_run_ts()
    run_path = run_dir(slug_path, run_ts)
    run_path.mkdir(parents=True, exist_ok=True)

    constants = _selection_constants(selection)
    write_selection_report(
        candidates,
        run_path,
        constants,
        health=health,
        view_counts=view_counts,
        coverage=coverage,
        grid=grid,
        footprints=footprints,
        cell_ground_z=cell_ground_z,
    )

    if not health.passed:
        raise SelectionFailed(health)

    export_run(record, candidates, run_path, run_ts, selection)
    return run_path
