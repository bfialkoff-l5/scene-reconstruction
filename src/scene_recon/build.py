from __future__ import annotations

import logging
from pathlib import Path

from scene_recon.candidates import selected_candidates
from scene_recon.export import write_build_manifest, write_geo_txt
from scene_recon.intrinsics import write_cameras_json
from scene_recon.odm import write_odm_options
from scene_recon.selection import (
    DEFAULT_SELECTION_PARAMS,
    SELECTION_POLICY,
    SelectionParams,
    compute_view_counts,
    coverage_metrics,
    select_keyframes,
)
from scene_recon.paths import run_dir, slug_dir, stamp_run_ts
from scene_recon.record import Record
from scene_recon.scoring import QUALITY_WEIGHT_FEATURES, QUALITY_WEIGHT_SHARPNESS
from scene_recon.scoring_cache import load_or_score_record, load_scored_candidates
from scene_recon.selection_health import SelectionFailed, assess_selection
from scene_recon.selection_report import write_selection_report
from scene_recon.video import extract_frames

log = logging.getLogger(__name__)


def _selection_constants(params: SelectionParams) -> dict:
    return {
        **params.as_constants(),
        "quality_weight_features": QUALITY_WEIGHT_FEATURES,
        "quality_weight_sharpness": QUALITY_WEIGHT_SHARPNESS,
    }


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
    write_odm_options(odm_input, cameras_path=cameras_path)
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

    candidates = select_keyframes(candidates, selection)
    view_counts = compute_view_counts(candidates, selection)
    coverage = coverage_metrics(view_counts, selection.target_views_per_cell)
    health = assess_selection(candidates, selection, view_counts=view_counts)

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
    )

    if not health.passed:
        raise SelectionFailed(health)

    export_run(record, candidates, run_path, run_ts, selection)
    return run_path
