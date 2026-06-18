from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from scene_recon.build import build_record
from scene_recon.selection import (
    DEFAULT_SELECTION_PARAMS,
    SelectionParams,
    compute_view_counts,
    coverage_metrics,
    params_from_constants,
    select_keyframes,
)
from scene_recon.paths import resolve_run, slug_dir
from scene_recon.record import Record
from scene_recon.scoring_cache import load_scored_candidates
from scene_recon.selection_health import assess_selection
from scene_recon.selection.parallax import approx_cell_ground_z
from scene_recon.selection_report import write_selection_report


def _configure_logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )


def _resolve_record_path(path: str | Path, data_root: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    if data_root is None:
        raise click.ClickException(
            f"Relative record path {p!r} requires DATA_ROOT or --data-root"
        )
    return (data_root / "raw" / p).resolve()


def _data_root_from_env(data_root: Path | None) -> Path | None:
    import os

    if data_root is not None:
        return data_root
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    return None


def _selection_params_from_cli(
    bin_size_m: float | None,
    overlap_target: float | None,
    max_keyframes: int | None,
    target_views_per_cell: int | None,
    max_motion_gap_m: float | None,
    cluster_radius_m: float | None,
    terrain_gpkg: Path | None,
    ray_grid: tuple[int, int] | None,
    datum_offset_m: float | None,
) -> SelectionParams:
    defaults = DEFAULT_SELECTION_PARAMS
    return SelectionParams(
        bin_size_m=bin_size_m if bin_size_m is not None else defaults.bin_size_m,
        terrain_gpkg=terrain_gpkg if terrain_gpkg is not None else defaults.terrain_gpkg,
        ray_grid=ray_grid if ray_grid is not None else defaults.ray_grid,
        datum_offset_m=datum_offset_m if datum_offset_m is not None else defaults.datum_offset_m,
        overlap_jaccard_target=(
            overlap_target if overlap_target is not None else defaults.overlap_jaccard_target
        ),
        max_keyframes=max_keyframes if max_keyframes is not None else defaults.max_keyframes,
        target_views_per_cell=(
            target_views_per_cell
            if target_views_per_cell is not None
            else defaults.target_views_per_cell
        ),
        max_motion_gap_m=max_motion_gap_m if max_motion_gap_m is not None else defaults.max_motion_gap_m,
        cluster_radius_m=cluster_radius_m if cluster_radius_m is not None else defaults.cluster_radius_m,
    )


def _selection_options(f):
    f = click.option(
        "--terrain-gpkg",
        type=click.Path(path_type=Path),
        default=None,
        help="DTM GeoPackage for frustum ground intersection (required for build)",
    )(f)
    f = click.option(
        "--ray-grid",
        type=(int, int),
        default=None,
        help="Footprint ray sampling grid, e.g. 48 27",
    )(f)
    f = click.option(
        "--datum-offset-m",
        type=float,
        default=None,
        help="Added to DTM elevations to align with pose altamsl datum",
    )(f)
    f = click.option("--bin-size-m", type=float, default=None, help="Spatial bin size in meters")(f)
    f = click.option(
        "--overlap-target",
        type=float,
        default=None,
        help="Stage-1 spacing: keep next frame when footprint Jaccard drops to <= this (0.0–1.0)",
    )(f)
    f = click.option("--max-keyframes", type=int, default=None, help="Hard cap on selected frames")(f)
    f = click.option(
        "--target-views-per-cell", type=int, default=None, help="Views per ground cell target (reporting)"
    )(f)
    f = click.option(
        "--max-motion-gap-m", type=float, default=None, help="Motion gap warning threshold (warn only)"
    )(f)
    f = click.option(
        "--cluster-radius-m", type=float, default=None, help="Spatial cluster radius in meters (reporting)"
    )(f)
    return f


@click.group()
def main() -> None:
    """Scene reconstruction CLI."""


@main.command("build")
@click.argument("record_path")
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option("--select-only", is_flag=True, help="Skip scoring; reuse slug-level cache")
@click.option("--rescore", is_flag=True, help="Force re-decode and re-score all frames")
@_selection_options
def build_cmd(
    record_path: str,
    data_root: Path | None,
    select_only: bool,
    rescore: bool,
    bin_size_m: float | None,
    overlap_target: float | None,
    max_keyframes: int | None,
    target_views_per_cell: int | None,
    max_motion_gap_m: float | None,
    cluster_radius_m: float | None,
    terrain_gpkg: Path | None,
    ray_grid: tuple[int, int] | None,
    datum_offset_m: float | None,
) -> None:
    """Score (if needed), select keyframes, and export ODM input."""
    from scene_recon.selection_health import SelectionFailed

    _configure_logging()
    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    params = _selection_params_from_cli(
        bin_size_m,
        overlap_target,
        max_keyframes,
        target_views_per_cell,
        max_motion_gap_m,
        cluster_radius_m,
        terrain_gpkg,
        ray_grid,
        datum_offset_m,
    )
    try:
        run_path = build_record(
            path,
            select_only=select_only,
            rescore=rescore,
            params=params,
        )
    except SelectionFailed as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(run_path)


@main.command("resolve-run")
@click.argument("record_path")
@click.option("--run", "run_ts", default=None, help="Pin run timestamp YYYYMMDDHHMMSS")
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--odm-input",
    "odm_input_only",
    is_flag=True,
    help="Print odm_input/ path instead of run dir",
)
def resolve_run_cmd(
    record_path: str,
    run_ts: str | None,
    data_root: Path | None,
    odm_input_only: bool,
) -> None:
    """Print latest (or pinned) run directory for a record."""
    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    record = Record.from_path(path)
    slug_path = slug_dir(record.data_root, record.slug)
    resolved = resolve_run(slug_path, run_ts)
    if odm_input_only:
        click.echo(resolved / "odm_input")
    else:
        click.echo(resolved)


@main.command("prepare-odm")
@click.argument("record_path")
@click.option("--run", "run_ts", default=None, help="Run timestamp YYYYMMDDHHMMSS")
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
def prepare_odm_cmd(record_path: str, run_ts: str | None, data_root: Path | None) -> None:
    """Write cameras.json and odm_options.json for an existing run."""
    from scene_recon.intrinsics import write_cameras_json
    from scene_recon.odm import write_odm_options

    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    record = Record.from_path(path)
    slug_path = slug_dir(record.data_root, record.slug)
    resolved = resolve_run(slug_path, run_ts)
    odm_input = resolved / "odm_input"
    if not odm_input.is_dir():
        raise click.ClickException(f"missing {odm_input}")

    cameras_path = odm_input / "cameras.json"
    write_cameras_json(record, cameras_path)
    write_odm_options(odm_input, cameras_path=cameras_path)
    click.echo(cameras_path)


@main.command("report")
@click.argument("record_path")
@click.option("--run", "run_ts", default=None, help="Run timestamp YYYYMMDDHHMMSS")
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run select_keyframes from scored cache instead of using saved audit",
)
def report_cmd(
    record_path: str,
    run_ts: str | None,
    data_root: Path | None,
    force: bool,
) -> None:
    """Regenerate selection report for a run."""
    _configure_logging()
    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    record = Record.from_path(path)
    slug_path = slug_dir(record.data_root, record.slug)
    resolved = resolve_run(slug_path, run_ts)

    build_json = resolved / "build.json"
    if not build_json.is_file():
        raise click.ClickException(f"missing {build_json}")

    manifest = json.loads(build_json.read_text())
    constants = manifest["selection_constants"]
    params = params_from_constants(constants)

    from scene_recon.build import FOOTPRINTS_FILENAME
    from scene_recon.selection import GroundGrid, load_footprints

    footprints_path = slug_path / FOOTPRINTS_FILENAME
    if not footprints_path.is_file():
        raise click.ClickException(
            f"missing {footprints_path}; re-run build to regenerate the footprint cache"
        )
    footprints = load_footprints(footprints_path)

    if force:
        candidates = load_scored_candidates(slug_path)
        grid = GroundGrid.from_poses(
            candidates, bin_size_m=params.bin_size_m, margin_m=params.terrain_margin_m
        )
        candidates = select_keyframes(candidates, footprints, grid, params)
    else:
        audit_path = resolved / "selection_audit.csv"
        if not audit_path.is_file():
            raise click.ClickException(
                f"missing {audit_path}; use --force to re-select from scored cache"
            )
        candidates = pd.read_csv(audit_path, index_col="FrameNumber")
        grid = GroundGrid.from_poses(
            candidates, bin_size_m=params.bin_size_m, margin_m=params.terrain_margin_m
        )

    mission_cells = grid.mission_cells(footprints.values())
    cell_ground_z = approx_cell_ground_z(grid, mission_cells, candidates)
    view_counts = compute_view_counts(candidates, footprints)
    coverage = coverage_metrics(
        view_counts,
        params.target_views_per_cell,
        mission_cells=mission_cells,
        bin_size_m=params.bin_size_m,
    )
    health = assess_selection(
        candidates,
        params,
        view_counts=view_counts,
        mission_cells=mission_cells,
        footprints=footprints,
        grid=grid,
        cell_ground_z=cell_ground_z,
    )
    write_selection_report(
        candidates,
        resolved,
        constants,
        health=health,
        view_counts=view_counts,
        coverage=coverage,
        grid=grid,
        footprints=footprints,
        cell_ground_z=cell_ground_z,
    )
    click.echo(resolved / "selection_report")


@main.command("audit-run")
@click.argument("record_path")
@click.option("--run", "run_ts", default=None, help="Run timestamp YYYYMMDDHHMMSS")
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
def audit_run_cmd(record_path: str, run_ts: str | None, data_root: Path | None) -> None:
    """Audit selection and ODM/OpenSfM outputs for an existing run."""
    from scene_recon.run_audit import write_run_audit

    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    record = Record.from_path(path)
    slug_path = slug_dir(record.data_root, record.slug)
    resolved = resolve_run(slug_path, run_ts)
    click.echo(write_run_audit(resolved))


if __name__ == "__main__":
    main()
