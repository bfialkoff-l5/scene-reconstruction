from __future__ import annotations

import json
from pathlib import Path

import click

from scene_recon.build import build_record
from scene_recon.frame_select import DEFAULT_SELECTION_PARAMS, SelectionParams, select_keyframes
from scene_recon.paths import resolve_run, slug_dir
from scene_recon.record import Record
from scene_recon.scoring_cache import load_scored_candidates
from scene_recon.selection_health import assess_selection
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


def _params_from_constants(constants: dict) -> SelectionParams:
    defaults = DEFAULT_SELECTION_PARAMS
    return SelectionParams(
        bin_size_m=constants.get("bin_size_m", defaults.bin_size_m),
        min_altitude_m=constants.get("min_altitude_m", defaults.min_altitude_m),
        min_translation_m=constants.get("min_translation_m", defaults.min_translation_m),
        min_rotation_deg=constants.get("min_rotation_deg", defaults.min_rotation_deg),
        max_frame_gap=constants.get("max_frame_gap", defaults.max_frame_gap),
        cluster_radius_m=constants.get("cluster_radius_m", defaults.cluster_radius_m),
        max_per_cluster=constants.get("max_per_cluster", defaults.max_per_cluster),
        coverage_warn_m=constants.get("coverage_warn_m", defaults.coverage_warn_m),
        max_keyframes=constants.get("max_keyframes", defaults.max_keyframes),
    )


def _selection_params_from_cli(
    bin_size_m: float | None,
    min_translation_m: float | None,
    min_rotation_deg: float | None,
    max_frame_gap: int | None,
    max_per_cluster: int | None,
    max_keyframes: int | None,
) -> SelectionParams:
    defaults = DEFAULT_SELECTION_PARAMS
    return SelectionParams(
        bin_size_m=bin_size_m if bin_size_m is not None else defaults.bin_size_m,
        min_translation_m=(
            min_translation_m if min_translation_m is not None else defaults.min_translation_m
        ),
        min_rotation_deg=min_rotation_deg if min_rotation_deg is not None else defaults.min_rotation_deg,
        max_frame_gap=max_frame_gap if max_frame_gap is not None else defaults.max_frame_gap,
        max_per_cluster=max_per_cluster if max_per_cluster is not None else defaults.max_per_cluster,
        max_keyframes=max_keyframes if max_keyframes is not None else defaults.max_keyframes,
    )


def _selection_options(f):
    f = click.option("--bin-size-m", type=float, default=None, help="Spatial bin size in meters")(f)
    f = click.option(
        "--min-translation-m", type=float, default=None, help="Min translation in meters"
    )(f)
    f = click.option("--min-rotation-deg", type=float, default=None, help="Min rotation in degrees")(
        f
    )
    f = click.option(
        "--max-frame-gap", type=int, default=None, help="Max pose rows between selected frames"
    )(f)
    f = click.option(
        "--max-per-cluster", type=int, default=None, help="Max selected frames per spatial cluster"
    )(f)
    f = click.option("--max-keyframes", type=int, default=None, help="Cap on selected frames")(f)
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
    min_translation_m: float | None,
    min_rotation_deg: float | None,
    max_frame_gap: int | None,
    max_per_cluster: int | None,
    max_keyframes: int | None,
) -> None:
    """Score (if needed), select keyframes, and export ODM input."""
    from scene_recon.selection_health import SelectionFailed

    _configure_logging()
    path = _resolve_record_path(record_path, _data_root_from_env(data_root))
    params = _selection_params_from_cli(
        bin_size_m,
        min_translation_m,
        min_rotation_deg,
        max_frame_gap,
        max_per_cluster,
        max_keyframes,
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
def report_cmd(record_path: str, run_ts: str | None, data_root: Path | None) -> None:
    """Regenerate selection report for a run (re-selects using build.json params)."""
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
    params = _params_from_constants(constants)

    candidates = load_scored_candidates(slug_path)
    candidates = select_keyframes(candidates, params)
    health = assess_selection(candidates, params)
    write_selection_report(candidates, resolved, constants, health=health)
    click.echo(resolved / "selection_report")


if __name__ == "__main__":
    main()
