from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LOW_FEATURE_THRESHOLD = 100
WEAK_GRAPH_DEGREE_THRESHOLD = 5
WEAK_GRAPH_STRENGTH_THRESHOLD = 100
NEAR_REPEATED_POSITION_M = 0.1
LOW_BASELINE_M = 1.0


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}

    s = pd.Series(values, dtype=float)
    return {
        "min": round(float(s.min()), 6),
        "p10": round(float(s.quantile(0.10)), 6),
        "p25": round(float(s.quantile(0.25)), 6),
        "p50": round(float(s.quantile(0.50)), 6),
        "p75": round(float(s.quantile(0.75)), 6),
        "p90": round(float(s.quantile(0.90)), 6),
        "max": round(float(s.max()), 6),
    }


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _frame_name(frame_number: int) -> str:
    return f"{frame_number:06d}.png"


def _selected_from_audit(selection_audit: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not selection_audit.is_file():
        return None, []

    df = pd.read_csv(selection_audit)
    if "selected" not in df.columns:
        return df, []
    selected = df[_bool_series(df["selected"])].copy()
    if "FrameNumber" not in selected.columns:
        return df, []
    images = [_frame_name(int(n)) for n in selected["FrameNumber"].tolist()]
    return df, images


def _selected_from_build(build_json: dict[str, Any] | None) -> list[str]:
    if not build_json:
        return []
    frames = build_json.get("selected_frame_numbers") or []
    return [_frame_name(int(n)) for n in frames]


def _selection_metrics(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {"available": False}

    metrics: dict[str, Any] = {
        "available": True,
        "n_candidates": int(len(df)),
    }

    if "selected" not in df.columns:
        metrics["n_selected"] = None
        return metrics

    selected = df[_bool_series(df["selected"])].copy()
    metrics["n_selected"] = int(len(selected))

    if selected.empty:
        return metrics

    if "FrameNumber" in selected.columns:
        frames = [int(n) for n in selected["FrameNumber"].tolist()]
        metrics["frame_range"] = {"min": min(frames), "max": max(frames)}
        frame_gaps = [float(b - a) for a, b in zip(frames, frames[1:])]
        metrics["adjacent_frame_gap"] = _quantiles(frame_gaps)

    if "altamsl" in selected.columns:
        altitudes = [_safe_float(v) for v in selected["altamsl"].tolist()]
        altitudes = [v for v in altitudes if v is not None]
        metrics["altitude_m"] = _quantiles(altitudes)

    if {"easting", "northing"}.issubset(selected.columns):
        ordered = selected
        if "FrameNumber" in ordered.columns:
            ordered = ordered.sort_values("FrameNumber")
        easting = ordered["easting"].astype(float).tolist()
        northing = ordered["northing"].astype(float).tolist()
        baselines = [
            math.hypot(e2 - e1, n2 - n1)
            for e1, n1, e2, n2 in zip(easting, northing, easting[1:], northing[1:])
        ]
        metrics["adjacent_baseline_m"] = _quantiles(baselines)
        metrics["near_repeated_adjacent_positions"] = {
            "threshold_m": NEAR_REPEATED_POSITION_M,
            "count": int(sum(d <= NEAR_REPEATED_POSITION_M for d in baselines)),
        }
        metrics["low_adjacent_baseline"] = {
            "threshold_m": LOW_BASELINE_M,
            "count": int(sum(d < LOW_BASELINE_M for d in baselines)),
        }

    if "quality_score" in selected.columns:
        quality = [_safe_float(v) for v in selected["quality_score"].tolist()]
        quality = [v for v in quality if v is not None]
        metrics["selected_quality_score"] = _quantiles(quality)

    return metrics


def _feature_report(opensfm: Path, selected_images: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    reports = opensfm / "reports" / "features.json"
    data = _read_json(reports)
    if not data:
        return {"available": False}, {}

    image_reports = data.get("image_reports") or []
    features = {
        str(row["image"]): int(row.get("num_features") or 0)
        for row in image_reports
        if "image" in row
    }
    selected_counts = [
        count for image, count in features.items() if not selected_images or image in selected_images
    ]
    zero = sorted(image for image, count in features.items() if count == 0 and (not selected_images or image in selected_images))
    low = sorted(
        image
        for image, count in features.items()
        if 0 < count < LOW_FEATURE_THRESHOLD and (not selected_images or image in selected_images)
    )

    return (
        {
            "available": True,
            "n_images": len(features),
            "feature_count": _quantiles([float(c) for c in selected_counts]),
            "zero_feature_images": zero,
            "n_zero_feature_images": len(zero),
            "low_feature_threshold": LOW_FEATURE_THRESHOLD,
            "low_feature_images": low,
            "n_low_feature_images": len(low),
        },
        features,
    )


def _reconstruction_report(opensfm: Path, selected_images: set[str]) -> dict[str, Any]:
    reconstruction = _read_json(opensfm / "reconstruction.json")
    report = _read_json(opensfm / "reports" / "reconstruction.json") or {}
    if reconstruction is None and not report:
        return {"available": False}

    reconstructed: set[str] = set()
    if isinstance(reconstruction, list):
        for component in reconstruction:
            if isinstance(component, dict):
                reconstructed.update(str(name) for name in (component.get("shots") or {}).keys())

    reported_missing = set(str(name) for name in report.get("not_reconstructed_images") or [])
    if selected_images:
        not_reconstructed = sorted((selected_images - reconstructed) | reported_missing)
    else:
        not_reconstructed = sorted(reported_missing)

    return {
        "available": True,
        "n_reconstructed_shots": len(reconstructed),
        "n_selected_images": len(selected_images) if selected_images else None,
        "reconstructed_selected_ratio": (
            round(len(reconstructed & selected_images) / len(selected_images), 6)
            if selected_images
            else None
        ),
        "not_reconstructed_images": not_reconstructed,
        "n_not_reconstructed_images": len(not_reconstructed),
        "num_candidate_image_pairs": report.get("num_candidate_image_pairs"),
    }


def _tracks_report(opensfm: Path, selected_images: set[str]) -> dict[str, Any]:
    data = _read_json(opensfm / "reports" / "tracks.json")
    if not data:
        return {"available": False}

    degree: Counter[str] = Counter()
    strength: Counter[str] = Counter()
    weights: list[float] = []
    for edge in data.get("view_graph") or []:
        if len(edge) < 3:
            continue
        a, b, weight = str(edge[0]), str(edge[1]), int(edge[2])
        degree[a] += 1
        degree[b] += 1
        strength[a] += weight
        strength[b] += weight
        weights.append(float(weight))

    nodes = set(degree)
    scope = selected_images if selected_images else nodes
    degree_values = [float(degree[node]) for node in scope if node in degree]
    strength_values = [float(strength[node]) for node in scope if node in strength]
    weak = sorted(
        node
        for node in scope
        if degree[node] <= WEAK_GRAPH_DEGREE_THRESHOLD
        or strength[node] < WEAK_GRAPH_STRENGTH_THRESHOLD
    )

    return {
        "available": True,
        "n_images_in_tracks": data.get("num_images"),
        "n_tracks": data.get("num_tracks"),
        "n_view_graph_edges": len(data.get("view_graph") or []),
        "edge_weight": _quantiles(weights),
        "node_degree": _quantiles(degree_values),
        "node_strength": _quantiles(strength_values),
        "weak_graph_degree_threshold": WEAK_GRAPH_DEGREE_THRESHOLD,
        "weak_graph_strength_threshold": WEAK_GRAPH_STRENGTH_THRESHOLD,
        "weak_graph_nodes": weak,
        "n_weak_graph_nodes": len(weak),
    }


def _opensfm_stats(opensfm: Path) -> dict[str, Any]:
    data = _read_json(opensfm / "stats" / "stats.json")
    if not data:
        return {"available": False}

    reconstruction = data.get("reconstruction_statistics") or {}
    gps = data.get("gps_errors") or {}
    return {
        "available": True,
        "reprojection_error_pixels": reconstruction.get("reprojection_error_pixels"),
        "reprojection_error_normalized": reconstruction.get("reprojection_error_normalized"),
        "average_track_length": reconstruction.get("average_track_length"),
        "reconstructed_points_count": reconstruction.get("reconstructed_points_count"),
        "observations_count": reconstruction.get("observations_count"),
        "gps_errors": gps,
        "camera_errors": data.get("camera_errors") or {},
    }


def _camera_report(odm_input: Path) -> dict[str, Any]:
    opensfm_camera = _read_json(odm_input / "opensfm" / "camera_models.json")
    exported_camera = _read_json(odm_input / "cameras.json")
    return {
        "opensfm_camera_models_available": opensfm_camera is not None,
        "exported_cameras_available": exported_camera is not None,
        "opensfm_camera_models": opensfm_camera or {},
        "exported_cameras": exported_camera or {},
    }


def _artifact_presence(run_dir: Path) -> dict[str, bool]:
    odm_input = run_dir / "odm_input"
    return {
        "build_json": (run_dir / "build.json").is_file(),
        "selection_summary_json": (run_dir / "selection_summary.json").is_file(),
        "selection_audit_csv": (run_dir / "selection_audit.csv").is_file(),
        "geo_txt": (odm_input / "geo.txt").is_file(),
        "opensfm": (odm_input / "opensfm").is_dir(),
        "opensfm_reconstruction": (odm_input / "opensfm" / "reconstruction.json").is_file(),
        "opensfm_stats": (odm_input / "opensfm" / "stats" / "stats.json").is_file(),
        "orthophoto": (odm_input / "odm_orthophoto" / "odm_orthophoto.tif").is_file(),
        "overlap_raster": (odm_input / "opensfm" / "stats" / "overlap.tif").is_file(),
        "georeferenced_point_cloud": (
            odm_input / "odm_georeferencing" / "odm_georeferenced_model.laz"
        ).is_file(),
    }


def build_run_audit(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    odm_input = run_dir / "odm_input"
    opensfm = odm_input / "opensfm"

    build_json = _read_json(run_dir / "build.json")
    selection_summary = _read_json(run_dir / "selection_summary.json")
    selection_df, selected_from_audit = _selected_from_audit(run_dir / "selection_audit.csv")
    selected_images = set(selected_from_audit or _selected_from_build(build_json))

    features, feature_counts = _feature_report(opensfm, selected_images)
    reconstruction = _reconstruction_report(opensfm, selected_images)
    tracks = _tracks_report(opensfm, selected_images)

    bad_images = sorted(
        set(features.get("zero_feature_images") or [])
        | set(features.get("low_feature_images") or [])
        | set(reconstruction.get("not_reconstructed_images") or [])
        | set(tracks.get("weak_graph_nodes") or [])
    )

    return {
        "run_dir": str(run_dir),
        "artifact_presence": _artifact_presence(run_dir),
        "build": build_json or {},
        "selection_summary": selection_summary or {},
        "selection": _selection_metrics(selection_df),
        "opensfm_features": features,
        "opensfm_reconstruction": reconstruction,
        "opensfm_tracks": tracks,
        "opensfm_stats": _opensfm_stats(opensfm),
        "camera": _camera_report(odm_input),
        "bad_images": [
            {
                "image": image,
                "num_features": feature_counts.get(image),
                "zero_features": feature_counts.get(image) == 0,
                "low_features": (
                    feature_counts.get(image) is not None
                    and 0 < int(feature_counts[image]) < LOW_FEATURE_THRESHOLD
                ),
                "not_reconstructed": image
                in set(reconstruction.get("not_reconstructed_images") or []),
                "weak_graph_node": image in set(tracks.get("weak_graph_nodes") or []),
            }
            for image in bad_images
        ],
    }


def write_bad_images_csv(audit: dict[str, Any], output_path: Path) -> None:
    rows = audit.get("bad_images") or []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "num_features",
                "zero_features",
                "low_features",
                "not_reconstructed",
                "weak_graph_node",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_run_audit(run_dir: Path) -> Path:
    audit = build_run_audit(run_dir)
    output_path = run_dir / "run_audit.json"
    output_path.write_text(json.dumps(audit, indent=2) + "\n")
    write_bad_images_csv(audit, run_dir / "run_audit_bad_images.csv")
    return output_path
