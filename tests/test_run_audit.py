from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scene_recon.run_audit import build_run_audit, write_run_audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_selection_audit(run_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "FrameNumber": [0, 10, 20, 30],
            "selected": [True, True, True, False],
            "easting": [100.0, 101.0, 101.0, 110.0],
            "northing": [200.0, 200.0, 200.05, 210.0],
            "altamsl": [150.0, 151.0, 152.0, 153.0],
            "quality_score": [0.9, 0.5, 0.4, 0.8],
        }
    )
    df.to_csv(run_dir / "selection_audit.csv", index=False)


def test_build_run_audit_without_odm_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "odm-results" / "slug" / "runs" / "20260617000000"
    run_dir.mkdir(parents=True)
    _write_selection_audit(run_dir)
    _write_json(
        run_dir / "build.json",
        {
            "n_candidates": 4,
            "n_selected": 3,
            "selected_frame_numbers": [0, 10, 20],
        },
    )

    audit = build_run_audit(run_dir)

    assert audit["selection"]["available"]
    assert audit["selection"]["n_candidates"] == 4
    assert audit["selection"]["n_selected"] == 3
    assert audit["selection"]["near_repeated_adjacent_positions"]["count"] == 1
    assert not audit["opensfm_features"]["available"]
    assert not audit["opensfm_reconstruction"]["available"]
    assert not audit["opensfm_tracks"]["available"]


def test_build_run_audit_with_opensfm_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "odm-results" / "slug" / "runs" / "20260617000000"
    opensfm = run_dir / "odm_input" / "opensfm"
    run_dir.mkdir(parents=True)
    _write_selection_audit(run_dir)

    _write_json(
        opensfm / "reports" / "features.json",
        {
            "image_reports": [
                {"image": "000000.png", "num_features": 200},
                {"image": "000010.png", "num_features": 0},
                {"image": "000020.png", "num_features": 50},
                {"image": "000030.png", "num_features": 500},
            ]
        },
    )
    _write_json(
        opensfm / "reconstruction.json",
        [
            {
                "shots": {
                    "000000.png": {},
                    "000020.png": {},
                },
                "points": {"p1": {}},
            }
        ],
    )
    _write_json(
        opensfm / "reports" / "reconstruction.json",
        {
            "not_reconstructed_images": ["000010.png"],
            "num_candidate_image_pairs": 3,
        },
    )
    _write_json(
        opensfm / "reports" / "tracks.json",
        {
            "num_images": 3,
            "num_tracks": 7,
            "view_graph": [
                ["000000.png", "000020.png", 120],
                ["000010.png", "000020.png", 1],
            ],
        },
    )
    _write_json(
        opensfm / "stats" / "stats.json",
        {
            "reconstruction_statistics": {
                "reprojection_error_pixels": 1.25,
                "average_track_length": 3.5,
                "reconstructed_points_count": 7,
                "observations_count": 21,
            },
            "gps_errors": {"average_error": 4.2},
            "camera_errors": {"camera": {"optimized_values": {"focal": 1.1}}},
        },
    )
    _write_json(
        opensfm / "camera_models.json",
        {"camera": {"projection_type": "brown", "focal_x": 1.1}},
    )

    audit = build_run_audit(run_dir)

    assert audit["opensfm_features"]["available"]
    assert audit["opensfm_features"]["zero_feature_images"] == ["000010.png"]
    assert audit["opensfm_features"]["low_feature_images"] == ["000020.png"]
    assert audit["opensfm_reconstruction"]["n_reconstructed_shots"] == 2
    assert audit["opensfm_reconstruction"]["not_reconstructed_images"] == ["000010.png"]
    assert "000010.png" in audit["opensfm_tracks"]["weak_graph_nodes"]
    assert audit["opensfm_stats"]["reprojection_error_pixels"] == 1.25
    assert audit["opensfm_stats"]["gps_errors"]["average_error"] == 4.2
    assert audit["camera"]["opensfm_camera_models_available"]


def test_write_run_audit_writes_json_and_bad_images_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "odm-results" / "slug" / "runs" / "20260617000000"
    run_dir.mkdir(parents=True)
    _write_selection_audit(run_dir)

    output = write_run_audit(run_dir)

    assert output == run_dir / "run_audit.json"
    assert output.is_file()
    assert (run_dir / "run_audit_bad_images.csv").is_file()
