from __future__ import annotations

import gzip
import pickle

import numpy as np

from scene_recon.matching.covisibility import CoVisEdge, CoVisGraph
from scene_recon.matching.scoreboard import pair_coverage


def _name(n: int) -> str:
    return f"{n:06d}.png"


def _write_matches(matches_dir, solid_pairs, *, inliers: int = 25) -> None:
    """Write OpenSfM-style matches/{img}_matches.pkl.gz for the given solid pairs.

    Each pair is stored once, under the lexicographically smaller image name, as an
    (inliers, 2) int array so the scoreboard counts it as a solid pair.
    """
    matches_dir.mkdir(parents=True, exist_ok=True)
    per_base: dict[str, dict[str, np.ndarray]] = {}
    for a, b in solid_pairs:
        na, nb = _name(a), _name(b)
        base, partner = (na, nb) if na < nb else (nb, na)
        per_base.setdefault(base, {})[partner] = np.zeros((inliers, 2), dtype=int)
    for base, m in per_base.items():
        with gzip.open(matches_dir / f"{base}_matches.pkl.gz", "wb") as fh:
            pickle.dump(m, fh)


def _edge(i: int, j: int) -> CoVisEdge:
    return CoVisEdge(i=i, j=j, iom=0.5, shared_cells=10, centroid_dist_m=10.0,
                     view_angle_deg=0.0)


def test_pair_coverage_overall_and_cross_track(tmp_path):
    # Dense along-track chain 0..20 so every frame is present and the scoreboard's *dense*
    # rank equals the frame number (a cross-track pair then has rank gap > 16).
    chain = [(i, i + 1) for i in range(20)]  # 20 along-track pairs, frames 0..20
    cross = [(0, 20), (1, 19)]  # gaps 20 and 18 > 16
    solid = chain + cross
    _write_matches(tmp_path / "matches", solid)

    # Predicted graph: the whole chain + 1 of the 2 cross-track pairs + 1 along-track pair
    # that is NOT solid (so efficiency < 1.0).
    edges = [_edge(i, i + 1) for i in range(20)] + [_edge(0, 20), _edge(4, 6)]
    covis = CoVisGraph(frames=list(range(21)), edges=edges)

    rep = pair_coverage(covis, tmp_path / "matches", min_inliers=20,
                        cross_track_rank_gap=16)

    assert rep.n_solid == 22
    assert rep.n_predicted == 22
    assert rep.n_hit == 21  # whole chain (20) + (0,20)
    assert rep.coverage == 21 / 22
    assert rep.efficiency == 21 / 22

    # Cross-track: solid {(0,20),(1,19)}; predicted cross {(0,20)}; hit cross {(0,20)}.
    assert rep.n_solid_cross == 2
    assert rep.n_predicted_cross == 1
    assert rep.n_hit_cross == 1
    assert rep.coverage_cross == 1 / 2
    assert rep.efficiency_cross == 1.0


def test_pair_coverage_threshold_filters_weak_matches(tmp_path):
    # One strong pair (>= min_inliers) and one weak pair (< min_inliers).
    _write_matches(tmp_path / "matches", [(0, 1)], inliers=25)
    _write_matches(tmp_path / "matches", [(2, 3)], inliers=5)  # appended, weak

    covis = CoVisGraph(frames=[0, 1, 2, 3], edges=[_edge(0, 1), _edge(2, 3)])
    rep = pair_coverage(covis, tmp_path / "matches", min_inliers=20)

    # Only (0,1) clears the inlier threshold, so solid == 1 and (2,3) is not counted.
    assert rep.n_solid == 1
    assert rep.n_hit == 1
    assert rep.coverage == 1.0
    assert rep.efficiency == 0.5  # predicted 2, hit 1
