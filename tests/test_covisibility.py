from __future__ import annotations

import numpy as np

from scene_recon.matching.covisibility import build_covisibility


def test_gps_rank_requirement_reports_reach_to_far_partner():
    """Diagnostic: `gps_rank_requirement` reports, per frame, the GPS-k-NN rank needed to
    reach its co-visible partners -- intel for *why* a global k would have to be large, not
    an instruction to set one. Here frame 0 co-sees only frame 10, with frames 1..9 sitting
    GPS-closer but not co-visible, so reaching the partner needs rank ~10."""
    n = 11
    positions = {i: (float(i), 0.0) for i in range(n)}
    # Disjoint cells per frame, except 0 and 10 share one -> the only co-visible pair.
    cells = {i: frozenset({(100 * i, 0)}) for i in range(n)}
    cells[0] = frozenset({(0, 0), (100 * 0, 0)})
    cells[10] = frozenset({(0, 0), (100 * 10, 0)})

    graph = build_covisibility(cells, positions, iom_min=0.1)
    assert any({e.i, e.j} == {0, 10} for e in graph.edges)

    needed = graph.gps_rank_requirement(per_frame_percentile=90.0)
    # Only the two co-visible frames contribute; each must reach a rank-10 partner.
    assert needed.size == 2
    assert np.all(needed >= 10)


def test_gps_rank_requirement_empty_without_positions_or_edges():
    assert build_covisibility({}, {}).gps_rank_requirement().size == 0


def test_summary_exposes_gps_rank_percentiles():
    """The co-visibility summary surfaces gps_rank_{p50,p95,max} so `analyze_matching`
    can report matcher-reach diagnostics."""
    positions = {i: (float(i), 0.0) for i in range(5)}
    common = frozenset({(0, 0), (1, 0), (-1, 0)})
    cells = {i: common for i in range(5)}
    summary = build_covisibility(cells, positions, iom_min=0.1).summary()
    for key in ("gps_rank_p50", "gps_rank_p95", "gps_rank_max"):
        assert key in summary
