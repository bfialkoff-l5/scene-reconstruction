from __future__ import annotations

import json

from scene_recon.matching.covisibility import CoVisEdge, CoVisGraph
from scene_recon.matching.profile import ExplicitPairListBackend, explicit_pairs


def _edge(i: int, j: int) -> CoVisEdge:
    return CoVisEdge(i=i, j=j, iom=0.5, shared_cells=10, centroid_dist_m=10.0,
                     view_angle_deg=0.0)


def test_explicit_pairs_union_covis_and_sequential():
    # frames sorted [0,1,2,5]; one cross-track covis edge (0,5).
    graph = CoVisGraph(frames=[0, 1, 2, 5], edges=[_edge(0, 5)])
    pairs = explicit_pairs(graph, seq_window=2)

    expected = {
        ("000000.png", "000005.png"),  # covis
        ("000000.png", "000001.png"),  # seq d=1
        ("000001.png", "000002.png"),
        ("000002.png", "000005.png"),
        ("000000.png", "000002.png"),  # seq d=2
        ("000001.png", "000005.png"),
    }
    assert set(pairs) == expected
    assert pairs == sorted(pairs)  # deterministic, sorted
    assert all(a < b for a, b in pairs)  # each pair is sorted


def test_explicit_pairs_dedup_when_covis_equals_sequential():
    # covis edge (0,1) coincides with a sequential pair -> no double count.
    graph = CoVisGraph(frames=[0, 1, 2], edges=[_edge(0, 1)])
    pairs = explicit_pairs(graph, seq_window=1)
    assert set(pairs) == {
        ("000000.png", "000001.png"),
        ("000001.png", "000002.png"),
    }


def test_backend_writes_pairs_and_options(tmp_path):
    graph = CoVisGraph(frames=[0, 1, 2, 5], edges=[_edge(0, 5)])
    (tmp_path / "odm_options.json").write_text(json.dumps({"existing": 1}))

    backend = ExplicitPairListBackend(seq_window=2)
    profile = backend.recommend(graph)
    backend.apply(tmp_path, profile, graph)

    payload = json.loads((tmp_path / "covis_pairs.json").read_text())
    assert payload["source"] == "explicit_pairs"
    assert payload["seq_window"] == 2
    assert payload["n_pairs"] == 6
    assert [tuple(p) for p in payload["pairs"]] == explicit_pairs(graph, seq_window=2)

    # odm_options.json keeps the stock matcher_neighbors fallback + prior keys.
    opts = json.loads((tmp_path / "odm_options.json").read_text())
    assert opts["existing"] == 1
    assert opts["matcher_neighbors"] == profile.gps_neighbors

    audit = json.loads((tmp_path / "matching_profile.json").read_text())
    assert audit["profile"]["source"] == "explicit_pairs"
    assert audit["profile"]["extra"]["n_pairs"] == 6
    assert audit["profile"]["extra"]["n_sequential_added"] == 5  # 6 total - 1 covis


def test_write_pairs_file_does_not_touch_options(tmp_path):
    graph = CoVisGraph(frames=[0, 1, 2], edges=[_edge(0, 2)])
    n = ExplicitPairListBackend().write_pairs_file(tmp_path, graph)
    assert n == 3  # (0,2) covis + (0,1),(1,2) sequential
    assert (tmp_path / "covis_pairs.json").is_file()
    assert not (tmp_path / "odm_options.json").exists()
    assert not (tmp_path / "matching_profile.json").exists()
