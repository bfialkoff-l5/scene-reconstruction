from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

SHIM_PATH = (
    Path(__file__).resolve().parents[1] / "docker" / "covis_shim" / "covis_pairs_shim.py"
)


def _load_shim():
    spec = importlib.util.spec_from_file_location("covis_pairs_shim_under_test", SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # _install() is a no-op without COVIS_PAIRS_FILE
    return mod


def _fake_opensfm_module():
    """Stand-in for opensfm.pairs_selection with the two helpers the shim relies on."""
    m = types.SimpleNamespace()
    m.sorted_pair = lambda a, b: (a, b) if a < b else (b, a)
    m.ordered_pairs = lambda pairs, ref: sorted(pairs)
    m.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    return m


def _write_pairs(tmp_path, pairs):
    p = tmp_path / "covis_pairs.json"
    p.write_text(json.dumps({"source": "explicit_pairs", "n_pairs": len(pairs),
                             "pairs": [list(x) for x in pairs]}))
    return p


def test_shim_filters_to_available_and_returns_explicit(tmp_path):
    shim = _load_shim()
    mod = _fake_opensfm_module()
    pairs_file = _write_pairs(
        tmp_path,
        [("000000.png", "000005.png"), ("000000.png", "999999.png")],  # 2nd unavailable
    )
    shim._patch(mod, str(pairs_file))

    images = ["000000.png", "000001.png", "000002.png", "000005.png"]
    out, report = mod.match_candidates_from_metadata(images, images, {}, None, {})

    assert out == [("000000.png", "000005.png")]  # unavailable pair dropped
    assert report["num_pairs_explicit"] == 1
    assert report["num_pairs_distance"] == 0


def test_shim_hard_fails_when_no_usable_pairs(tmp_path):
    shim = _load_shim()
    mod = _fake_opensfm_module()
    pairs_file = _write_pairs(tmp_path, [("aaa.png", "bbb.png")])  # none in dataset
    shim._patch(mod, str(pairs_file))

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        mod.match_candidates_from_metadata(["000000.png", "000001.png"],
                                           ["000000.png", "000001.png"], {}, None, {})


def test_shim_install_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("COVIS_PAIRS_FILE", raising=False)
    shim = _load_shim()
    before = list(__import__("sys").meta_path)
    shim._install()
    assert list(__import__("sys").meta_path) == before  # no finder added
