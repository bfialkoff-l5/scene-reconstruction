"""In-container shim that makes OpenSfM match exactly our explicit co-visibility pair list.

ODM/OpenSfM has no stock "read candidate pairs from a file" input: its
`pairs_selection.match_candidates_from_metadata` builds the candidate set as the *union*
of GPS-distance / graph / time / order / BoW / VLAD strategies. We compute the matchable
pairs ourselves from the footprint co-visibility graph (cross-track + loop-closure pairs a
GPS k-NN silently drops) and write them to ``covis_pairs.json``; this shim swaps OpenSfM's
candidate generator for one that returns precisely that list.

Activation is opt-in and side-effect-free unless ``COVIS_PAIRS_FILE`` is set, so the same
image can run stock or explicit-pairs matching with no code change. It is loaded at
interpreter start via a sibling ``.pth`` file dropped next to it on ``site-packages``; the
patch itself is *deferred* (installed through a one-shot meta-path finder) so unrelated
Python processes never import OpenSfM just because the shim is present.

If ``COVIS_PAIRS_FILE`` is set but unusable, matching aborts loudly -- we never silently
fall back to the along-track-only k-NN that this whole effort exists to avoid.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys

_TARGET = "opensfm.pairs_selection"


def _load_pairs(pairs_file: str) -> list[tuple[str, str]]:
    with open(pairs_file) as fh:
        payload = json.load(fh)
    raw = payload["pairs"] if isinstance(payload, dict) else payload
    return [(str(a), str(b)) for a, b in raw]


def _patch(module, pairs_file: str) -> None:
    sorted_pair = module.sorted_pair
    ordered_pairs = module.ordered_pairs

    def match_candidates_from_metadata(
        images_ref, images_cand, exifs, data, config_override
    ):
        available = set(images_ref) | set(images_cand)
        pairs = set()
        dropped = 0
        for a, b in _load_pairs(pairs_file):
            if a == b:
                continue
            if a in available and b in available:
                pairs.add(sorted_pair(a, b))
            else:
                dropped += 1
        if not pairs:
            raise RuntimeError(
                f"COVIS_PAIRS_FILE={pairs_file} yielded 0 usable pairs "
                f"(available images={len(available)}); refusing to fall back to k-NN."
            )
        ordered = ordered_pairs(pairs, list(images_ref))
        msg = (
            f"[covis-shim] explicit pairs: {len(ordered)} matched "
            f"({dropped} dropped, not in dataset) from {pairs_file}"
        )
        print(msg, file=sys.stderr, flush=True)
        try:
            module.logger.info(msg)
        except Exception:
            pass
        report = {
            "num_pairs_explicit": len(ordered),
            "num_pairs_distance": 0,
            "num_pairs_graph": 0,
            "num_pairs_time": 0,
            "num_pairs_order": 0,
            "num_pairs_bow": 0,
            "num_pairs_vlad": 0,
        }
        return ordered, report

    module.match_candidates_from_metadata = match_candidates_from_metadata


class _DeferredPatcher(importlib.abc.MetaPathFinder):
    """One-shot finder: when ``opensfm.pairs_selection`` is first imported, let the real
    finders load it, then patch the loaded module."""

    def __init__(self, pairs_file: str) -> None:
        self.pairs_file = pairs_file

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)  # one-shot; avoid recursing into find_spec below
        except ValueError:
            pass
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        pairs_file = self.pairs_file

        def exec_module(module, _orig=orig_exec, _pf=pairs_file):
            _orig(module)
            _patch(module, _pf)

        spec.loader.exec_module = exec_module
        return spec


def _install() -> None:
    pairs_file = os.environ.get("COVIS_PAIRS_FILE")
    if not pairs_file:
        return
    already = sys.modules.get(_TARGET)
    if already is not None:
        _patch(already, pairs_file)
        return
    sys.meta_path.insert(0, _DeferredPatcher(pairs_file))


_install()
