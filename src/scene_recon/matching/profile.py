"""Matcher profile + the backend seam that drives the matcher (Q2 extension point).

`StockKnobsBackend` is the default: it derives ODM's `--matcher-neighbors` /
`--matcher-distance` from the co-visibility graph and writes them into
`odm_options.json`, staying on the existing `run_odm.sh` path. The deeper-integration
options (an explicit OpenSfM pair list, or matches we compute ourselves) are declared as
same-interface stubs so they drop in later without touching callers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from scene_recon.matching.covisibility import CoVisGraph


@dataclass
class MatcherProfile:
    """A matcher configuration the tooling can recommend and apply.

    Fields map onto ODM matching knobs; `extra` carries backend-specific payload (e.g. an
    explicit pair list) without widening the stock interface.
    """

    gps_neighbors: int
    gps_distance_m: float
    graph_rounds: int = 0
    source: str = "stock_knobs"
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


@runtime_checkable
class ProfileBackend(Protocol):
    """Seam between the co-visibility graph and how the matcher is actually driven."""

    name: str

    def recommend(self, graph: CoVisGraph) -> MatcherProfile:
        ...

    def apply(self, odm_input: Path, profile: MatcherProfile, graph: CoVisGraph) -> None:
        ...


class StockKnobsBackend:
    """Default backend: translate the co-visibility graph into ODM matcher knobs.

    - `gps_neighbors` = p95 of per-frame co-visible degree -> enough candidates per frame
      to reach the parallel/return strip, not just along-track neighbours.
    - `gps_distance_m` = p95 of co-visible baseline length -> a metric reach gate that
      lets cross-track pairs in regardless of how many along-track frames sit closer.

    Percentiles (not max) so a handful of outlier frames do not blow up matching cost.
    """

    name = "stock_knobs"

    def __init__(
        self,
        *,
        neighbor_percentile: float = 95.0,
        distance_percentile: float = 95.0,
        neighbor_floor: int = 8,
        neighbor_cap: int = 64,
        graph_rounds: int = 0,
    ) -> None:
        self.neighbor_percentile = neighbor_percentile
        self.distance_percentile = distance_percentile
        self.neighbor_floor = neighbor_floor
        self.neighbor_cap = neighbor_cap
        self.graph_rounds = graph_rounds

    def recommend(self, graph: CoVisGraph) -> MatcherProfile:
        deg = np.array(list(graph.degrees().values()), dtype=float)
        dists = np.array([e.centroid_dist_m for e in graph.edges], dtype=float)
        if deg.size == 0 or dists.size == 0:
            # No co-visibility signal: fall back to a conservative non-zero profile.
            return MatcherProfile(
                gps_neighbors=self.neighbor_floor, gps_distance_m=0.0,
                graph_rounds=self.graph_rounds, source=self.name,
            )
        neighbors = int(np.ceil(np.percentile(deg, self.neighbor_percentile)))
        neighbors = max(self.neighbor_floor, min(self.neighbor_cap, neighbors))
        distance = float(np.percentile(dists, self.distance_percentile))
        return MatcherProfile(
            gps_neighbors=neighbors,
            gps_distance_m=round(distance, 2),
            graph_rounds=self.graph_rounds,
            source=self.name,
        )

    def apply(self, odm_input: Path, profile: MatcherProfile, graph: CoVisGraph) -> None:
        """Write the profile so `run_odm.sh` picks it up.

        Stock ODM exposes only `--matcher-neighbors` (no metric distance / graph-rounds
        flags), so that is the single knob written into `odm_options.json`; the GPS k-NN
        count is derived from co-visible degree, large enough to reach the cross-track /
        loop-closure pairs. `gps_distance_m` and `graph_rounds` are recorded in the
        `matching_profile.json` audit for the deeper-integration backends that can honour
        them (explicit pair list / self-computed matches), but are not passed to ODM.
        """
        opts_path = odm_input / "odm_options.json"
        opts = json.loads(opts_path.read_text()) if opts_path.is_file() else {}
        opts["matcher_neighbors"] = profile.gps_neighbors
        opts_path.write_text(json.dumps(opts, indent=2) + "\n")

        audit = {
            "profile": profile.to_json(),
            "covisibility": graph.summary(),
        }
        (odm_input / "matching_profile.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )


class ExplicitPairListBackend:
    """Q2 deeper-integration stub: match exactly the co-visible pair set by handing
    OpenSfM a precomputed pair list (captures sparse loop-closure links without paying
    for all-pairs matching). Not yet implemented."""

    name = "explicit_pairs"

    def recommend(self, graph: CoVisGraph) -> MatcherProfile:  # pragma: no cover
        raise NotImplementedError(
            "ExplicitPairListBackend is a planned Q2 extension; use StockKnobsBackend."
        )

    def apply(self, odm_input: Path, profile: MatcherProfile, graph: CoVisGraph) -> None:  # pragma: no cover
        raise NotImplementedError(
            "ExplicitPairListBackend is a planned Q2 extension; use StockKnobsBackend."
        )


class SelfMatchBackend:
    """Q2 deepest-integration stub: compute feature matches ourselves over the co-visible
    pairs and write them straight into `opensfm/matches/*.pkl.gz`, bypassing ODM's
    matcher entirely. Not yet implemented."""

    name = "self_match"

    def recommend(self, graph: CoVisGraph) -> MatcherProfile:  # pragma: no cover
        raise NotImplementedError(
            "SelfMatchBackend is a planned Q2 extension; use StockKnobsBackend."
        )

    def apply(self, odm_input: Path, profile: MatcherProfile, graph: CoVisGraph) -> None:  # pragma: no cover
        raise NotImplementedError(
            "SelfMatchBackend is a planned Q2 extension; use StockKnobsBackend."
        )
