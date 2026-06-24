"""Profile selection seam (Q1 extension point).

`RecommendOnly` is the default: take the single profile the backend derives from the
co-visibility graph and use it. `GuidedSweep` is the planned extension -- propose a small
set of profiles seeded from that recommendation, re-run the cheap SfM stage on each
(via `runner.rerun_sfm`), score them on the flight-agnostic `Scoreboard`, and pick the
winner by a composite objective. Both satisfy the same `Tuner` interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scene_recon.matching.covisibility import CoVisGraph
from scene_recon.matching.profile import MatcherProfile, ProfileBackend
from scene_recon.matching.scoreboard import Scoreboard


@runtime_checkable
class Tuner(Protocol):
    def propose(self, graph: CoVisGraph) -> list[MatcherProfile]:
        ...

    def select(
        self, results: list[tuple[MatcherProfile, Scoreboard]]
    ) -> MatcherProfile:
        ...


class RecommendOnly:
    """Default tuner: one geometry-derived profile, no re-running."""

    def __init__(self, backend: ProfileBackend) -> None:
        self.backend = backend

    def propose(self, graph: CoVisGraph) -> list[MatcherProfile]:
        return [self.backend.recommend(graph)]

    def select(
        self, results: list[tuple[MatcherProfile, Scoreboard]]
    ) -> MatcherProfile:
        if not results:
            raise ValueError("no candidate profiles to select from")
        return results[0][0]


def composite_objective(s: Scoreboard) -> float:
    """Single number a sweep maximises: reward cross-track ties, connectivity and longer
    tracks; penalise reprojection error. Kept here so `RecommendOnly` and a future
    `GuidedSweep` agree on what "better" means."""
    return (
        2.0 * s.cross_track_solid_ratio
        + 1.0 * s.algebraic_connectivity
        + 0.2 * s.track_len_mean
        - 0.5 * s.reproj_err_px
        - 0.5 * max(0, s.match_graph_components - 1)
    )


class GuidedSweep:
    """Planned Q1 extension: seeded sweep + re-run + scoreboard-driven selection.

    propose() perturbs the backend recommendation; the driver re-runs SfM per candidate
    and scores it; select() returns the argmax of `composite_objective`. Wiring the
    re-run loop (`runner.rerun_sfm`) is the remaining work."""

    def __init__(self, backend: ProfileBackend) -> None:
        self.backend = backend

    def propose(self, graph: CoVisGraph) -> list[MatcherProfile]:  # pragma: no cover
        raise NotImplementedError(
            "GuidedSweep is the planned Q1 extension; use RecommendOnly for now."
        )

    def select(
        self, results: list[tuple[MatcherProfile, Scoreboard]]
    ) -> MatcherProfile:
        return max(results, key=lambda r: composite_objective(r[1]))[0]
