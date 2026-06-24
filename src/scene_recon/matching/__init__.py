"""Data-driven matching-profile tooling.

Stock OpenSfM/ODM matching picks candidate image pairs by GPS k-nearest-neighbours
(`matching_gps_neighbors`). At tight along-track spacing the k nearest frames are all
*along-track* neighbours, so the parallel/return strip never enters the candidate set and
the reconstruction degenerates into a drifting chain (warped orthophoto). The right
invariant is not "how many neighbours" but "which frames actually see the same ground" --
the **co-visibility graph**, derived per-flight from that flight's own poses + terrain.

This package is built around three decoupled pieces with explicit extension seams:

- `covisibility` -- geometry -> ideal candidate-pair set (reuses the footprint cache).
- `profile`      -- ideal pairs -> matcher config, behind a `ProfileBackend` seam
                    (Q2: stock knobs now; explicit pair-list / self-computed matches later).
- `scoreboard`   -- flight-agnostic intrinsic metrics from an OpenSfM output dir.
- `tuner`        -- `Tuner` seam (Q1: `RecommendOnly` now; `GuidedSweep` later).
"""

from scene_recon.matching.covisibility import (
    CoVisEdge,
    CoVisGraph,
    build_covisibility,
    covisibility_from_footprints,
)
from scene_recon.matching.profile import (
    MatcherProfile,
    ProfileBackend,
    StockKnobsBackend,
)
from scene_recon.matching.scoreboard import Scoreboard, score_opensfm
from scene_recon.matching.tuner import RecommendOnly, Tuner

__all__ = [
    "CoVisEdge",
    "CoVisGraph",
    "MatcherProfile",
    "ProfileBackend",
    "RecommendOnly",
    "Scoreboard",
    "StockKnobsBackend",
    "Tuner",
    "build_covisibility",
    "covisibility_from_footprints",
    "score_opensfm",
]
