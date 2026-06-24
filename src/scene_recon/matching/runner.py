"""Re-run hook for the auto-tuner (Q1 extension point).

A profile sweep needs to re-run *only* the cheap OpenSfM stage (matching + tracks +
reconstruction, reusing the already-extracted features) so each candidate costs minutes,
not a full GPU pipeline. That hook lives here so `tuner.GuidedSweep` can call it without
knowing how ODM/Docker is invoked. Implemented when the tuner is built.
"""

from __future__ import annotations

from pathlib import Path


def rerun_sfm(odm_input: Path) -> Path:  # pragma: no cover
    """Re-run OpenSfM matching + reconstruction in place, reusing extracted features.

    Returns the `opensfm` dir to score. Intended to drive the ODM container with feature
    extraction already cached (≈5 min on this dataset per `profile.log`).
    """
    raise NotImplementedError(
        "rerun_sfm is the Q1 auto-tuner hook; not yet wired to the ODM container."
    )
