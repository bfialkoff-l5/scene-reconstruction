#!/usr/bin/env python3
"""Analyse the matching profile of an ODM run: score the existing SfM output and compare
it against the geometric co-visibility we *should* have matched.

    python scripts/analyze_matching.py <run_dir> [<run_dir> ...]

`<run_dir>` is a run folder (…/runs/<name>) or its `odm_input`. For each run it prints the
flight-agnostic scoreboard, the recommended matcher profile, and (when the slug footprint
cache + reconstruction poses are available) the co-visibility recall/precision. Writes a
report JSON + diagnostic plots into `<run>/match_analysis/`.

Read-only: it never re-runs ODM and never edits the run's opensfm artefacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_recon.matching.covisibility import build_covisibility  # noqa: E402
from scene_recon.matching.profile import StockKnobsBackend  # noqa: E402
from scene_recon.matching.scoreboard import score_opensfm  # noqa: E402
from scene_recon.selection.footprint import load_footprints  # noqa: E402


def _resolve(run: Path) -> tuple[Path, Path]:
    """Return (odm_input, opensfm) for a run dir or an odm_input dir."""
    odm_input = run if (run / "opensfm").is_dir() else run / "odm_input"
    return odm_input, odm_input / "opensfm"


def _frame_num(name: str) -> int:
    return int(name.split(".")[0])


def _load_geo(geo: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for ln in geo.read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) >= 3:
            out[_frame_num(p[0])] = (float(p[1]), float(p[2]))
    return out


def _view_dirs_from_reconstruction(opensfm: Path) -> dict[int, np.ndarray]:
    """Optical axis (world frame) per shot, from OpenSfM axis-angle rotation.

    OpenSfM stores world->camera rotation; the camera +Z axis in world coords is R^T·ẑ.
    """
    import cv2

    rec_path = opensfm / "reconstruction.json"
    if not rec_path.is_file():
        return {}
    rec = json.loads(rec_path.read_text())[0]
    out: dict[int, np.ndarray] = {}
    z = np.array([0.0, 0.0, 1.0])
    for name, shot in rec["shots"].items():
        R, _ = cv2.Rodrigues(np.asarray(shot["rotation"], dtype=float))
        out[_frame_num(name)] = R.T @ z
    return out


def _covisibility_for_run(odm_input: Path, opensfm: Path):
    """Build the co-visibility graph from the slug footprint cache + run poses.

    Returns None when the footprint cache is unavailable (e.g. fresh box) -- the
    scoreboard still works, just without recall/precision.
    """
    geo = _load_geo(odm_input / "geo.txt")
    selected = set(geo)
    # footprints.pkl lives at the slug root: …/<slug>/footprints.pkl (runs/<name>/odm_input).
    cache = odm_input.parents[2] / "footprints.pkl"
    if not cache.is_file():
        print(f"  [covis] no footprint cache at {cache}; skipping co-visibility")
        return None
    print(f"  [covis] loading footprint cache {cache} …")
    fps = load_footprints(cache)
    cells = {
        f: fp.cells
        for f, fp in fps.items()
        if f in selected and getattr(fp, "valid", False) and fp.cells
    }
    view_dirs = _view_dirs_from_reconstruction(opensfm)
    return build_covisibility(cells, geo, view_dirs=view_dirs or None)


def _plots(out_dir: Path, geo: dict[int, tuple[float, float]], opensfm: Path, covis) -> None:
    try:
        import gzip
        import pickle

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [plots] matplotlib unavailable ({exc}); skipping plots")
        return

    names = sorted(geo, key=int)
    rank = {f: r for r, f in enumerate(names)}

    # solid matched pairs from the matches dir
    solid: list[tuple[int, int, int]] = []
    for f in (opensfm / "matches").glob("*_matches.pkl.gz"):
        base = _frame_num(f.name.replace("_matches.pkl.gz", ""))
        with gzip.open(f, "rb") as fh:
            m = pickle.load(fh)
        for partner, arr in m.items():
            a = np.asarray(arr)
            n = int(a.shape[0]) if a.ndim == 2 else 0
            pn = _frame_num(partner)
            if n >= 20 and base < pn:
                solid.append((base, pn, n))

    # Plot 1: actual match graph over the GPS track, edges coloured by along-track gap.
    fig, ax = plt.subplots(figsize=(10, 9))
    xs = [geo[f][0] for f in names]
    ys = [geo[f][1] for f in names]
    if solid:
        gaps = np.array([abs(rank[i] - rank[j]) for i, j, _ in solid])
        norm = plt.Normalize(0, max(gaps.max(), 1))
        cmap = plt.cm.viridis
        for (i, j, _), g in zip(solid, gaps):
            ax.plot(
                [geo[i][0], geo[j][0]], [geo[i][1], geo[j][1]],
                color=cmap(norm(g)), lw=0.4, alpha=0.5,
            )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="along-track rank gap (high = cross-track / loop)")
    ax.scatter(xs, ys, s=6, c="k", zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Actual solid match graph (>=20 inliers)")
    fig.savefig(out_dir / "match_graph_actual.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: predicted co-visibility graph, same layout, for side-by-side comparison.
    if covis is not None and covis.edges:
        fig, ax = plt.subplots(figsize=(10, 9))
        gaps = np.array([abs(rank[e.i] - rank[e.j]) for e in covis.edges])
        norm = plt.Normalize(0, max(gaps.max(), 1))
        cmap = plt.cm.plasma
        for e, g in zip(covis.edges, gaps):
            if e.i in geo and e.j in geo:
                ax.plot(
                    [geo[e.i][0], geo[e.j][0]], [geo[e.i][1], geo[e.j][1]],
                    color=cmap(norm(g)), lw=0.4, alpha=0.5,
                )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="along-track rank gap")
        ax.scatter(xs, ys, s=6, c="k", zorder=3)
        ax.set_aspect("equal")
        ax.set_title("Predicted co-visibility graph (geometry)")
        fig.savefig(out_dir / "covisibility_predicted.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"  [plots] wrote diagnostics to {out_dir}")


def analyse(run: Path) -> None:
    odm_input, opensfm = _resolve(run)
    if not opensfm.is_dir():
        print(f"!! no opensfm dir under {run}")
        return
    print(f"\n=== {run} ===")
    covis = _covisibility_for_run(odm_input, opensfm)
    sb = score_opensfm(opensfm, covis=covis)
    print(sb.render())

    report = {"run": str(run), "scoreboard": sb.to_json()}
    if covis is not None:
        cs = covis.summary()
        print("\n  co-visibility (predicted):")
        for k, v in cs.items():
            print(f"    {k}: {v}")
        rec = StockKnobsBackend().recommend(covis)
        print("\n  recommended profile:", rec.to_json())
        report["covisibility"] = cs
        report["recommended_profile"] = rec.to_json()

    out_dir = run / "match_analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _plots(out_dir, _load_geo(odm_input / "geo.txt"), opensfm, covis)
    print(f"  report -> {out_dir / 'report.json'}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for run in argv:
        analyse(Path(run).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
