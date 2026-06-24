"""Flight-agnostic scoreboard: intrinsic quality metrics from an OpenSfM output dir.

No ground truth and no DTM required -- every metric is computable from the artefacts ODM
already writes, so it works on any flight and is the ruler the tuner optimises against.

Sources:
- `matches/*.pkl.gz` + `geo.txt`  -> pairwise inlier graph, cross-track ratio, connectivity
- `stats/stats.json`              -> track length, reprojection error, components, GPS CE90/LE90

The headline metric is `cross_track_solid_ratio`: the share of strong image pairs that
link temporally distant frames. A drifting along-track chain scores ~0; a well-tied
reconstruction scores high. Pure numpy (eigendecomposition on a few-hundred-node graph).
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scoreboard:
    n_images: int
    solid_pairs: int
    mean_inliers: float
    cross_track_solid_pairs: int
    cross_track_solid_ratio: float
    match_graph_components: int
    algebraic_connectivity: float  # Fiedler value of the largest component
    track_len_mean: float
    track_len_median: float
    reproj_err_px: float
    recon_shot_frac: float
    gps_ce90: float
    gps_le90: float
    covis_recall: float  # of predicted co-visible pairs, fraction that became solid
    covis_precision: float  # of solid pairs, fraction that were predicted

    def to_json(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        rows = [
            ("images", f"{self.n_images}"),
            ("solid pairs (>=thr inliers)", f"{self.solid_pairs}"),
            ("mean inliers / solid pair", f"{self.mean_inliers:.0f}"),
            ("cross-track solid pairs", f"{self.cross_track_solid_pairs}"),
            ("cross-track solid ratio", f"{self.cross_track_solid_ratio:.3f}"),
            ("match-graph components", f"{self.match_graph_components}"),
            ("algebraic connectivity", f"{self.algebraic_connectivity:.4f}"),
            ("track length mean", f"{self.track_len_mean:.2f}"),
            ("track length median", f"{self.track_len_median:.1f}"),
            ("reprojection error (px)", f"{self.reproj_err_px:.3f}"),
            ("reconstructed shot frac", f"{self.recon_shot_frac:.3f}"),
            ("GPS CE90 (m)", f"{self.gps_ce90:.2f}"),
            ("GPS LE90 (m)", f"{self.gps_le90:.2f}"),
            ("co-vis recall", _fmt(self.covis_recall)),
            ("co-vis precision", _fmt(self.covis_precision)),
        ]
        w = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k.ljust(w)} : {v}" for k, v in rows)


def _fmt(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.3f}"


def _load_geo(geo_path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for ln in geo_path.read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) >= 3:
            out[p[0]] = (float(p[1]), float(p[2]))
    return out


def _frame_rank(names) -> dict[str, int]:
    """Along-track rank from the frame number embedded in the filename."""
    def num(n: str) -> int:
        try:
            return int(n.split(".")[0])
        except ValueError:
            return 0
    order = sorted(names, key=num)
    return {n: r for r, n in enumerate(order)}


def _load_solid_pairs(
    matches_dir: Path, min_inliers: int
) -> tuple[dict[tuple[str, str], int], set[str]]:
    """Return {unordered image pair: inlier count} for pairs with >= min_inliers, and the
    set of all images that produced a match file."""
    pairs: dict[tuple[str, str], int] = {}
    images: set[str] = set()
    for f in matches_dir.glob("*_matches.pkl.gz"):
        base = f.name.replace("_matches.pkl.gz", "")
        images.add(base)
        with gzip.open(f, "rb") as fh:
            m = pickle.load(fh)
        for partner, arr in m.items():
            images.add(partner)
            a = np.asarray(arr)
            n = int(a.shape[0]) if a.ndim == 2 else 0
            if n < min_inliers:
                continue
            key = (base, partner) if base < partner else (partner, base)
            pairs[key] = max(pairs.get(key, 0), n)
    return pairs, images


def _components_and_fiedler(
    nodes: list[str], edges: list[tuple[str, str]]
) -> tuple[int, float]:
    idx = {n: i for i, n in enumerate(nodes)}
    adj: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for a, b in edges:
        if a in idx and b in idx:
            adj[idx[a]].add(idx[b])
            adj[idx[b]].add(idx[a])
    # connected components via BFS
    seen = [False] * len(nodes)
    comps: list[list[int]] = []
    for s in range(len(nodes)):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    q.append(v)
        comps.append(comp)
    if not comps:
        return 0, 0.0
    # Fiedler value of the largest component's normalised Laplacian.
    big = max(comps, key=len)
    if len(big) < 2:
        return len(comps), 0.0
    local = {g: k for k, g in enumerate(big)}
    A = np.zeros((len(big), len(big)))
    for u in big:
        for v in adj[u]:
            if v in local:
                A[local[u], local[v]] = 1.0
    d = A.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(len(big)) - (dinv[:, None] * A * dinv[None, :])
    w = np.linalg.eigvalsh((L + L.T) / 2)
    fiedler = float(w[1]) if len(w) > 1 else 0.0
    return len(comps), fiedler


def _track_len_median(hist: dict[str, int]) -> float:
    lengths = sorted(int(k) for k in hist)
    counts = np.array([hist[str(k)] for k in lengths], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, total / 2.0))
    return float(lengths[min(i, len(lengths) - 1)])


def score_opensfm(
    opensfm_dir: str | Path,
    *,
    covis=None,
    min_inliers: int = 20,
    cross_track_rank_gap: int = 16,
) -> Scoreboard:
    """Compute the scoreboard for one OpenSfM output directory.

    Args:
        opensfm_dir: a run's `.../odm_input/opensfm` directory.
        covis: optional `CoVisGraph` to compute recall/precision of the predicted pairs.
        min_inliers: inlier threshold for a pair to count as "solid".
        cross_track_rank_gap: along-track rank gap above which a pair is "cross-track".
    """
    opensfm_dir = Path(opensfm_dir)
    geo = _load_geo(opensfm_dir.parent / "geo.txt")
    pairs, images = _load_solid_pairs(opensfm_dir / "matches", min_inliers)

    all_names = set(images) | set(geo)
    rank = _frame_rank(all_names)
    edges = list(pairs.keys())
    inliers = np.array(list(pairs.values()), dtype=float)
    gaps = np.array([abs(rank[a] - rank[b]) for a, b in edges], dtype=int)
    cross = int((gaps > cross_track_rank_gap).sum())

    nodes = sorted(all_names)
    n_comp, fiedler = _components_and_fiedler(nodes, edges)

    stats = json.loads((opensfm_dir / "stats" / "stats.json").read_text())
    rs = stats.get("reconstruction_statistics", {})
    fs = stats.get("features_statistics", {})
    hist = rs.get("histogram_track_length", {})
    gps = stats.get("gps_errors", {})
    initial = rs.get("initial_shots_count", 0) or 0
    recon = rs.get("reconstructed_shots_count", 0) or 0

    recall = precision = float("nan")
    if covis is not None and covis.edges:
        def key(i, j):
            ni, nj = f"{i:06d}.png", f"{j:06d}.png"
            return (ni, nj) if ni < nj else (nj, ni)

        predicted = {key(e.i, e.j) for e in covis.edges}
        solid = set(pairs.keys())
        hit = len(predicted & solid)
        recall = hit / len(predicted) if predicted else float("nan")
        precision = hit / len(solid) if solid else float("nan")

    return Scoreboard(
        n_images=len(all_names),
        solid_pairs=len(pairs),
        mean_inliers=float(inliers.mean()) if inliers.size else 0.0,
        cross_track_solid_pairs=cross,
        cross_track_solid_ratio=float(cross / len(edges)) if edges else 0.0,
        match_graph_components=n_comp,
        algebraic_connectivity=fiedler,
        track_len_mean=float(rs.get("average_track_length", 0.0)),
        track_len_median=_track_len_median(hist),
        reproj_err_px=float(rs.get("reprojection_error_pixels", 0.0)),
        recon_shot_frac=float(recon / initial) if initial else 0.0,
        gps_ce90=float(gps.get("ce90", float("nan"))),
        gps_le90=float(gps.get("le90", float("nan"))),
        covis_recall=recall,
        covis_precision=precision,
    )
