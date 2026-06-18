from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scene_recon.selection import (
    FootprintCache,
    GroundGrid,
    SelectionParams,
    coverage_metrics,
    iter_selection_gaps,
    params_from_constants,
)
from scene_recon.selection.footprint import ViewCounts
from scene_recon.selection.metrics import CoverageMetrics
from scene_recon.selection_health import SelectionHealth

log = logging.getLogger(__name__)


def build_audit_df(candidates: pd.DataFrame) -> pd.DataFrame:
    audit = candidates.copy()
    audit["bin_rank"] = (
        audit.groupby(["cell_x", "cell_y"], observed=True)["quality_score"]
        .rank(ascending=False, method="first")
        .astype("Int64")
    )

    selected = audit[audit["selected"]]
    audit["dist_to_nearest_selected_m"] = pd.NA
    if not selected.empty:
        se = selected["easting"].to_numpy(dtype=float)
        sn = selected["northing"].to_numpy(dtype=float)
        ce = audit["easting"].to_numpy(dtype=float)
        cn = audit["northing"].to_numpy(dtype=float)
        de = ce[:, None] - se[None, :]
        dn = cn[:, None] - sn[None, :]
        dist = np.hypot(de, dn)
        is_selected = audit["selected"].to_numpy()
        dist[is_selected, :] = np.where(
            dist[is_selected, :] > 0,
            dist[is_selected, :],
            np.inf,
        )
        nearest = dist.min(axis=1)
        audit["dist_to_nearest_selected_m"] = np.where(
            np.isfinite(nearest),
            nearest,
            pd.NA,
        )

    return audit


def _decision_breakdown(candidates: pd.DataFrame) -> dict:
    selected = candidates[candidates["selected"]]
    if selected.empty:
        return {}
    out: dict = {}
    if "selection_reason" in selected.columns:
        counts = selected["selection_reason"].value_counts(dropna=False)
        out["selection_reason_counts"] = {str(k): int(v) for k, v in counts.items()}
    if "coverage_gain_cells" in selected.columns:
        gain = pd.to_numeric(selected["coverage_gain_cells"], errors="coerce").fillna(0)
        marginal = selected.index[gain <= 0].tolist()
        out["n_marginal_selections"] = int(len(marginal))
        out["marginal_frames"] = [int(i) for i in marginal[:20]]
    if "bin_rank" in selected.columns:
        bin_rank = pd.to_numeric(selected["bin_rank"], errors="coerce").dropna()
        out["selected_bin_rank"] = {
            "median": float(bin_rank.median()) if len(bin_rank) else None,
            "max": int(bin_rank.max()) if len(bin_rank) else None,
        }
    return out


def build_summary(
    candidates: pd.DataFrame,
    constants: dict,
    *,
    health: SelectionHealth | None = None,
    view_counts: ViewCounts | None = None,
    coverage: CoverageMetrics | None = None,
) -> dict:
    selected = candidates[candidates["selected"]]
    reject_counts = (
        candidates.loc[~candidates["selected"], "reject_reason"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict()
    )

    params = params_from_constants(constants)
    coverage_gaps: list[dict] = []
    temporal_gaps: list[dict] = []
    if len(selected) >= 2:
        for gap in iter_selection_gaps(selected.sort_index()):
            temporal_gaps.append(gap.as_dict())
            if gap.gap_m > params.coverage_warn_m:
                coverage_gaps.append(
                    {
                        "from_frame": gap.from_frame,
                        "to_frame": gap.to_frame,
                        "gap_m": round(gap.gap_m, 2),
                    }
                )

    if coverage is None:
        coverage = coverage_metrics(
            view_counts or {}, params.target_views_per_cell, bin_size_m=params.bin_size_m
        )

    quality = selected["quality_score"].astype(float)
    summary = {
        **constants,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "reject_counts": {str(k): int(v) for k, v in reject_counts.items()},
        "decisions": _decision_breakdown(candidates),
        "coverage": coverage.as_dict(),
        "temporal_gaps": temporal_gaps,
        "coverage_gaps": coverage_gaps,
        "quality_score_selected": {
            "min": round(float(quality.min()), 4) if len(quality) else None,
            "median": round(float(quality.median()), 4) if len(quality) else None,
            "max": round(float(quality.max()), 4) if len(quality) else None,
        },
    }
    if health is not None:
        summary["health"] = health.as_dict()
    return summary


def _add_relative_axes(ax, ref_e, ref_n) -> None:
    """Secondary top/right axes showing metres relative to the start point."""
    sx = ax.secondary_xaxis("top", functions=(lambda e: e - ref_e, lambda e: e + ref_e))
    sx.set_xlabel("east of start (m)")
    sy = ax.secondary_yaxis("right", functions=(lambda n: n - ref_n, lambda n: n + ref_n))
    sy.set_ylabel("north of start (m)")


def _scatter_by_category(ax, candidates, column, title, ref_e, ref_n, xlim, ylim) -> None:
    subset = candidates[candidates[column].notna()]
    cats = sorted(subset[column].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        rows = subset[subset[column].astype(str) == cat]
        ax.scatter(
            rows["easting"],
            rows["northing"],
            s=6,
            color=cmap(i % 10),
            label=cat,
        )
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(title)
    _add_relative_axes(ax, ref_e, ref_n)
    if cats:
        ax.legend(loc="upper right", fontsize=7, markerscale=1.5)


def _cell_grid(values: dict[tuple[int, int], float]):
    cells = np.array(list(values.keys()), dtype=int)
    vals = np.array(list(values.values()), dtype=float)
    cell_min = cells.min(axis=0)
    cell_max = cells.max(axis=0)
    w = int(cell_max[0] - cell_min[0] + 1)
    h = int(cell_max[1] - cell_min[1] + 1)
    heat = np.full((w, h), np.nan)
    heat[cells[:, 0] - cell_min[0], cells[:, 1] - cell_min[1]] = vals
    return heat, cell_min, cell_max


def _extent(cell_min, cell_max, bin_size_m, origin_e, origin_n):
    return [
        cell_min[0] * bin_size_m + origin_e,
        (cell_max[0] + 1) * bin_size_m + origin_e,
        cell_min[1] * bin_size_m + origin_n,
        (cell_max[1] + 1) * bin_size_m + origin_n,
    ]


def _square_limits(xmin, xmax, ymin, ymax, pad=0.02):
    """A common square window (so every plot shares origin + extent under equal aspect)."""
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def _write_plots(
    candidates: pd.DataFrame,
    report_dir: Path,
    *,
    params: SelectionParams,
    view_counts: dict[tuple[int, int], int],
    grid: GroundGrid | None,
    footprints=None,
    spread_by_cell: dict[tuple[int, int], float] | None = None,
) -> None:
    selected = candidates[candidates["selected"]]
    bin_size_m = params.bin_size_m
    origin_e = grid.origin_e if grid is not None else 0.0
    origin_n = grid.origin_n if grid is not None else 0.0
    ref_e = float(candidates.loc[candidates.index.min(), "easting"])
    ref_n = float(candidates.loc[candidates.index.min(), "northing"])

    ex = [float(candidates["easting"].min()), float(candidates["easting"].max())]
    ny = [float(candidates["northing"].min()), float(candidates["northing"].max())]
    cell_sources = [view_counts, spread_by_cell]
    if footprints is not None and grid is not None:
        cell_sources.append({c: 0 for c in grid.mission_cells(footprints.values())})
    for cells in cell_sources:
        if not cells:
            continue
        arr = np.array(list(cells), dtype=int)
        ex[0] = min(ex[0], arr[:, 0].min() * bin_size_m + origin_e)
        ex[1] = max(ex[1], (arr[:, 0].max() + 1) * bin_size_m + origin_e)
        ny[0] = min(ny[0], arr[:, 1].min() * bin_size_m + origin_n)
        ny[1] = max(ny[1], (arr[:, 1].max() + 1) * bin_size_m + origin_n)
    xlim, ylim = _square_limits(ex[0], ex[1], ny[0], ny[1])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        candidates["easting"],
        candidates["northing"],
        s=1,
        c="lightgray",
        alpha=0.4,
        label="candidates",
    )
    if not selected.empty:
        sc = ax.scatter(
            selected["easting"],
            selected["northing"],
            s=12,
            c=selected["quality_score"],
            cmap="viridis",
            label="selected",
        )
        fig.colorbar(sc, ax=ax, label="quality_score")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Selected frames over candidate trajectory")
    _add_relative_axes(ax, ref_e, ref_n)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / "trajectory_map.png", dpi=120)
    plt.close(fig)

    if "selection_reason" in candidates.columns and not selected.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        _scatter_by_category(
            ax,
            candidates,
            "selection_reason",
            "Selection reason by location",
            ref_e,
            ref_n,
            xlim,
            ylim,
        )
        fig.tight_layout()
        fig.savefig(report_dir / "selection_reason_map.png", dpi=120)
        plt.close(fig)

    rejected = candidates[~candidates["selected"]]
    if "reject_reason" in candidates.columns and not rejected.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        _scatter_by_category(
            ax, rejected, "reject_reason", "Reject reason by location", ref_e, ref_n, xlim, ylim
        )
        fig.tight_layout()
        fig.savefig(report_dir / "reject_map.png", dpi=120)
        plt.close(fig)

    if view_counts:
        cells = np.array(list(view_counts.keys()), dtype=int)
        counts = np.array(list(view_counts.values()), dtype=int)
        cell_min = cells.min(axis=0)
        cell_max = cells.max(axis=0)
        grid_w = int(cell_max[0] - cell_min[0] + 1)
        grid_h = int(cell_max[1] - cell_min[1] + 1)
        heat = np.zeros((grid_w, grid_h), dtype=int)
        heat[cells[:, 0] - cell_min[0], cells[:, 1] - cell_min[1]] = counts

        fig, ax = plt.subplots(figsize=(10, 8))
        target = params.target_views_per_cell
        im = ax.imshow(
            heat.T,
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=0,
            vmax=max(target * 2, int(counts.max())),
            extent=[
                cell_min[0] * bin_size_m + origin_e,
                (cell_max[0] + 1) * bin_size_m + origin_e,
                cell_min[1] * bin_size_m + origin_n,
                (cell_max[1] + 1) * bin_size_m + origin_n,
            ],
        )
        cb = fig.colorbar(im, ax=ax, label=f"views per {bin_size_m:g}m cell (target={target})")
        cb.ax.axhline(target, color="white", linewidth=1.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
        _add_relative_axes(ax, ref_e, ref_n)
        ax.set_title("Frustum view count per ground cell (selected frames)")
        fig.tight_layout()
        fig.savefig(report_dir / "views_per_cell.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.arange(0, max(int(counts.max()) + 2, target + 2)) - 0.5
        ax.hist(counts, bins=bins, color="tab:blue", edgecolor="white")
        ax.axvline(target, color="tab:red", linestyle="--", label=f"target={target}")
        ax.set_xlabel("views per cell")
        ax.set_ylabel("# cells")
        ax.set_title("Distribution of frustum view count per ground cell")
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / "views_histogram.png", dpi=120)
        plt.close(fig)

    if spread_by_cell:
        heat, cmin, cmax = _cell_grid(spread_by_cell)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            heat.T,
            origin="lower",
            aspect="equal",
            cmap="magma",
            vmin=0,
            vmax=max(params.parallax_min_convergence_deg * 3, 30.0),
            extent=_extent(cmin, cmax, bin_size_m, origin_e, origin_n),
        )
        fig.colorbar(im, ax=ax, label="max convergence angle (deg)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
        _add_relative_axes(ax, ref_e, ref_n)
        ax.set_title("Per-cell 3D convergence angle")
        fig.tight_layout()
        fig.savefig(report_dir / "view_convergence.png", dpi=120)
        plt.close(fig)

    if footprints is not None and grid is not None and view_counts:
        mission = grid.mission_cells(footprints.values())
        target = params.target_views_per_cell
        status = {c: 0 for c in mission}
        for c, v in view_counts.items():
            status[c] = 2 if v >= target else 1
        if status:
            heat, cmin, cmax = _cell_grid({c: float(v) for c, v in status.items()})
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#fdae61", "#1a9850"])
            im = ax.imshow(
                heat.T,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=0,
                vmax=2,
                extent=_extent(cmin, cmax, bin_size_m, origin_e, origin_n),
            )
            cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
            cb.ax.set_yticklabels(["unseen", "under target", "at target"])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("easting (m)")
            ax.set_ylabel("northing (m)")
            _add_relative_axes(ax, ref_e, ref_n)
            ax.set_title("Mission region coverage vs selected footprints")
            fig.tight_layout()
            fig.savefig(report_dir / "footprint_union.png", dpi=120)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(candidates.index, candidates["quality_score"], color="0.8", linewidth=0.5)
    if not selected.empty:
        ax.scatter(
            selected.index,
            selected["quality_score"],
            s=8,
            c="tab:blue",
            label="selected",
        )
    ax.set_xlabel("FrameNumber")
    ax.set_ylabel("quality_score")
    ax.set_title("Quality vs frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "quality_vs_frame.png", dpi=120)
    plt.close(fig)


def write_selection_report(
    candidates: pd.DataFrame,
    run_dir: Path,
    constants: dict,
    *,
    health: SelectionHealth | None = None,
    view_counts: ViewCounts | None = None,
    coverage: CoverageMetrics | None = None,
    grid: GroundGrid | None = None,
    footprints: FootprintCache | None = None,
    cell_ground_z: dict[tuple[int, int], float] | None = None,
) -> None:
    report_dir = run_dir / "selection_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit_df(candidates)
    audit.to_csv(run_dir / "selection_audit.csv", index=True)

    params = params_from_constants(constants)
    if view_counts is None and footprints is not None and candidates["selected"].any():
        from scene_recon.selection import compute_view_counts

        view_counts = compute_view_counts(candidates, footprints)
    if coverage is None:
        coverage = coverage_metrics(
            view_counts or {}, params.target_views_per_cell, bin_size_m=params.bin_size_m
        )

    summary = build_summary(
        audit,
        constants,
        health=health,
        view_counts=view_counts,
        coverage=coverage,
    )

    spread_by_cell: dict[tuple[int, int], float] = {}
    if footprints is not None and grid is not None and view_counts and candidates["selected"].any():
        from scene_recon.selection_insight import compute_insight

        summary["insight"], spread_by_cell = compute_insight(
            candidates, footprints, view_counts, grid, params, cell_ground_z=cell_ground_z
        )

    (run_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _write_plots(
        candidates,
        report_dir,
        params=params,
        view_counts=view_counts or {},
        grid=grid,
        footprints=footprints,
        spread_by_cell=spread_by_cell,
    )

    health_data = summary.get("health")
    if health_data:
        if health_data["passed"]:
            para = health_data.get("parallax") or {}
            log.info(
                "selection health OK: %d frames, parallax satisfied %.0f%% of covered "
                "(%d/%d cells), view-count mission %.0f%%, max cluster %d",
                summary["n_selected"],
                100.0 * para.get("pct_covered_parallax_satisfied", 0.0),
                para.get("n_parallax_satisfied", 0),
                para.get("n_cells_covered", 0),
                100.0 * health_data["pct_mission_at_target"],
                health_data["max_cluster_size"],
            )
        else:
            log.error("selection health FAILED: %s", "; ".join(health_data["failures"]))

    coverage_data = summary.get("coverage", {})
    log.info(
        "selection: %d / %d frames, %d cells covered (mean %.1f views), %d spatial coverage warnings",
        summary["n_selected"],
        summary["n_candidates"],
        coverage_data.get("n_cells_covered", 0),
        coverage_data.get("mean_views_per_covered_cell", 0.0),
        len(summary["coverage_gaps"]),
    )
    for gap in summary["coverage_gaps"][:5]:
        log.warning(
            "coverage warning %.1fm between frames %d and %d",
            gap["gap_m"],
            gap["from_frame"],
            gap["to_frame"],
        )
