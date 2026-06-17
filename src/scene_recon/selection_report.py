from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scene_recon.selection import (
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
        if view_counts is None:
            from scene_recon.selection import compute_view_counts

            view_counts = compute_view_counts(candidates, params) if not selected.empty else {}
        coverage = coverage_metrics(view_counts or {}, params.target_views_per_cell)

    quality = selected["quality_score"].astype(float)
    summary = {
        **constants,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "reject_counts": reject_counts,
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


def _write_plots(
    candidates: pd.DataFrame,
    report_dir: Path,
    *,
    params: SelectionParams,
    view_counts: dict[tuple[int, int], int],
) -> None:
    selected = candidates[candidates["selected"]]
    bin_size_m = params.bin_size_m

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
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Selected frames over candidate trajectory")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / "trajectory_map.png", dpi=120)
    plt.close(fig)

    if view_counts:
        cells = np.array(list(view_counts.keys()), dtype=int)
        counts = np.array(list(view_counts.values()), dtype=int)
        cell_min = cells.min(axis=0)
        cell_max = cells.max(axis=0)
        grid_w = int(cell_max[0] - cell_min[0] + 1)
        grid_h = int(cell_max[1] - cell_min[1] + 1)
        grid = np.zeros((grid_w, grid_h), dtype=int)
        grid[cells[:, 0] - cell_min[0], cells[:, 1] - cell_min[1]] = counts

        fig, ax = plt.subplots(figsize=(10, 8))
        target = params.target_views_per_cell
        im = ax.imshow(
            grid.T,
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=0,
            vmax=max(target * 2, int(counts.max())),
            extent=[
                cell_min[0] * bin_size_m,
                (cell_max[0] + 1) * bin_size_m,
                cell_min[1] * bin_size_m,
                (cell_max[1] + 1) * bin_size_m,
            ],
        )
        cb = fig.colorbar(im, ax=ax, label=f"views per {bin_size_m:g}m cell (target={target})")
        cb.ax.axhline(target, color="white", linewidth=1.5)
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
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
) -> None:
    report_dir = run_dir / "selection_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit_df(candidates)
    audit.to_csv(run_dir / "selection_audit.csv", index=True)

    params = params_from_constants(constants)
    if view_counts is None and candidates["selected"].any():
        from scene_recon.selection import compute_view_counts

        view_counts = compute_view_counts(candidates, params)
    if coverage is None:
        coverage = coverage_metrics(view_counts or {}, params.target_views_per_cell)

    summary = build_summary(
        candidates,
        constants,
        health=health,
        view_counts=view_counts,
        coverage=coverage,
    )
    (run_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _write_plots(candidates, report_dir, params=params, view_counts=view_counts or {})

    health_data = summary.get("health")
    if health_data:
        if health_data["passed"]:
            log.info(
                "selection health OK: %d frames, %d / %d cells at target views (%.0f%%), max cluster %d",
                summary["n_selected"],
                health_data["n_cells_at_target"],
                health_data["n_cells_covered"],
                100.0 * health_data["pct_cells_at_target"],
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
        coverage_data.get("mean_views_per_cell", 0.0),
        len(summary["coverage_gaps"]),
    )
    for gap in summary["coverage_gaps"][:5]:
        log.warning(
            "coverage warning %.1fm between frames %d and %d",
            gap["gap_m"],
            gap["from_frame"],
            gap["to_frame"],
        )
