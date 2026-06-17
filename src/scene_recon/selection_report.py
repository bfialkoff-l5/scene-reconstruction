from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scene_recon.frame_select import translation_m
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
        for idx in audit.index:
            de = float(audit.loc[idx, "easting"]) - se
            dn = float(audit.loc[idx, "northing"]) - sn
            dist = np.hypot(de, dn)
            if audit.loc[idx, "selected"]:
                dist = dist[dist > 0]
            audit.loc[idx, "dist_to_nearest_selected_m"] = (
                float(dist.min()) if len(dist) else pd.NA
            )

    return audit


def build_summary(
    candidates: pd.DataFrame,
    constants: dict,
    *,
    health: SelectionHealth | None = None,
) -> dict:
    selected = candidates[candidates["selected"]]
    reject_counts = (
        candidates.loc[~candidates["selected"], "reject_reason"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict()
    )
    occupied_bins = candidates.groupby(["cell_x", "cell_y"], observed=True).ngroups
    selected_bins = (
        selected.groupby(["cell_x", "cell_y"], observed=True).ngroups if not selected.empty else 0
    )

    coverage_gaps: list[dict] = []
    temporal_gaps: list[dict] = []
    if len(selected) >= 2:
        sel = selected.sort_index()
        prev_idx = None
        for idx, row in sel.iterrows():
            if prev_idx is not None:
                gap_frames = int(idx) - int(prev_idx)
                gap_m = translation_m(row, sel.loc[prev_idx])
                temporal_gaps.append(
                    {
                        "from_frame": int(prev_idx),
                        "to_frame": int(idx),
                        "gap_frames": gap_frames,
                        "gap_m": round(gap_m, 2),
                    }
                )
                warn_m = constants.get("coverage_warn_m", 15.0)
                if gap_m > warn_m:
                    coverage_gaps.append(
                        {
                            "from_frame": int(prev_idx),
                            "to_frame": int(idx),
                            "gap_m": round(gap_m, 2),
                        }
                    )
            prev_idx = idx

    quality = selected["quality_score"].astype(float)
    summary = {
        **constants,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "n_bins_occupied": int(occupied_bins),
        "n_bins_with_selection": int(selected_bins),
        "reject_counts": reject_counts,
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


def _write_plots(candidates: pd.DataFrame, report_dir: Path, *, bin_size_m: float) -> None:
    selected = candidates[candidates["selected"]]
    origin_e = candidates["easting"].min()
    origin_n = candidates["northing"].min()

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
    xmin, xmax = candidates["easting"].min(), candidates["easting"].max()
    ymin, ymax = candidates["northing"].min(), candidates["northing"].max()
    for x in np.arange(origin_e, xmax + bin_size_m, bin_size_m):
        ax.axvline(x, color="0.9", linewidth=0.5)
    for y in np.arange(origin_n, ymax + bin_size_m, bin_size_m):
        ax.axhline(y, color="0.9", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Spatial bins and selected frames")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / "bins_map.png", dpi=120)
    plt.close(fig)

    heat, xedges, yedges = np.histogram2d(
        candidates["easting"],
        candidates["northing"],
        bins=[
            max(1, int(np.ceil((xmax - xmin) / bin_size_m))),
            max(1, int(np.ceil((ymax - ymin) / bin_size_m))),
        ],
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        heat.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="hot",
    )
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Candidate density per bin")
    fig.tight_layout()
    fig.savefig(report_dir / "bins_heatmap.png", dpi=120)
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
) -> None:
    report_dir = run_dir / "selection_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit_df(candidates)
    audit.to_csv(run_dir / "selection_audit.csv", index=True)

    summary = build_summary(candidates, constants, health=health)
    (run_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _write_plots(candidates, report_dir, bin_size_m=constants.get("bin_size_m", 5.0))

    health_data = summary.get("health")
    if health_data:
        if health_data["passed"]:
            log.info(
                "selection health OK: %d frames, max temporal gap %d, max cluster %d",
                summary["n_selected"],
                health_data["max_temporal_gap"],
                health_data["max_cluster_size"],
            )
        else:
            log.error("selection health FAILED: %s", "; ".join(health_data["failures"]))

    log.info(
        "selection: %d / %d frames, %d bins occupied, %d spatial coverage warnings",
        summary["n_selected"],
        summary["n_candidates"],
        summary["n_bins_occupied"],
        len(summary["coverage_gaps"]),
    )
    for gap in summary["coverage_gaps"][:5]:
        log.warning(
            "coverage warning %.1fm between frames %d and %d",
            gap["gap_m"],
            gap["from_frame"],
            gap["to_frame"],
        )
