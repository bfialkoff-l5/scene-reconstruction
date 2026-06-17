from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scene_recon.frame_select import SelectionParams, group_local_clusters


@dataclass
class SelectionHealth:
    passed: bool
    failures: list[str]
    max_temporal_gap: int
    worst_temporal_gap: dict | None
    max_cluster_size: int
    largest_cluster: dict | None
    n_selected: int

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "max_temporal_gap": self.max_temporal_gap,
            "worst_temporal_gap": self.worst_temporal_gap,
            "max_cluster_size": self.max_cluster_size,
            "largest_cluster": self.largest_cluster,
            "n_selected": self.n_selected,
        }


class SelectionFailed(Exception):
    def __init__(self, health: SelectionHealth) -> None:
        self.health = health
        super().__init__("; ".join(health.failures))


def assess_selection(candidates: pd.DataFrame, params: SelectionParams) -> SelectionHealth:
    selected = candidates[candidates["selected"]].sort_index()
    failures: list[str] = []
    n_selected = len(selected)

    max_temporal_gap = 0
    worst_temporal_gap = None
    if n_selected >= 2:
        prev = None
        for idx in selected.index:
            idx = int(idx)
            if prev is not None:
                gap = idx - prev
                if gap > max_temporal_gap:
                    max_temporal_gap = gap
                    worst_temporal_gap = {
                        "from_frame": prev,
                        "to_frame": idx,
                        "gap_frames": gap,
                    }
            prev = idx

    if n_selected == 0:
        failures.append("no frames selected")
    elif max_temporal_gap > params.max_frame_gap:
        failures.append(
            f"max temporal gap {max_temporal_gap} exceeds limit {params.max_frame_gap}"
        )

    clusters = group_local_clusters([int(i) for i in selected.index], candidates, params)
    max_cluster_size = max((len(c) for c in clusters), default=0)
    largest_cluster = None
    if clusters:
        largest = max(clusters, key=len)
        if len(largest) == max_cluster_size:
            largest_cluster = {
                "size": len(largest),
                "frame_numbers": largest[:10],
            }

    if max_cluster_size > params.max_per_cluster:
        failures.append(
            f"spatial cluster size {max_cluster_size} exceeds limit {params.max_per_cluster}"
        )

    return SelectionHealth(
        passed=not failures,
        failures=failures,
        max_temporal_gap=max_temporal_gap,
        worst_temporal_gap=worst_temporal_gap,
        max_cluster_size=max_cluster_size,
        largest_cluster=largest_cluster,
        n_selected=n_selected,
    )
