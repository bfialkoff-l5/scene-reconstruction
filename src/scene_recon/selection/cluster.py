from __future__ import annotations

import numpy as np
import pandas as pd

from scene_recon.selection.params import SelectionParams


class BallIndex:
  """Spatial ball queries for cluster-density caps."""

  def __init__(
      self,
      eastings: np.ndarray,
      northings: np.ndarray,
      index_pos: dict[int, int],
      radius_m: float,
  ) -> None:
      self.eastings = eastings
      self.northings = northings
      self.index_pos = index_pos
      self.radius_sq = radius_m * radius_m

  def count_neighbors(self, idx: int, selected: set[int]) -> int:
      if not selected:
          return 0
      p = self.index_pos.get(idx)
      if p is None:
          return 0
      e0 = self.eastings[p]
      n0 = self.northings[p]
      count = 0
      for s in selected:
          sp = self.index_pos.get(s)
          if sp is None:
              continue
          de = self.eastings[sp] - e0
          dn = self.northings[sp] - n0
          if de * de + dn * dn <= self.radius_sq:
              count += 1
      return count

  def would_exceed_cap(
      self,
      idx: int,
      selected: set[int],
      cap: int,
      ball_size: dict[int, int],
  ) -> bool:
      my_count = self.count_neighbors(idx, selected)
      if 1 + my_count > cap:
          return True
      if my_count == 0:
          return False
      p = self.index_pos.get(idx)
      if p is None:
          return False
      e0 = self.eastings[p]
      n0 = self.northings[p]
      for s in selected:
          sp = self.index_pos.get(s)
          if sp is None:
              continue
          de = self.eastings[sp] - e0
          dn = self.northings[sp] - n0
          if de * de + dn * dn <= self.radius_sq:
              if ball_size.get(s, 0) + 1 > cap:
                  return True
      return False

  def record_selection(self, idx: int, selected: set[int], ball_size: dict[int, int]) -> None:
      p = self.index_pos.get(idx)
      if p is None:
          ball_size[idx] = 1
          return
      e0 = self.eastings[p]
      n0 = self.northings[p]
      my_neighbors = 0
      for s in selected:
          sp = self.index_pos.get(s)
          if sp is None:
              continue
          de = self.eastings[sp] - e0
          dn = self.northings[sp] - n0
          if de * de + dn * dn <= self.radius_sq:
              ball_size[s] = ball_size.get(s, 1) + 1
              my_neighbors += 1
      ball_size[idx] = 1 + my_neighbors

  def rebuild_ball_sizes(self, selected: set[int]) -> dict[int, int]:
      fresh: dict[int, int] = {}
      members = [s for s in selected if s in self.index_pos]
      for s in members:
          sp = self.index_pos[s]
          es, ns = self.eastings[sp], self.northings[sp]
          count = 0
          for other in members:
              op = self.index_pos[other]
              de = self.eastings[op] - es
              dn = self.northings[op] - ns
              if de * de + dn * dn <= self.radius_sq:
                  count += 1
          fresh[s] = count
      return fresh

  def max_ball_size(self, selected: set[int]) -> tuple[int, list[int]]:
      if not selected:
          return 0, []
      members = [s for s in selected if s in self.index_pos]
      best_count = 0
      best_members: list[int] = []
      for s in members:
          sp = self.index_pos[s]
          es, ns = self.eastings[sp], self.northings[sp]
          in_ball = [
              members[j]
              for j, other in enumerate(members)
              if (self.eastings[self.index_pos[other]] - es) ** 2
              + (self.northings[self.index_pos[other]] - ns) ** 2
              <= self.radius_sq
          ]
          if len(in_ball) > best_count:
              best_count = len(in_ball)
              best_members = in_ball
      return best_count, best_members


def max_local_density(
    indices: list[int],
    out: pd.DataFrame,
    params: SelectionParams,
) -> tuple[int, list[int]]:
    if not indices:
        return 0, []
    eastings = out.loc[indices, "easting"].astype(float).to_numpy()
    northings = out.loc[indices, "northing"].astype(float).to_numpy()
    index_pos = {idx: i for i, idx in enumerate(indices)}
    ball = BallIndex(eastings, northings, index_pos, params.cluster_radius_m)
    return ball.max_ball_size(set(indices))


def spatial_components(
    indices: list[int],
    out: pd.DataFrame,
    radius_m: float,
) -> list[list[int]]:
    if not indices:
        return []
    sorted_idx = sorted(int(i) for i in indices)
    n = len(sorted_idx)
    eastings = out.loc[sorted_idx, "easting"].astype(float).to_numpy()
    northings = out.loc[sorted_idx, "northing"].astype(float).to_numpy()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    radius_sq = radius_m * radius_m
    for i in range(n):
        for j in range(i + 1, n):
            de = eastings[i] - eastings[j]
            dn = northings[i] - northings[j]
            if de * de + dn * dn <= radius_sq:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(sorted_idx[i])
    return sorted(groups.values(), key=len, reverse=True)


def filter_to_main_component(
    selected: set[int],
    out: pd.DataFrame,
    params: SelectionParams,
) -> set[int]:
    if len(selected) < 2:
        return set(selected)
    components = spatial_components(list(selected), out, params.connection_radius_m)
    if not components:
        return set(selected)
    largest = components[0]
    if len(largest) / len(selected) < params.main_component_ratio:
        return set(selected)
    return set(largest)
