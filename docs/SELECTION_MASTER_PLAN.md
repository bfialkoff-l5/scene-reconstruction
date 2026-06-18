# Keyframe Selection — Master Plan (two-stage overlap selection)

Status: **design locked, ready to implement.** This is the authoritative selection
plan. It supersedes the greedy parallax objective in
`docs/SELECTION_OBJECTIVE_PLAN.md` (kept for history). The geometry half
(`docs/DTM_FRUSTUM_SELECTION_PLAN.md`: camera, terrain, ray-march, footprint, grid)
is unchanged and is the substrate this plan stands on.

## Objective

Pick keyframes from drone video so ODM produces a clean, fully-covered orthophoto.
ODM does global feature matching + bundle adjustment, so we do **not** need a
per-frame connectivity spine or a per-cell parallax optimizer. We need: usable
frames, even overlap across the whole site, and a hard budget cap when required.
Overlap and parallax are **emergent** properties of even sampling over the survey —
we measure them, we don't grind for them.

## What we learned the hard way

The greedy parallax objective + consecutive-overlap connectivity spine was a dead
end: on 0088 it gave **17% coverage, 1.8° median convergence, 21.7% parallax-
satisfied**, confined to a 182×127 m corner, and ran for tens of minutes at high
budget. Root causes: (1) the spine crawled through adjacent high-overlap frames and
ate the whole budget (coverage/parallax fill never ran); (2) high consecutive
overlap means tiny baselines = tiny convergence — connectivity was *fighting*
parallax; (3) the centroid-proxy gain barely used the parallax prior anyway.

The fix is not a better optimizer — it's deleting the optimizer. Simple overlap
spacing **on the frustum/DSM footprints** realizes the priors that the greedy threw
away.

## The design — two O(n) passes, no greedy

**Stage 0 — eligibility gate (keep from Phase 1).** Drop ineligible frames before
selection: relative-AGL takeoff/landing trim (`airborne_span`), footprint invalid,
missing quality. Reject reasons: `on_ground`, `invalid_footprint`,
`below_altitude`, `missing_quality_score`.

**Stage 1 — no-cap superset (the ideal set).** Walk eligible frames in time order;
keep the first; keep the next frame when its footprint **Jaccard** with the last
kept frame drops to ≤ `overlap_jaccard_target`. This adapts keyframe density to true
ground overlap via the DSM/frustum (higher altitude ⇒ bigger footprint ⇒ larger
spacing). Output: the overlap-complete superset.

**Stage 2 — thin to the hard cap (best of the best).** If the superset exceeds
`max_keyframes`, partition it into `max_keyframes` contiguous path-ordered bins and
keep the **highest-quality** frame in each bin. Uniform coverage is preserved by
construction; image quality is maximized per neighborhood; overlap/coverage degrade
gracefully as the cap shrinks — **parallax does not degrade** (thinning stays
spatially uniform).

**Verification (not optimization).** Compute per-cell convergence, coverage, and
view counts on the final set for the report and health. Parallax target
(≥3 views / ≥10°) is something we *check*, plus an optional future top-up knob.

## Evidence (0088, read-only probes /tmp/{trivial,priors,subselect}.py)

| selector | frames | coverage | median conv | parallax-sat |
|---|---|---|---|---|
| greedy (old, current code) | 500 | 17.3% | 1.8° | 21.7% |
| Stage 1 superset (T=0.5, no cap) | 3810 | 96.3% | 16.4° | 57.3% |
| + Stage 2 cap 500 (best-quality/bin) | 500 | 69.6% | 15.5° | 44.8% |
| + Stage 2 cap 1000 | 1000 | 79.2% | 16.6° | 50.9% |
| + Stage 2 cap 1500 | 1500 | 85.8% | 16.0° | 52.5% |

Convergence is ~16° at **every** cap: the cap trades coverage, never parallax.

## Locked decisions / dials

| # | Decision | Value |
|---|----------|-------|
| 1 | Overlap metric | footprint-cell Jaccard (frustum/DSM) |
| 2 | Stage-1 spacing dial | `overlap_jaccard_target` = 0.5 (≈ high overlap) |
| 3 | Stage-2 thinning | best-quality frame per path-ordered bin |
| 4 | Hard cap dial | `max_keyframes` (default TBD; superset ~3810 at T=0.5) |
| 5 | Eligibility | relative-AGL trim + valid footprint + quality present |
| 6 | Parallax / convergence | **verification metric**, not an objective |
| 7 | Connectivity | not enforced per-frame (ODM matches globally); reported only |

Open: final default `max_keyframes` (depends on ODM image-count tolerance) and
whether `T` is exposed on the CLI. Decide during/after the first end-to-end ODM run.

## Implementation plan — with hygiene

Geometry modules untouched: `camera.py`, `geometry/*`, `selection/grid.py`.

### Keep (assets)
- `selection/footprint.py`: `footprint_jaccard` (Stage 1), `compute_view_counts`,
  `recount_views`, `translation_m`, `rotation_deg`, `assign_bins`,
  `save/load_footprints`, `FootprintCache`.
- `selection/selector.py`: `airborne_span` + the eligibility gating in
  `select_keyframes`; `_write_footprint_columns`.
- `selection/parallax.py` measurement half: `max_convergence_deg`, `cell_direction`,
  `build_parallax_context`, `convergence_by_cell`, `parallax_satisfied`,
  `parallax_metrics`, `ParallaxMetrics`, `mission_cell_ground_z`,
  `approx_cell_ground_z`, `cell_viewers_from_selection`.
- `selection/metrics.py`: `coverage_metrics`, `CoverageMetrics`, `GapInfo`,
  `iter_selection_gaps`.
- `selection/cluster.py`: `max_local_density` (health reporting only).
- `selection_insight.py` (convergence-based), `selection_report.py`,
  `selection_health.py`, `build.py`, `cli.py` — adapt, don't rewrite.

### Delete (dead greedy objective + connectivity spine)
- `selector.py`: `_build_overlap_chain`, `_best_in_window`, `_select_frame`,
  `_reject_reason`, `_frame_windows`, `_nearest_selected_before`, `SelectionState`,
  `_rebuild_cell_viewers`, and the multi-phase body of `_select_keyframes_inner`.
- `metrics.py`: `score_candidate`, `score_reject_candidate`, `coverage_selects`,
  `cluster_penalty`, `CandidateMetrics`, `write_debug_columns` (greedy score cols).
- `parallax.py` greedy half: `ParallaxState`, `CellParallax`, `_candidate_helps`,
  `parallax_gain_cells`, `record_parallax_views`, `rebuild_parallax_state`,
  `count_gain_cells`, `_updated_max_conv`.
- `cluster.py`: `BallIndex`, `filter_to_main_component`.
- `params.py`: `score_quality_weight`, `score_coverage_weight`,
  `score_novelty_weight`, `min_coverage_gain_ratio`, `min_coverage_gain_cells`,
  `connection_radius_m`, `main_component_ratio`, `max_per_cluster`,
  and the now-unused greedy parallax thresholds if not reused for verification.
  Update `as_constants`/`params_from_constants`, `__init__` exports, and any tests
  referencing deleted names.

### Repurpose
- `overlap_jaccard_threshold` → `overlap_jaccard_target` (Stage-1 dial), default 0.5.
- Keep `max_keyframes`, `ground_trim_rise_m`, `parallax_min_views`,
  `parallax_min_convergence_deg` (verification), `target_views_per_cell`
  (reporting), `bin_size_m`, terrain params, `max_motion_gap_m`/`coverage_warn_m`
  (warn-only reporting).

### New selector (`select_keyframes`)
Eligibility (as today) → Stage-1 superset → Stage-2 cap. Write columns for the
existing visuals/audit:
- `selected = True` for final keyframes; `selection_reason = "keyframe"`.
- superset frames dropped by the cap → `reject_reason = "thinned_by_budget"`.
- eligible frames not in superset (too much overlap) → `reject_reason =
  "redundant_overlap"`.
- ineligible → existing `on_ground` / `invalid_footprint` / `below_altitude` /
  `missing_quality_score`.
- populate footprint columns + per-frame `quality_score` (already present).

### Wiring (pipeline parity)
- `build.py`: unchanged flow (score → footprints → grid → terrain →
  `mission_cell_ground_z` → `select_keyframes` → `coverage_metrics` +
  `parallax_metrics` → `assess_selection` → `write_selection_report` → ODM export).
  Remove greedy params from the call path.
- `cli.py`: drop greedy options (score weights, cluster cap, coverage-gain); add
  `--overlap-target` (optional) and keep `--max-keyframes`. `report --force` must
  re-run the new selector from the footprint cache, same as today.

### Visuals (must remain identical set)
Regenerate the existing plots from the final selection: `trajectory_map.png`,
`selection_reason_map.png`, `reject_map.png`, `views_per_cell.png`,
`views_histogram.png`, `view_convergence.png`, `footprint_union.png`,
`quality_vs_frame.png`. `reject_map` stays meaningful via the new reject reasons
above. `view_convergence` uses the kept convergence measurement. Axes stay
start-relative metres.

### Logging
INFO on completion: eligible count, superset size, cap applied (superset→N),
final selected, coverage %, median convergence, parallax-satisfied %, overlap
target. WARN (informational only): large motion gaps, view-count below legacy
target, dense clusters. No silent budget exhaustion.

### Health (honest gates, not tuned-to-pass)
Hard-fail on: zero frames; coverage % below a floor calibrated to the achievable
(Stage-1 ≈ full; capped degrades predictably). Report convergence/parallax-sat
against the ≥3/≥10° target as measured truth. Do **not** lower a threshold to force
a pass.

### Tests
- Unit: Stage-1 superset spacing on a synthetic strip (overlap ≤ target between
  consecutive kept); Stage-2 best-quality-per-bin (uniform bins, max-quality pick).
- End-to-end on a synthetic multi-pass fixture asserting coverage breadth and that
  median convergence is materially above zero (guards the regression that green
  unit tests previously missed).
- Delete/update tests bound to removed greedy symbols.

## Acceptance criteria (0088, then a 2nd flight)
- Selected spans ~full eligible E/N extent (not a corner).
- At the chosen cap: coverage ≥ ~65%, median convergence ≥ ~12°, parallax-sat
  ≥ ~40% (Stage-1 no-cap ≈ 96% / 16° / 57%).
- 0 ground/sky frames selected; one overlap-connected set (reported).
- Wall time < ~30 s for select+health; full test suite green incl. new guards.
- **Final gate: the ODM orthophoto visibly covers the site without large holes.**

## Validation harness
Reuse `/tmp/subselect.py` (Stage-1 + Stage-2 + metrics) after each change; full run
via Docker `./scripts/build.sh <slug>` then `./scripts/run_odm.sh`. Unit tests:
`PYTHONPATH=src ./.venv/bin/python -m pytest tests/ -q` (runtime image has no
pytest).

## Progress tracker
- [x] Params cleanup + repurpose `overlap_jaccard_target` (default 0.5)
- [x] New two-stage `select_keyframes`; deleted greedy/spine code
- [x] Trimmed `parallax.py`/`metrics.py`/`cluster.py` to the kept surface; fixed `__init__`
- [x] Wired build.py + cli.py (`--overlap-target`, `--max-keyframes`); `report --force` re-runs the new selector
- [x] Plot set unchanged; reject_map now keyed on `thinned_by_budget` / `redundant_overlap` / `on_ground` / `invalid_footprint`
- [x] Honest health gates (hard-fail: zero frames, mission coverage < `min_pct_mission_covered`=0.5) + INFO logging; convergence/parallax reported only
- [x] Tests (Stage-1 spacing, Stage-2 best-quality-per-bin, e2e multi-pass spread guard); full suite 57 green
- [x] 0088 select+health validation — read-only harness `/tmp/validate.py`:
      no-cap=3810f/96.3%/16.4°/57.3%, cap500=500f/69.6%/15.5°/44.8%,
      cap1500=1500f/85.8%/16.0°/52.5%; full 463×388 m extent; 0 ground frames; ~6.4 s
- [ ] Full build → ODM → orthophoto eyeball
- [ ] Second-flight validation
