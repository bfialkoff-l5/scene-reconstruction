# Selection Objective Redesign — Reconstructability over Coverage

> **SUPERSEDED (2026-06-18) by `docs/SELECTION_MASTER_PLAN.md`.** The greedy
> parallax objective + connectivity spine described below was implemented and
> measured on 0088 at **17% coverage / 1.8° convergence** — a regression. It is
> replaced by a two-stage overlap-spacing + best-of-best thinning design. Kept here
> for history and for the Phase-0/1 findings (eligibility gate, calibration) that
> remain valid.

Status: **superseded — see master plan**. This doc captures what keyframe selection
*should* optimize for, the decisions we've locked, the ones still open, and the
staged plan to get there. It supersedes the objective described in
`DTM_FRUSTUM_SELECTION_PLAN.md` (the geometry half of that doc — camera, terrain,
ray-march, footprint, grid — stays; only the *objective* changes).

---

## The problem

We upgraded the **geometry** (flat-plane → DTM ray-marched footprints) but never
changed the **objective function**. The selector still optimizes a 2D
mapping/mowing goal:

> cover every ground cell with N views, keep frames spaced apart, stay under budget.

That is the wrong objective for photogrammetric 3D reconstruction (what ODM
consumes). Evidence from the `0088_20260122_eitan_1` run (audit log, 13,856
frames, 500 selected):

- **Count is a blind proxy for multi-view.** ~15k cells sit at >2× the view-count
  target (mean 11 views/cell) while the view-diversity map shows poor angular
  spread. The scorer saturates cells on redundant same-angle frames, then rejects
  the off-angle frame that carried the only baseline (`coverage_saturated` 3327,
  `low_coverage_gain` 1603 — both 100% valid footprints, median 627 cells, median
  |pitch| 35°: good oblique geometry, thrown away).
- **Connectivity is a frame-gap, not overlap.** `temporal_chain` keeps continuity
  via "≤90-frame hops" and lets coverage/cluster rejects sever it. Result: a
  1,796-frame hole over the climb-out (frames ~30→1825). The build logged
  `temporal gap 1796 frames exceeds max_frame_gap 90 (warn only)` and shrugged.
- **No eligibility gate.** `temporal_seed`/`temporal_chain` are coverage- and
  altitude-blind, so the first 8 selected frames (17–29) are pre-takeoff ground
  shots (altamsl 145.2, pitch −1°).
- **Procedural phases fight each other.** seed vs chain vs coverage-fill vs
  cluster-cap each apply ad-hoc rules; there is no single notion of "is this frame
  worth a keyframe slot."

---

## What we actually want (priority order)

**P0 — Usable frames only (a gate, not a goal).** Airborne (not sitting on the
ground), scene visible (footprint valid — not sky/horizon), sharp/exposed.
Ineligible frames cannot be selected by *any* phase.

**P1 — One connected reconstruction.** Selected frames must form an unbroken
*overlap* chain from first-usable to last-usable frame. Overlap = shared footprint
(Jaccard), not frame index. Never open a gap that breaks matching; if imagery
genuinely can't sustain overlap, raise it loudly instead of silently bridging.

**P2 — Parallax-diverse multi-view, not view count.** A cell is "done" when it is
seen by enough frames whose viewing directions span a real convergence angle — not
when a counter hits N. This is the highest-leverage change and the one that makes
the orthophoto correct (triangulation needs baseline).

**P3 — Budget & spacing as tie-breakers.** Under the above, prefer the
sharpest/best-geometry frame per slot and respect `max_keyframes`. `cluster_cap`
and `pose_novelty` become penalties among otherwise-redundant frames, never hard
rejects that sever connectivity or diversity.

Reframed definition of "coverage done": **every cell is parallax-reconstructable
AND the selection is one connected component** — not "every cell has N views."

---

## Decisions

### Locked

- **Overlap metric = footprint-cell Jaccard.** Cheap, already have the cells. Used
  for the P1 connectivity chain (overlap between consecutive keyframes).
- **Parallax metric = 3D convergence angle.** Per cell, for each viewing frame take
  the direction `(camera_position − cell_centroid_3d)`; the cell's parallax is the
  **max pairwise angle** between those directions. Robust for near-nadir aerial
  (azimuth spread is ill-defined when cameras look straight down). Cell centroid 3D
  = grid cell center (E,N) + DTM elevation.
  - **Provisional target:** ≥3 views with max pairwise convergence ≥10°
    (standard aerial photogrammetry). **Exact angle + count are a calibration knob
    measured in Phase 0** against the achievable distribution before locking.
- **Sharpness/quality gate = reuse existing** `quality_score`/`sharpness` gate. No
  new knob.
- **Grounded-frame gate = relative-AGL takeoff/landing trim, auto-detected knee.**
  Drop *stationary on-the-ground* frames (pre-takeoff / post-landing) without
  dropping useful *low-altitude takeoff/climb* frames. `AGL = altamsl −
  DTM_elevation(camera)`. The resting (head/tail) segment and the takeoff/landing
  knee are detected **statistically from the AGL profile** — no hand-picked margin.
  Measuring the rise *relative to resting* cancels the `datum_offset_m`
  uncertainty, so it's stable even if absolute heights are wrong. Keeps low
  climbing/descending frames; drops only the static ground segments. Pairs with the
  footprint-valid gate (which independently drops pitched-up sky frames during
  climb). The audit/poses CSV already carries everything needed (`altamsl`, pose
  E/N) to detect the knee — can be prototyped on the existing run.

---

## Staged plan

Geometry modules (`camera`, `geometry/{terrain,raymarch,extrinsics,footprint}`,
`selection/grid`) are correct and unchanged. The objective lives in
`selection/{selector,metrics,params}` and *promotes* the convergence/connectivity
math currently sitting in `selection_insight` from report decoration into
first-class criteria.

### Phase 0 — Instrument the objective (no behavior change)
Turn what we care about into computed acceptance metrics on the **current** run so
every later change is judged against a scoreboard:
- connectivity: is the selected set one Jaccard-overlap-connected component? size
  of the largest overlap break.
- per-cell convergence-angle distribution (promote `selection_insight` math; switch
  diversity from azimuth-spread to 3D convergence angle).
- usable-frame audit: how many selected are airborne / footprint-valid / sharp.
- **Calibrate the parallax target** here: read the achievable convergence
  distribution and set the ≥k / ≥θ° threshold at a percentile we can actually hit.

### Phase 1 — Eligibility gate (P0)
Pre-filter marking frames ineligible: relative-AGL takeoff/landing trim
(auto-detected knee) + footprint-valid + quality floor. Forbid every phase from
selecting ineligible frames. Kills the ground-shot seeds. Validate the knee
detector against the existing `0088` AGL profile before wiring it in.

### Phase 2 — Connectivity-first backbone (P1)
Replace "chain by frame-gap hops" with "chain by overlap": walk forward selecting
the next eligible frame that keeps footprint Jaccard with the last keyframe above
a threshold; never skip a gap that drops overlap below it. Surface unrecoverable
overlap loss as a hard warning/error. Forces the usable ascent back in and
guarantees a single connected component.

### Phase 3 — Parallax-aware coverage gain (P2)
Redefine "gain" from "cells brought to count target" to "cells brought toward the
**parallax** target": a candidate scores for new cells **and** for adding a new
angular contribution to an existing under-diverse cell (convergence angle below
target). Reuse the Phase 0 convergence machinery inside the scorer. Removes the
"reject the only off-angle frame" pathology.

### Phase 4 — Demote spacing/cluster (P3)
`spatial_cluster_cap` and `low_pose_novelty` become score penalties applied only
among redundant candidates, not hard gates. Budget allocates remaining slots to
highest parallax-gain + connectivity frames.

### Phase 5 — Recalibrate & validate
Run on `0088` + one more flight. **Acceptance bar:**
- one connected component (no overlap break);
- ascent present, or provably unusable (all-sky) and flagged;
- median per-cell convergence angle ≥ target on covered cells;
- zero grounded/sky frames selected;
- a sane orthophoto.

---

## Phase 0 findings (0088 run, read-only `tmp/phase0.py`)

The scoreboard, measured on the current count-greedy selection (500 frames):

**Usable frames.** Selected AGL ranges 0.6 → 105.8 m (med 91). 13 selected sit
below 10 m AGL — the 8 grounded seeds (17–29) at **0.6 m**, plus ~5 genuine
mid-flight low passes. footprint_valid is already 500/500 and quality is healthy
(med 0.838 vs pool 0.696, only 2 below pool p10). ⇒ the *only* eligibility problem
is grounded frames; the footprint-valid + existing quality gates handle the rest.

**Connectivity.** Consecutive footprint Jaccard: med 0.40, but **2 zero-overlap
hard breaks** — the 1796-frame ascent hole (29→1825) and a 48-frame gap
(10172→10220). Connected components: one component only at "any shared cell"; at
**Jaccard ≥ 0.05 the 8 grounded frames split off as a disconnected island**
(490 + 8 + singletons). ⇒ a connectivity threshold of **τ ≈ 0.05–0.10** is the
meaningful operating point; the grounded island and the zero-overlap breaks are the
defects to fix.

**Parallax (the calibration).** Views/cell med 6 (target was 5), p90 30, max 60 —
heavy over-coverage. Max-pairwise convergence angle: med 18°, but **p25 = 4.9°,
p10 = 1.6°** — a large tail of degenerate same-angle redundancy despite many views.
Feasibility of "≥k views AND ≥θ convergence" among covered cells:

| k \ θ | 5° | 10° | 15° | 20° |
|-------|----|-----|-----|-----|
| 3 | 82.8% | **69.4%** | 61.0% | 53.4% |
| 4 | 88.6% | 75.1% | 66.1% | 58.3% |
| 5 | 92.6% | 79.9% | 70.4% | 62.3% |

⇒ **≥3 views / ≥10°** is the sweet spot: met by ~69% of covered cells *today*, so
it's achievable but leaves a real ~31% gap — exactly the headroom the redesign
reclaims by trading redundant same-angle views (median 6/cell) for diverse ones.
θ=5° is trivially met (83%); θ=20° is aggressive (53%).

**Takeoff/landing knee.** Relative-AGL detector (resting 0.4 m, auto δ=2.0 m,
sustained-airborne over 60-frame window): takeoff knee = **frame 700** (4.3 m AGL,
700-frame head trimmed), landing knee = **frame 12626** (1229-frame tail trimmed).
It removes exactly the 8 grounded seeds (17–29) **while keeping the climb-out
(frames 700→2400) and the mid-flight low passes** — validating the relative-AGL
trim over a blunt floor, which would have wrongly killed the legitimate low passes.

---

## Progress tracker

- [x] Phase 0 — instrumented connectivity + convergence + usable-frame metrics; parallax target calibrated to ≥3 views / ≥10°
- [x] Phase 1 — eligibility gate: relative-AGL trim (`airborne_span`) + existing footprint-valid/quality gates. On 0088: 1881 frames rejected `on_ground`, min selected frame 17→648, the 8 grounded ground-shots eliminated, still 500 keyframes. 53/53 tests pass.
- [x] Phase 2 — overlap-driven connectivity backbone (`overlap_seed`/`overlap_chain`, `footprint_jaccard`, `overlap_jaccard_threshold=0.05`). On 0088: all 500 keyframes are chain frames; **0 zero-overlap breaks** (was 2); single component @ τ; climb 700–2400 has 286 selected frames (was 0 in ascent gap). Chain min Jaccard 0.052. One large temporal hop remains (1256→8795, jaccard 0.052) where no intermediate frame met τ against the anchor — logged path for Phase 3/4 tuning. 52/52 tests pass.
- [x] Phase 3 — parallax-aware coverage gain (`parallax_gain_cells`, incremental `ParallaxState`, centroid proxy, precomputed `ParallaxContext`). Performance fix: dropped pandas hot-path + O(k²) rescan + full parallax on reject pass → **40.8 s** on 0088 (was ~40 min / hung). On 0088: 416 selected (chain broke at 1256 — same footprint gap as Phase 2's logged hop, no forward Jaccard ≥ τ within 90 frames); **0 chain-order overlap breaks**, **0 ground frames**, single component; parallax-satisfied 4750/24830 covered cells (19.1%, vs 69.4% count-blind baseline on old 500-frame set — expected drop while P2 still fills by count in reject labels only). 55/55 tests pass.
- [x] Phase 4 — cluster/novelty demoted to score tie-breakers: `coverage_selects` is parallax-gain-only; cluster cap is a soft score penalty, not a hard skip in `_best_in_window`. On 0088: **432/500** in **37.5 s** (+16 frames vs Phase 3).
- [x] Chain long-hop bridge — when no Jaccard ≥ τ within `max_frame_gap`, scan forward for earliest overlap (restores Phase-2 connectivity). On 0088: **500/500** in **23.7 s**, 0 chain-order breaks, 1 logged long hop (1256→8795). 55/55 tests pass.
- [x] Phase 5 — health/report recalibrated for parallax objective: `ParallaxMetrics` + `min_pct_covered_parallax_satisfied=0.15`; view-count mission target, cluster cap, and motion gap demoted to warnings. On 0088: **500/500**, **health.passed**, **21.7%** parallax-satisfied on covered cells, **28.3 s** end-to-end. Insight/plot now use **3D convergence** (replaced azimuth `view_diversity`). 56/56 tests pass.
- [ ] Phase 5b — second flight validation + ODM orthophoto check

---

## Open decisions log

| # | Decision | Status | Choice / leading candidate |
|---|----------|--------|----------------------------|
| 1 | Overlap metric | locked | footprint-cell Jaccard |
| 2 | Parallax metric | locked | 3D convergence angle (max pairwise) |
| 3 | Parallax target (k, θ) | locked | ≥3 views, ≥10° max-pairwise convergence (Phase-0 calibrated; ~69% baseline) |
| 4 | Sharpness gate | locked | reuse existing quality_score/sharpness |
| 5 | Grounded-frame gate | locked | relative-AGL trim, auto-detected takeoff/landing knee (validated: trims 8 seeds, keeps low passes) |
| 6 | Connectivity Jaccard threshold | locked | τ = 0.05 (`overlap_jaccard_threshold`; Phase-2 validated on 0088) |
| 7 | Parallax health floor | locked | ≥15% of covered cells parallax-satisfied (Phase-5 calibrated on 0088 @ 500 kf → 21.7%) |
