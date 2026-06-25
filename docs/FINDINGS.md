# FINDINGS — Orthophoto quality investigation (flight `0088_20260122_eitan_1`)

> Living log of what we explored, what worked, what failed, and *why*. The goal of
> recording this verbosely is so we never re-run a dead end by accident. Newest
> investigation block at the top; keep older blocks intact.

Data root: `/home/bfialkoff/s3/odm-results/0088_20260122_eitan_1/runs/<TS>`
Selection used throughout: **475 keyframes**, `--terrain-gpkg /geo/DSM/israelDTM.gpkg --ray-grid 48 27`
(deterministic — re-selecting produces the identical 474-line `geo.txt`, byte-for-byte).

---

## 2026-06-25 — SMOKING GUN: catastrophic CE90 = global rotation flip (rotation is unconstrained in BA)

Compared solved shot rotations to our validated `R_cam_to_enu` prior (rotation residual
`angle(R_solved @ R_cam_to_enu)`, ~0° if the solve agrees with our orientation):

| run | CE90 | rot-residual median | p90 | max | shots >15° |
|---|---|---|---|---|---|
| `20260624061747` GOOD | 5.06 m | **3.3°** | 12.9° | 28° | 31/469 |
| `20260625103311` BAD  | 43.8 m | **137.8°** | 172.9° | 180° | 470/470 |

**The catastrophic runs are a GLOBAL ROTATION FLIP** — every shot rotated ~138° from truth,
not position jitter. Mechanism: OpenSfM BA constrains only GPS *position*
(`AddRigInstancePositionPrior`); **rotation is entirely free** (no rotation prior wired in the
default pipeline, confirmed). With weak cross-track overlap, the position-only alignment has a
discrete rotation/mirror ambiguity; the solver sometimes lands in a flipped basin that still
fits features + GPS positions. Because the camera is oblique (ground ~600 m ahead of the
drone), even small rotation error is amplified into large ground/CE90 error — and a 138° flip
is total. **This is THE disease behind the run-to-run variance.**

Good news: `R_cam_to_enu` (from `geometry/extrinsics.py`, validated to ~3° on good solves) is
both a perfect *oracle* to detect flipped runs and the exact *prior* to prevent them.

### Options to fix (rotation constraint)
1. **Best-of-N + rotation oracle (do-now, low risk):** run the solve N times, reject any whose
   median rotation residual vs `R_cam_to_enu` > ~15° (the flips), keep the first good one.
   Defeats the exact failure mode; zero distortion risk; uses our prior only as a validator.
2. **Per-image rotation prior in BA (root fix):** OpenSfM's BA *does* expose per-shot
   `add_absolute_up_vector` / `add_absolute_pan` / `add_absolute_tilt` / `add_absolute_roll`
   (pybind), but the stock C++ `ba_helpers.cc` only adds a single GLOBAL up-vector and only when
   `align_method=orientation_prior` (std hardcoded 1e-3, assumes uniform attitude — useless for
   our ±35° roll). Real fix = patch `ba_helpers.cc` to add per-shot up-vector + pan from our
   `R_cam_to_enu` (build a derived ODM image). Prevents the flip by construction.
3. **Synthetic terrain GCPs (medium, riskier):** ray-march image-center to terrain → soft GCPs;
   uses stock GCP machinery but is circular-ish and the 600 m lever amplifies orientation error.

---

## 2026-06-25 — Orientation priors in `geo.txt` (AVENUE CLOSED — reverted)

> TL;DR: orientation priors cannot fix the CE90 variance. The geo.txt YPR channel physically
> cannot represent our strapdown camera (ODM `compute_opk` is lossy, ~40° floor), and opk is
> never an OpenSfM bundle-adjustment rotation prior anyway (only matching + non-default
> triangulation init). Reverted to position-only geo.txt. Details below.

**Hypothesis.** CE90 nondeterminism is rotational drift: we feed ODM only position
(`geo.txt` = `name x y z`) and leave camera attitude free, so the GPS-alignment / BA
solve occasionally drifts into a ~100 m basin. Feeding the 6-DOF attitude we already have
(`yaw_rad/pitch_rad/roll_rad`, strapdown camera, ±35° roll/pitch) as orientation priors
should pin rotation and kill the catastrophic tail.

**Plumbing (correct, works).** `write_geo_txt` (`src/scene_recon/export.py`) extended to
emit 7 fields/line: `name easting northing alt yaw_deg pitch_deg roll_deg` (degrees, order
`yaw pitch roll` per ODM `opendm/geo.py`). ODM ingests cleanly — "Updated 475 image
positions", no parse error; opk derived via `photo.compute_opk()`. Test added
(`tests/test_export.py`), full suite 65 passed.

**Result — orientation made it WORSE, not better** (cold-locked k=64, run `20260625115909`):

| Convention | CE90 samples (m) | blow-ups |
|---|---|---|
| Baseline (position-only) | 8.76, 8.85, 28, 44, 110, 112 | 2/6 (~33%) |
| Orientation, `yaw pitch roll` as-is | 155.5, 108.1 | 2/2 (100%) |
| Orientation, negated yaw (`-yaw`) | 109.6, 119.1 | 2/2 (100%) |

Every orientation run is a uniform ~100–156 m blow-up — strictly worse than baseline
(which at least lands ~1/3 in the good ~9 m regime). The prior means are *coherent* offsets
(as-is x≈−18 m; neg-yaw y≈+12–16 m), not noise → the rotation prior is systematically wrong.

**Why it regressed (root cause, fully traced):** wrong opk corrupts **matching pair
selection**, not bundle adjustment. OpenSfM's `pairs_selection.py` uses per-image opk
(viewing direction) to pick matching candidates; a wrong-but-confident orientation makes it
match the wrong images → bad reconstruction every run.

### Deep dive — the geo.txt orientation channel is a DEAD END for our camera

Investigated the full chain end-to-end against the in-container ODM/OpenSfM source and a
known-good reconstruction (`20260624061747`, CE90 5.06 m, which used **position-only**
geo.txt):

1. **Our extrinsics already match OpenSfM's frame exactly.** `R_solved @ R_cam_to_enu ≈ I`
   (residual 1–6°, just SfM refinement) for the good run. So `R_cam_to_enu` (from
   `geometry/extrinsics.py`, validated by the footprint ray-marcher) is the *correct* prior;
   the desired world→cam prior is simply `R_des = R_cam_to_enu.T`. Camera frame is
   X-right / Y-down / Z-forward = OpenSfM/OpenCV convention. No frame guessing needed.

2. **ODM's `compute_opk` (geo YPR → opk) CANNOT represent our orientations.** The chain is
   `geo YPR → ODM compute_opk (Bäumker cnb · cbb · cen) → omega/phi/kappa → OpenSfM
   rotation_from_opk → R_prior`. Replicated it exactly and **brute-force searched** the full
   YPR cube for the YPR that best reproduces real solved rotations: the global-min residual
   **floors at ~40°** for many shots (and 0° for others). Root cause: `compute_opk`'s `cen`
   is **non-orthonormal** (mixes ECEF north with a constant `[0,0,-1]`) and the code comments
   explicitly assume a **gimballed near-nadir** camera (roll≈0). Our strapdown camera (roll
   ±35°, oblique pitch, mean −16°) lives outside the representable set. So *no* yaw/pitch/roll
   we can write to geo.txt yields the correct prior — the subagent's convention search was
   doomed regardless of sign/frame.

3. **A direct opk injection WOULD be exact** — `opk_from_rotation(R_cam_to_enu.T)` round-trips
   through `rotation_from_opk` to **0.0°** (both orthonormal, true inverse). Injecting opk
   straight into OpenSfM's exif (bypassing `compute_opk`) is the only way to encode our
   orientation faithfully.

4. **But opk is NOT a bundle-adjustment rotation prior in ODM's default pipeline.** Grep of
   the entire OpenSfM tree: no `add_rotation_prior` / rotation-prior residual uses
   `opk_angles`. opk is consumed only by (a) `pairs_selection.py` (matching candidate
   selection) and (b) `reconstruction_from_metadata` for pose *init* — but that's used by
   `triangulation_reconstruction`, **not** the default *incremental* reconstruction ODM runs.
   Also, because we provide altitude, ODM sets `align_method: auto` (GPS), not
   `orientation_prior`. So even a perfect opk prior would **not** add a rotation constraint to
   BA — it cannot pin the attitude drift that causes the CE90 ~100 m tail.

**VERDICT — orientation priors are not the lever for CE90 variance. Avenue closed.**
- geo.txt YPR: dead end (can't represent our strapdown camera; ~40° floor).
- direct opk injection: exact, but only affects *matching* + non-default triangulation init,
  never the incremental-BA rotation. High effort (needs a shim), low expected payoff for the
  variance problem.
- **Reverted** `write_geo_txt` to position-only (with an explanatory comment); kept a
  position-only regression test. Clean baseline restored. The real disease remains
  **under-constrained global georeferencing** (weak cross-track overlap in repetitive
  terrain) — pursue that (e.g. GCPs, stronger cross-track matching, best-of-N), not
  orientation priors.

> ⚠️ **2026-06-25 late update — the "stable baseline" is NOT reliably stable.** A fresh
> cold-seed + LOCKED + k=64 e2e run (`20260625103311`) produced **CE90 = 110 m**, not the
> ~8.85 m we'd seen from the identical recipe (`20260624193806`). Same single healthy
> reconstruction (1 component, 470 shots, ~66k pts, reproj 1.40 px) — the difference is
> purely the **GPS georeferencing alignment**: the good run's gps-error mean is (0,0,0) and
> CE90 8.85 m; the bad run's mean is (1.4, **4.2**, -1.2) with a CE90 110 m long tail (a
> subset of poses drifted >100 m). So even cold-locked SfM is **nondeterministic with a
> catastrophic-failure mode** on this flight (weak global/cross-track constraints ⇒
> under-determined georef similarity that occasionally drifts). CONSEQUENCE: single-run
> CE90 numbers in the matrix below are *samples from a high-variance distribution*, not
> fixed truth. We must characterize the distribution (repeat N×) before "locking in" any
> config, and re-evaluate #3 on its effect on **variance / blow-up rate**, which we never
> measured (we only ever compared single CE90 samples).

### Variance batch — cold-locked k=64, 6 identical-recipe samples (`20260625103311`)

| sample | CE90 (m) | gps-error mean | regime |
|---|---|---|---|
| iter 1 | 8.76 | (0, 0, 0) | ✅ good |
| `193806` | 8.85 | (0, 0, 0) | ✅ good |
| iter 3 | 28.2 | ≈0 (centered, wide) | ⚠️ mediocre |
| iter 4 | 43.8 | (−0.2, −3.3, −0.2) | ❌ bad |
| `103311` | 110 | (1.4, 4.2, −1.2) | ❌ catastrophic |
| iter 2 | 112 | (1.6, 9.1, −2.1) | ❌ catastrophic |

**min 8.76 / median ~36 / max 112. Only 2/6 (33 %) in the good ~9 m regime; half ≥ 36 m;
a third ≥ 110 m.** Cameras LOCKED, no self-cal — so this variance is pure SfM /
GPS-alignment nondeterminism, the dominant effect for this flight.

### Reinterpretation of the whole session (important)

Because CE90 here is a high-variance random variable (8–112 m), **most single-sample
verdicts earlier in this doc are unreliable draws**, not stable facts:
- "explicit-locked 8.88 vs k64-locked 8.85" — both happened to be good rolls; the gap is noise.
- **"warm-LOCKED broke at 116–123 m"** — almost certainly the *same catastrophic-failure
  mode (bad rolls)*, NOT evidence the warm seed is bad. That conclusion is retracted pending
  a repeated-sample test.
- self-cal divergence (focal runaway) is a *separate, real* failure (it changes the focal),
  but its CE90 magnitude is also inflated by this same georef variance.

### The real disease

Global georeferencing is **severely under-constrained** for this flight (sparse cross-track
overlap in repetitive terrain ⇒ the GPS-alignment similarity transform is weakly determined
and the incremental SfM trajectory occasionally drifts). **Variance, not bias, is the enemy.**
CE90 point-comparisons are meaningless until the variance is down. This also finally gives
#3 (explicit pairs) a real, measurable job: **does adding cross-track constraints lower the
blow-up rate / tighten the CE90 distribution?** (Untested.)

### Recommended directions (drawing board)

1. **Measure #3 on variance**: run explicit-pairs cold-locked N× and compare the CE90
   distribution to the table above (does the blow-up rate drop?). This is #3's true test.
2. **Strengthen / stabilize georef**: investigate the GPS alignment (`align_method`,
   `--gps-accuracy`, bundle GPS weight) and whether more cross-track overlap in *selection*
   (not just matching) reduces drift.
3. **Pragmatic guardrail** (stopgap): best-of-N — run K solves, keep the lowest-CE90 one
   (CE90 is measurable post-hoc from `opensfm/stats/stats.json`). Cheap insurance while the
   root cause is worked.

---

## 2026-06-25 — Matching profile (#3 explicit pairs), camera self-cal, and reproducibility

### TL;DR / bottom line

1. **The only reproducible "good" config is: cold lab seed (focal 1.1845) + LOCKED cameras
   + k=64 GPS-kNN → CE90 ≈ 8.85 m.** This is our stable baseline. Lock it in.
2. **Explicit co-visibility pairs (#3) are correctly implemented and matching-clean, but
   CE90-neutral for this flight.** They add +26 % cross-track solid pairs at ~0.997
   precision, yet move neither CE90 nor global connectivity. Matching is **not** the lever.
3. **Camera self-calibration is unreliable here and usually diverges** (focal runs away
   from ~1.13 to 1.6–2.6; CE90 44–159 m). The historic 5.06 m run was a *lucky convergence
   basin*, not a reproducible recipe.
4. **Root cause of the self-cal divergence**: our frames are **EXIF-less**, so OpenSfM's
   focal prior falls back to `default_focal_prior = 0.85` (NOT our seed). With self-cal on,
   the focal is effectively unanchored, and in this low-parallax / repetitive-terrain
   aerial set the bundle exploits the **focal↔depth ambiguity** to lower reprojection error
   by inflating the focal — destroying georeferencing.
5. Exonerated (NOT the cause of anything): the covis-shim `.pth` mount, the ODM image
   (local `opendronemap/odm:gpu` = `365bf752b230`, 2 months old, unchanged across the
   experiments), and the matching profile.

### The experiment matrix (all 475-frame, `--gps-accuracy 3`)

| run TS | worktree | matching | camera mode | seed focal | converged focal | CE90 (m) | components | x-track solid | inliers/pair | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260624061747` | (historic) | k=64 | **self-cal** | 1.1298 | **1.1298** | **5.06** | — | — | — | ✅ good but NOT reproducible |
| `20260624193806` | main | k=64 | **locked** | 1.1845 | 1.1845 | **8.85** | 2 | 2101 | 306 | ✅ **reproducible baseline** |
| `20260624182016` | betzalel | **explicit** | locked | 1.1845 | 1.1845 | 8.88 | 3 | 2648 | 294 | ✅ matching-clean, CE90-neutral |
| `20260625073013`·a | betzalel | explicit | self-cal | 1.1845 (cold) | 2.04 | 159 | 3 | — | — | ❌ diverged |
| `20260625073013`·b | betzalel | explicit | locked | 1.1298 (warm) | 1.1298 | 123 | 3 | 2642 | 294 | ❌ warm-LOCKED broke |
| `20260625080329`·a | betzalel | k=64 | locked | 1.1298 (warm) | 1.1298 | 116 | 2 | 2091 | 306 | ❌ warm-LOCKED broke |
| `20260625073013`·c | betzalel | explicit | self-cal | 1.1298 (warm) | 2.61 | 44.2 | 3 | 2640 | 294 | ❌ diverged |
| `20260625080329`·b | betzalel | k=64 | self-cal | 1.1298 (warm) | 2.18 | 45.1 | 2 | 2086 | 307 | ❌ diverged |
| `20260625080329`·c | **main (no shim)** | k=64 | self-cal | 1.1298 (warm) | 1.61 | 44.6 | 2 | 2075 | 307 | ❌ diverged (isolation test) |

(`·a/·b/·c` = successive `--rerun-all` re-runs over the same run dir; each self-cal run
mutates `odm_input/cameras.json` by writing the converged intrinsics back, so the seed must
be restored from a known file before re-seeding — see "Camera seed mechanics".)

### What each result proves

- **Matching is CE90-neutral.** Explicit vs k=64 at every camera operating point:
  locked 8.85 ↔ 8.88; warm-self-cal 44.2 ↔ 45.1; warm-locked 116 ↔ 123. The +26 %
  cross-track pairs and 0.997 precision are real (the shim works perfectly) but do not
  change the georeferencing or merge the 2↔3 component split. The serendipitous bridge
  that GPS-kNN's ~3.5 % non-covis pairs sometimes catches is *lost* by an exact covis∪seq
  match — explaining why explicit can even show 1 *more* component. Net: #3 is good, safe
  infrastructure, not a quality lever for this flight.

- **Self-cal divergence is matching-independent and seed-independent.** Cold-seed self-cal
  → 2.04; warm-seed self-cal → 2.18 (k=64) / 2.61 (explicit) / 1.61 (main). All diverge,
  all land ~44–159 m. Different converged focals each run ⇒ nondeterministic (GPU SIFT +
  RANSAC, amplified by free-focal BA).

- **`061747`'s 5.06 m is not reproducible.** Re-running its *exact* recipe (warm seed,
  k=64, self-cal) on byte-identical inputs — same 474 frames, same `geo.txt` UTM coords,
  same options, same 2-month-old image — diverges to 44.6 m. It was a lucky basin.

- **Warm-LOCKED is reproducibly broken (116–123 m).** Locking at `061747`'s exact converged
  intrinsics (1.1298 + its distortion/principal point) gives 116–123 m, while locking the
  *cold* 1.1845 gives 8.85 m. So the fragility is in the full BA / GPS-alignment landscape,
  not just the focal value: good intrinsics alone don't guarantee a good solve, and
  locking the "better" intrinsics from scratch lands incremental SfM in a bad alignment.

### Root cause (self-cal focal runaway)

OpenSfM `config.py` (in the image):
- `default_focal_prior: 0.85`
- `exif_focal_sd: 0.01` (log-scale std of the focal prior; small = tight)
- `principal_point_sd: 0.01`
- `optimize_camera_parameters: True` (all-or-nothing; no "distortion-only" flag)

Because the frames carry no EXIF focal, the prior anchor is **0.85**, not our 1.1298 seed
(the seed only sets the *initial* `focal`, via `cameras.json` → `camera_models_overrides`,
not the *prior*). In near-nadir, low-parallax, repetitive terrain the reprojection cost has
a near-degenerate focal/depth direction; with the focal unanchored the optimizer slides
along it (focal ↑, depths/distortion compensate), lowering reprojection error while
wrecking the global scale/orientation → CE90 blows up.

### Proposed fix (NOT yet implemented) — "focal-pin"

Refine **distortion** (the thing that originally fixed lens warp) while **pinning the focal**
to the seed: a tiny `opensfm.config` monkeypatch (same opt-in `.pth` mechanism as the
covis-shim) that (a) sets `exif_focal_sd` very small and (b) sets the camera's `focal_prior`
to the seed value. Expected to make the good solve **deterministic** instead of lucky.
Alternative/again-to-test: re-seed prior from the cold lab focal and lock only distortion.

### Explicit-pairs (#3) implementation — keep, it's good infra

Lives on worktree `betzalel-improve-keyframe-selection` (uncommitted at time of writing):
- `src/scene_recon/matching/profile.py`: `explicit_pairs(graph, seq_window=2)` =
  co-visibility edges ∪ a ±2 sequential window; `ExplicitPairListBackend` writes
  `covis_pairs.json` + `matching_profile.json`.
- `docker/covis_shim/covis_pairs_shim.{py,pth}`: in-container `.pth` that, only when
  `COVIS_PAIRS_FILE` is set, monkeypatches `opensfm.pairs_selection.match_candidates_from_metadata`
  to return exactly our pairs (hard-fails on 0 usable pairs; never silently falls back).
- `scripts/run_odm.sh`: `EXPLICIT_PAIRS=1` wires `COVIS_PAIRS_FILE` and skips `--matcher-neighbors`.
- `docker-compose.yml`: mounts the shim into `/code/venv/lib/python3.12/site-packages/`.
- Verified in-container: OpenSfM signature matches the shim, shim fires (`13798 matched,
  0 dropped`), pairs are consumed. `src/scene_recon/matching/scoreboard.py` gained
  `pair_coverage`/`CoverageReport` to quantify predicted-vs-solid coverage.
- **Status**: works end-to-end, CE90-neutral on this flight. Safe to keep as opt-in; do not
  make it the default expecting a CE90 win.

### Camera seed mechanics (gotchas)

- `cameras.json` in a run's `odm_input/` is the **initial intrinsics** ODM/OpenSfM seeds
  from. With `OPTIMIZE_CAMERAS=1` (self-cal), the converged intrinsics are written **back**
  into `cameras.json`, so a second `--rerun-all` no longer starts from the original seed.
  Always restore the seed from an immutable source (e.g. `061747/odm_input/cameras.json` for
  the warm 1.1298, or rebuild for the cold 1.1845) before re-seeding.
- ODM **re-keys** an override `cameras.json` to its detected camera id. A no-`v2` input key
  (`'  1920 1080 brown 0.85'`) binds fine to the detected `'v2   1920 1080 brown 0.85'`;
  confirmed via `opensfm/camera_models_overrides.json` carrying the right `focal_x`.

### Run-it-back commands (for posterity)

```bash
# Reproducible baseline — cold seed + LOCKED + k=64 (full e2e ortho) from main:
cd <main-worktree>
TS=$(./scripts/build.sh 0088_20260122_eitan_1 --select-only \
       --terrain-gpkg /geo/DSM/israelDTM.gpkg --ray-grid 48 27 | tail -1)
TS=$(basename "$TS")
./scripts/run_odm.sh 0088_20260122_eitan_1 --run "$TS" -- \
       --auto-boundary --gps-accuracy 3 --fast-orthophoto --rerun-all   # LOCKED (no OPTIMIZE_CAMERAS)

# Score any run:
PYTHONPATH=src .venv/bin/python scripts/analyze_matching.py <RUN_DIR>     # scoreboard + CE90 + match-graph plots
```

### Open questions / drawing-board for "improve more"

- **Focal-pin** (above) — most principled path to a reliable, deterministic sub-9 m solve.
- Why does **warm-LOCKED** reconstruct *worse* than cold-LOCKED despite better intrinsics?
  Suspect BA/GPS-alignment basin sensitivity from-scratch; worth a controlled look.
- The 2-component split: is there a genuine coverage gap between the halves (selection /
  flight-path), or only a weak-link in matching? Explicit pairs did **not** close it.
- Original complaint was **orthophoto warp/tiling**, of which CE90 is only a proxy — judge
  candidate configs on the actual ortho (warp/seams), not CE90 alone.
