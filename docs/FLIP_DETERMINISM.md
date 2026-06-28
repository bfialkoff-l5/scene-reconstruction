# Georeferencing flip / CE90 non-determinism — investigation & next steps

Status: **paused / lock-in.** Root cause identified and proven. Next phase = config-level
determinism experiments (no source edits yet). This doc is the resume context.

## TL;DR

The pipeline at `main@02b95e8` is **non-deterministic**: byte-identical input produces CE90
anywhere from ~9 m (good) to ~200 m (catastrophic "global rotation flip"). The "stable
single-digit baseline" was real but was a *lucky draw* of a ~50/50 process, not a stable
state. Reverting `main` did **not** buy stability (it can't — `02b95e8` flips too). The flip
is the same problem the (now parked) rotation-prior experiment was attacking.

The flip fingerprint: **low reprojection error + high GPS CE90** = the reconstruction is
internally consistent but globally rotated relative to the GPS/world frame.

## Proof (byte-identical input, stock image, same machine)

| run dir | CE90 | le90 | reproj_norm | outcome |
|---|---|---|---|---|
| `20260628040841` (demo, full pipeline) | 195.92 | 15.69 | 0.240 | full flip |
| `20260628043735` (probe 1) | 35.64 | 22.53 | 0.280 | partial flip |
| `20260628050129` (probe 2) | 8.99 | 6.06 | 0.211 | good |
| `20260628051203` (probe 3) | 8.90 | 5.94 | 0.211 | good |

Last week's "good" runs (same recipe, same machine): `20260624182016`=8.88, `20260624193806`=8.85,
`20260625103311`=8.74. The 8.90/8.99 probes reproduce them exactly → the good basin is real.

### Ruled out (these are byte-identical between an 8.85 m run and the 196 m run)
- ODM image: `opendronemap/odm@sha256:365bf752…`, built **2026-04-20** (rolling tag did NOT drift)
- ODM options.json (except trivial `end_with` / `rerun_all` / `project_path`)
- `cameras.json` focal/distortion (focal_x = 1.184540)
- `geo.txt` (md5 `287a48a5…`, position-only)
- selected frames (md5 `8bb5eea8…`, 475 frames)
- same host (laptop, ThinkPad-P16s)

So it is NOT args/image/calibration/geo/selection. It is solver non-determinism.

## Root cause (from the OpenSfM config the runs printed)

- `matcher_type: FLANN` (+ `flann_algorithm: KDTREE`) — approximate NN, randomized
- `processes: 22` — parallel feature extraction/matching → ordering varies run-to-run
- GPU/CUDA feature extraction (DSPSIFT) — another non-determinism source
- `reconstruction_algorithm: incremental` — bootstrap pair chosen from (jittery) match counts
- `align_method: auto`, `align_orientation_prior: vertical` — final GPS alignment; **the flip
  happens here**: for our oblique, near-planar acquisition the recon→world orientation has a
  ~180° ambiguity that `auto` sometimes resolves the wrong way.

**Key principle: determinism ≠ correctness.** Locking the run (e.g. single-thread) makes it
*repeatable* but could lock onto the *flipped* basin. We need determinism AND the good basin.

## Next steps — config-level experiments (no source edits)

All of these are clean ODM CLI flags forwarded through `scripts/run_odm.sh ... -- <flags>`.
Harness: `scripts/experiments/run_variance.sh <label> <N> -- <flags>` (copies a fixed
md5-identical input, runs OpenSfM N×, reports the CE90 distribution + GOOD/FLIP).

Relevant ODM flags (verified against `opendronemap/odm:gpu --help`):
- `--matcher-type {flann,bruteforce,bow}` — default flann. **bruteforce = exhaustive/deterministic** ("very slow but robust").
- `--max-concurrency N` — default 22. `1` ⇒ removes thread non-determinism.
- `--no-gpu` — CPU feature extraction (removes CUDA non-determinism).
- `--sfm-algorithm {incremental,triangulation,planar}` — default incremental. `triangulation`
  anchors to GPS positions+angles (we have positions only → benefit uncertain). `planar` is
  nadir/flat only (poor fit for our oblique camera).

### Experiment plan (ranked; run when resuming)

1. **Exp A — brute-force matcher.** `run_variance.sh bruteforce 3 -- --matcher-type bruteforce`
   Hypothesis: removes the FLANN match-graph lottery → tighter variance, maybe no flips.
   Likely to help; uncertain it fully eliminates flips (BA/thread residual remains).

2. **Exp B — full determinism.** `run_variance.sh single 2 -- --matcher-type bruteforce --max-concurrency 1 --no-gpu`
   Hypothesis: bit-reproducible runs (high confidence). Then check WHICH basin it locks onto.
   Slow (single-thread). Determinism proof, not a correctness guarantee.

3. **Exp C — GPS-anchored SfM (best shot at fixing the flip).**
   `run_variance.sh triangulate 3 -- --sfm-algorithm triangulation`
   Hypothesis: anchoring structure to the GPS track + vertical prior pins global orientation,
   killing the flip. Biggest upside for *correctness*. Risk: triangulation wants camera angles
   and we ship position-only `geo.txt`; if it underperforms, that's evidence the orientation
   prior is genuinely required.

4. **Exp D — combine** the winning determinism lever with the winning correctness lever.

Decision logic:
- If a config makes runs deterministic AND lands in the good basin (≥5/5 single-digit CE90) → ship it on `main` (pure flag change in `run_odm.sh` defaults).
- If deterministic-but-flipped → the orientation is the problem; revisit the parked
  rotation-prior experiment (done correctly) or an `align_method`/orientation_prior config
  override (the source-ish path we deferred).

## Repo / worktree state (as of lock-in)

| worktree | branch | commit | purpose |
|---|---|---|---|
| `main` | `main` | `02b95e8` | clean stable baseline (stock image, position-only geo) |
| `flip-determinism` | `investigate/flip-determinism` | off `02b95e8` | **this investigation** (config experiments) |
| `rotation-prior-experiment` | `experiment/rotation-prior` | `6e52ff5` | parked C++ BA rotation prior (submodule `rotation-prior-patch` @ `3b44c559`; transpose bug fixed + C++-unit-tested, but never recovered single-digit CE90) |
| `betzalel-improve-keyframe-selection` | `betzalel/improve_keyframe_selection` | `02b95e8` | untouched |
| `betzalel-speedups` | `betzalel/speedups` | `528a7c9` | untouched |

Nothing pushed; all local and reversible. The rotation-prior root-cause writeup lives at
`docs/FINDINGS.md` on the `experiment/rotation-prior` branch (dropped from `main` intentionally).

## How to run

```bash
cd /home/bfialkoff/projects/scene_reconstruction/flip-determinism
# IMPORTANT: a previous shell exported ODM_IMAGE=odm-osfm:rotprior (the PATCHED image).
# The harness forces stock, but for ad-hoc run_odm.sh calls: unset ODM_IMAGE first.
unset ODM_IMAGE
./scripts/experiments/run_variance.sh bruteforce 3 -- --matcher-type bruteforce
```

Good vs flip is judged by CE90 (<15 m = good) with low reprojection confirming the flip
signature. Each OpenSfM-only run is ~9 min on FLANN; brute-force / single-concurrency are
substantially slower — budget accordingly.

## Gotchas
- `ODM_IMAGE=odm-osfm:rotprior` may still be exported in old shells → would silently use the
  patched OpenSfM. Always `unset ODM_IMAGE` (or use the harness, which forces stock).
- `opendronemap/odm:gpu` is a rolling tag; pin/record the digest (`sha256:365bf752…`) if you
  need exact reproducibility later.
- The `SRC` input dir (`runs/20260628040841/odm_input`) is the md5-identical reference; if it
  gets cleaned, rebuild from main: `./scripts/build.sh 0088_20260122_eitan_1 --terrain-gpkg /geo/DSM/israelDTM.gpkg`.
