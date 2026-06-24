# AGENTS.md

Orientation for an agent waking up in this repo — especially on a fresh EC2 box. Read
this first, then `README.md` and `docs/PLAN.md` for depth.

## What this is

A manual two-step aerial photogrammetry pipeline that turns one drone video + its poses
into an **orthophoto** via stock OpenDroneMap (ODM):

1. **build** — score every video frame, select ~hundreds of keyframes, extract them, and
   write an ODM input layout (`images/` + `geo.txt`). Pure CPU, runs in the `builder`
   container.
2. **odm** — stock `opendronemap/odm` on GPU, driven by `scripts/run_odm.sh`.

Both steps run in Docker via `docker compose`; the host only needs Docker + the NVIDIA
Container Toolkit. Local laptop and EC2 use the **identical** commands — only `DATA_ROOT`
differs.

The working dataset is the slug **`0088_20260122_eitan_1`** (an ~80 m AGL oblique flight
over Eitan, Israel).

## Where data lives

On disk everything hangs off `DATA_ROOT` (`/data` on EC2, `/home/bfialkoff/s3` on the
laptop). The builder auto-discovers files inside a slug folder; you only ever pass the slug.

| What | On-disk path | Source on S3 |
|------|--------------|--------------|
| Raw flight (video, intrinsics, poses) | `$DATA_ROOT/raw/0088_20260122_eitan_1/` | `s3://line5-localization-evaluation-data-939103584914-eu-north-1-an/Flight logs/Flight 000/0088_20260122_eitan_1/` |
| DTM (terrain, required by build) | `$DATA_ROOT/geo-resources/DSM/israelDTM.gpkg` | `s3://line5-localization-evaluation-data-939103584914-eu-north-1-an/israelDTM.gpkg` |
| Reference orthophoto (overlay/QA) | `$DATA_ROOT/geo-resources/orthophoto/Eitan.gpkg` | `s3://line5-localization-evaluation-data-939103584914-eu-north-1-an/Maps/Eitan.gpkg` |

The builder needs only three files from the raw folder: `AvatarS0093.mp4`,
`intrinsicK.csv`, and `_derived/gt_AvatarS0093.csv` (the `G0000` stream is ignored — it has
no matching `gt_` poses). `geo-resources` is mounted read-only at `/geo` in the builder, so
the DTM is referenced as `/geo/DSM/israelDTM.gpkg`.

Outputs land under `$DATA_ROOT/odm-results/{slug}/runs/{timestamp}/`. The orthophoto is
`.../odm_input/odm_orthophoto/odm_orthophoto.tif`.

## Fetch the data (fresh box)

`scripts/launch-ec2.sh` attaches the per-bucket S3 instance role
`l5-localization-evaluation-data-rw` (scoped read-write to the eval-data bucket only;
`PassRole` is granted to power users, so no admin step). On such a box `aws s3 ...` just
works via IMDS — **no `aws sso login` needed** to fetch data. (The `ec2-vcpu-limiter`
Lambda only enforces the per-owner vCPU/GPU budget via the `owner` tag; it does *not* block
IAM roles.)

Fallback if you launched with `IAM_PROFILE=""` (no role): authenticate with SSO instead.
`scripts/bootstrap-ec2.sh` already wrote `~/.aws/config`, so the only human step is the
browser code:

```bash
aws sso login --profile bfialkoff --no-browser   # open the printed URL, approve, done
```

Note the two-region quirk baked into that config: `sso_region = us-east-1` (where IAM
Identity Center lives) but the profile `region = eu-north-1` (where the S3 data lives).
Setting `sso_region` to eu-north-1 makes login fail at `RegisterClient`.

Then pull everything (raw flight without the unused 1.1 GB G-stream, DTM, reference ortho)
into the layout the builder expects. `DATA_ROOT` and `AWS_PROFILE` come from `.env`:

```bash
./scripts/fetch-data.sh 0088_20260122_eitan_1 Eitan.gpkg
```

All inputs live in the one bucket
`line5-localization-evaluation-data-939103584914-eu-north-1-an` (raw under `Flight logs/
Flight 000/<slug>/`, orthos under `Maps/`, DTM at the root).

## Run it

```bash
cp .env.example .env
# edit .env: DATA_ROOT=/data, GEO_RESOURCES=/data/geo-resources, AWS_PROFILE=bfialkoff
docker compose build

# Step 1 — build (score + select). --terrain-gpkg and --ray-grid are required.
./scripts/build.sh 0088_20260122_eitan_1 --terrain-gpkg /geo/DSM/israelDTM.gpkg --ray-grid 48 27

# Re-select without re-scoring (scoring is cached per slug):
./scripts/build.sh 0088_20260122_eitan_1 --select-only --terrain-gpkg /geo/DSM/israelDTM.gpkg --ray-grid 48 27

# Step 2 — ODM. Best-known recipe: co-visibility matcher profile (auto, from build) +
# camera self-calibration seeded from cameras.json (OPTIMIZE_CAMERAS=1). See "Current
# best run" below for why each knob is set this way.
OPTIMIZE_CAMERAS=1 ./scripts/run_odm.sh 0088_20260122_eitan_1 -- --auto-boundary --gps-accuracy 3 --fast-orthophoto
```

Long runs: start them inside `tmux` so they survive SSH disconnects. The GPU box bills
continuously — terminate it when done (`scripts/terminate-ec2.sh`).

## Current best run

`odm-results/0088_20260122_eitan_1/runs/20260624061747` — best reconstruction + orthophoto
to date. This is the baseline every future change must beat.

Recipe (the single source of truth for "good"):
- **Selection**: stock 4 m spacing, ~475 keyframes (`build.sh --select-only`).
- **Matching**: co-visibility-driven matcher profile → `matcher_neighbors = 64` (was 16),
  written to `odm_options.json` at build time by `src/scene_recon/matching/` and read by
  `run_odm.sh`. This is what straightened the rows and killed most of the warp.
- **Camera**: self-calibration **on**, seeded from `cameras.json` (`OPTIMIZE_CAMERAS=1`, i.e.
  *no* `--use-fixed-camera-params`). BA refined focal 1.1845 → 1.126 plus the distortion
  terms; this removed the residual edge smear.
- **Features**: `--feature-quality high` (NOT ultra — ultra is worse, see below).
- `--gps-accuracy 3 --fast-orthophoto`.

SfM scoreboard (`scripts/analyze_matching.py`) — the bar to beat: reproj **1.084 px**, mean
track length **7.27**, match-graph **1 component**, cross-track solid ratio **0.243**, GPS
**CE90 ~5 m / LE90 ~2 m**, 469/475 shots reconstructed.

Falsified along the way (don't retry without a new reason):
- **`--feature-quality ultra`**: 3× more features but *shorter* tracks (7.27 → 6.3), graph
  split into 2 components, and with self-cal the focal ran away (1.18 → 1.93) → CE90 105 m.
  Repetitive vineyard texture turns extra features into false short tracks; resolution is
  not the lever. (Even with cameras locked, ultra still gave CE90 16 / LE90 20.)

## Pipeline facts worth knowing (hard-won)

- **Keyframe spacing = 4 m** is the empirical optimum at this AGL: ~475 keyframes, best ODM
  result (33.5 ha, healthy triangulation). Denser packing *degrades* quality unless the
  matcher reach scales with it — see `src/scene_recon/selection/params.py` for the full
  rationale. Don't add a per-cell view-count cull on top; it packs zero-baseline
  near-duplicates and wrecks triangulation.
- **`--matcher-neighbors` is set by the co-visibility predictor** (`src/scene_recon/matching/`),
  not the old 33 m reach heuristic. It builds a footprint-overlap + view-angle graph over the
  selection and picks neighbours so cross-track (loop-closure) pairs are actually matched —
  64 on 0088, up from 16. `run_odm.sh` reads it from `odm_options.json`. The legacy
  `recommend_matcher_neighbors` in `odm.py` remains only as the fallback when footprints are
  missing.
- **Camera intrinsics: seeded self-calibration is now best** — set `OPTIMIZE_CAMERAS=1` so
  `run_odm.sh` passes `--cameras cameras.json` *without* `--use-fixed-camera-params`; BA
  refines the lab intrinsics and removes edge smear (CE90 9 → 5 m on 0088). The old
  "self-cal gives 0-point clouds" failure was *unseeded* self-cal on a weak match graph; once
  matching was fixed (neighbours = 64), seeded self-cal converges (focal 1.18 → 1.13).
  `Camera.camera_id()` must still emit the exact key ODM expects for EXIF-less PNGs so the
  seed binds. Caveat: self-cal is fragile — with `--feature-quality ultra` the focal ran away
  (1.18 → 1.93), so keep it **seeded + high features**.
- **Selection requires the DTM** (`frustum_view_count_dtm` policy): it ray-marches each
  frustum onto the terrain to compute footprints. No `--terrain-gpkg` → build fails.

## Known open issues (not yet solved)

- **Interior coverage holes** (e.g. near `664696,3492563`): *not* genuine data gaps — ~88% of
  the holed cells are coverable by frames the selector dropped on 4 m spacing. This is a
  selection gap, being addressed by a **parallax-gated coverage backfill** (Stage 1c in
  `selection/selector.py`): re-admit dropped frames that uniquely raise an under-covered
  cell's convergence angle, parallax-gated so no zero-baseline near-duplicates get in.
- **`--fast-orthophoto` mesh artifacts**: warped/seam patches come from the 2.5 D fast mesh.
  A full dense reconstruction (drop `--fast-orthophoto`) + `pc-quality high` is a queued
  experiment, to run once matching/selection are dialled in.
- **Global mis-registration vs the reference ortho**: GPS-only georeferencing. Seeded self-cal
  cut the SfM GPS error to CE90 ~5 m / LE90 ~2 m; closing the last metres needs GCPs or
  external 2D co-registration, not a recon-level change.

## EC2 specifics (this account)

- An account-wide `ec2-vcpu-limiter` Lambda **auto-stops any instance without an `owner`
  tag**, or whose owner exceeds their per-owner vCPU/GPU budget (global, all regions).
  Owner `betzalel` = **24 vCPU / 1 GPU** (per the limiter's live `OWNER_LIMITS`) — a
  `g6.4xlarge` (16 vCPU) fits with headroom. The launch script (`scripts/launch-ec2.sh`)
  sets `owner=betzalel` automatically.
- Helper scripts: `scripts/launch-ec2.sh` (boot + write `~/.ssh/config` alias `scene-recon`),
  `scripts/terminate-ec2.sh`, `scripts/allow-my-ip.sh` (add the current network's IP to the
  SG — run once per network: office/home/hotspot).
- Code reaches the box via `git clone` over SSH agent forwarding (`ssh -A` / VS Code
  Remote-SSH). `origin/main` is always the source of truth.
