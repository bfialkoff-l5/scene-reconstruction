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

This account has **no EC2 IAM role** (the `ec2-vcpu-limiter` Lambda blocks role attach), so
authenticate with SSO. `scripts/bootstrap-ec2.sh` already wrote `~/.aws/config` (no
`aws configure sso` needed), so the only human step is approving the browser code:

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

# Step 2 — ODM. This is our best-known fast-ortho invocation.
./scripts/run_odm.sh 0088_20260122_eitan_1 -- --auto-boundary --gps-accuracy 3 --fast-orthophoto
```

Long runs: start them inside `tmux` so they survive SSH disconnects. The GPU box bills
continuously — terminate it when done (`scripts/terminate-ec2.sh`).

## Pipeline facts worth knowing (hard-won)

- **Keyframe spacing = 4 m** is the empirical optimum at this AGL: ~475 keyframes, best ODM
  result (33.5 ha, healthy triangulation). Denser packing *degrades* quality unless the
  matcher reach scales with it — see `src/scene_recon/selection/params.py` for the full
  rationale. Don't add a per-cell view-count cull on top; it packs zero-baseline
  near-duplicates and wrecks triangulation.
- **`--matcher-neighbors` is auto-scaled** per selection to hold a fixed ~33 m matching
  baseline reach (`recommend_matcher_neighbors` in `src/scene_recon/odm.py`); `run_odm.sh`
  reads it from `odm_options.json`. A fixed neighbour count silently collapses the baseline
  as spacing densifies.
- **Camera intrinsics are forced**: `run_odm.sh` always passes `--cameras cameras.json
  --use-fixed-camera-params`, and `Camera.camera_id()` emits the exact key ODM expects for
  EXIF-less PNGs so the override binds instead of ODM self-calibrating (self-calibration
  produced 0-point dense clouds).
- **Selection requires the DTM** (`frustum_view_count_dtm` policy): it ray-marches each
  frustum onto the terrain to compute footprints. No `--terrain-gpkg` → build fails.

## Known open issues (not yet solved)

- **Interior coverage holes** (e.g. near `664696,3492563`): genuine data gaps — no near-field
  views, not a selection bug. Fixing needs either more frames there or accepting it.
- **`--fast-orthophoto` mesh artifacts**: warped/seam patches come from the 2.5 D fast mesh.
  A full dense reconstruction (drop `--fast-orthophoto`) is the next experiment the EC2 box
  exists for.
- **Global mis-registration vs the reference ortho** (~15 m, CE90-scale): expected for
  GPS-only georeferencing. Fixing needs GCPs or external 2D co-registration, not a recon-level
  change.

## EC2 specifics (this account)

- An account-wide `ec2-vcpu-limiter` Lambda **auto-stops any instance without an `owner`
  tag**, or whose owner exceeds their per-owner vCPU/GPU budget (global, all regions).
  Owner `betzalel` = **16 vCPU / 1 GPU** — exactly one `g6.4xlarge`, zero headroom. The
  launch script (`scripts/launch-ec2.sh`) sets `owner=betzalel` automatically.
- Helper scripts: `scripts/launch-ec2.sh` (boot + write `~/.ssh/config` alias `scene-recon`),
  `scripts/terminate-ec2.sh`, `scripts/allow-my-ip.sh` (add the current network's IP to the
  SG — run once per network: office/home/hotspot).
- Code reaches the box via `git clone` over SSH agent forwarding (`ssh -A` / VS Code
  Remote-SSH). `origin/main` is always the source of truth.
