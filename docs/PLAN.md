# Scene Reconstruction — Plan

Manual, two-step photogrammetry pipeline. Docker everywhere. One config, one slug per run — local and EC2 use the same commands and directory layout.

---

## Goals

- **Local first** — iterate on frame selection and ODM flags on your laptop.
- **Manual trigger** — you decide when to run; no job queue yet.
- **Identical local / EC2** — same paths, same scripts; only `data_root` mount differs.
- **ODM is a black box** — stock `opendronemap/odm` for now; custom image only if we outgrow it.
- **Data already synced** — builder auto-discovers files inside a slug folder; no GCSV/BIN sync.
- **Convention over configuration** — slug is the only per-dataset identifier you pass.

---

## Standard layout

Every environment uses the same tree under a single `data_root`:

```text
{data_root}/
├── raw/
│   └── {slug}/                         e.g. 0088_20260122_eitan_1
│       ├── AvatarS0093.mp4             auto-discovered
│       ├── intrinsicK.csv
│       └── _derived/
│           └── gt_AvatarS0093.csv      auto-discovered (matches mp4 stem)
│
└── odm-results/
    └── {slug}_{timestamp}/             auto-stamped, never named by hand
        ├── run.json                    manifest (inputs, config hash, frame count)
        ├── odm_input/
        │   ├── images/
        │   └── geo.txt
        ├── odm_orthophoto/
        ├── odm_dem/
        └── odm_report/
```

**Example (local):** `data_root=/home/bfialkoff/s3`

```text
/home/bfialkoff/s3/raw/0088_20260122_eitan_1/
    AvatarS0093.mp4
    intrinsicK.csv
    _derived/gt_AvatarS0093.csv
```

**Example (S3):** `data_root=s3://my-bucket` → `s3://my-bucket/raw/0088_20260122_eitan_1/`

Same relative paths everywhere. If a slug folder currently sits directly under `s3/` without `raw/`, move or symlink it into `raw/` once — then never think about it again.

---

## Slug and auto-discovery

### Slug

The folder name under `raw/` is the **only ID** you pass:

```bash
./scripts/build.sh 0088_20260122_eitan_1
./scripts/run_odm.sh 0088_20260122_eitan_1
```

### Auto-discovery (no per-dataset config)

Given `{data_root}/raw/{slug}/`, the builder finds:

| File | Rule |
|------|------|
| Video | `*.mp4` where `_derived/gt_<stem>.csv` exists |
| Poses | `_derived/gt_<stem>.csv` matching the chosen mp4 |
| Intrinsics | `intrinsicK.csv` at slug root (required) |

If zero or multiple mp4s satisfy the gt-file rule, fail with a clear error listing candidates. In practice the Avatar **S** stream pairs with `gt_AvatarS0093.csv`; a co-located `AvatarG0000.mp4` is ignored because it has no matching gt file.

### Output naming

You never set the output folder name — only `data_root` (in config). Each build creates:

```text
{data_root}/odm-results/{slug}_{timestamp}/
```

`timestamp` is UTC compact, e.g. `20260616T143022Z`. `run.json` records slug, timestamp, discovered files, frame count, and config used. `run_odm.sh` defaults to the **latest** `{slug}_*` under `odm-results/`; pass `--run 0088_20260122_eitan_1_20260616T143022Z` to pin a specific build.

---

## Architecture

Two containers share `data_root` via bind mount. Host runs Docker only.

```text
┌──────────────────────────────────────────────────────────────┐
│  Host (laptop or EC2)                                        │
│                                                              │
│   {data_root}/raw/ ...          input datums (slug folders)  │
│   {data_root}/odm-results/ ...  stamped run outputs          │
│   ./repo → /app (dev mount)     live code while iterating    │
│                                                              │
│   ┌─────────────────┐         ┌─────────────────┐           │
│   │  builder        │         │  odm            │           │
│   │  Python+ffmpeg  │────────▶│  opendronemap   │           │
│   │  CPU            │         │  GPU            │           │
│   └─────────────────┘         └─────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

**Step 1 — `build`:** discover files in `raw/{slug}/`, select frames, write `odm-results/{slug}_{ts}/odm_input/`.

**Step 2 — `odm`:** `--project-path` = run dir; positional dataset name = `odm_input`. ODM writes outputs under `odm_input/odm_*/`.

---

## Configuration

One shared config for pipeline parameters — **not** per slug.

`configs/builder.yaml`:

```yaml
# Roots — only thing that may differ between laptop and EC2
data_root: /data                    # host path inside container mount

# Frame selection (tune as pipeline results dictate)
min_altitude_m: 20
min_spacing_m: 2.0
min_heading_delta_deg: 5
max_images: 500
jpeg_quality: 90
```

No `poses_file`, `video_file`, or output paths in config. Those come from slug + conventions.

`.env` (non-secret, local convenience):

```bash
DATA_ROOT=/home/bfialkoff/s3
AWS_REGION=eu-north-1
AWS_PROFILE=line5-dev              # local only — omit on EC2 with IAM role
```

`docker-compose.yml` mounts `${DATA_ROOT}` → `/data` so `data_root: /data` works identically everywhere.

CLI flags override `builder.yaml` for one-off experiments (`--min-spacing-m 3`).

---

## Daily workflow

```bash
# Step 1
./scripts/build.sh 0088_20260122_eitan_1

# Step 2 (latest build for this slug)
./scripts/run_odm.sh 0088_20260122_eitan_1 --fast-orthophoto

# Or chain both
./scripts/run_all.sh 0088_20260122_eitan_1 --fast-orthophoto
```

That's it. Slug is the only positional arg. Config path defaults to `configs/builder.yaml`.

---

## Step 1 — Dataset builder

### Responsibility

1. Resolve `raw/{slug}/`, auto-discover mp4 + gt csv + intrinsics.
2. Load poses, select frames (altitude, spacing, heading — not raw FPS).
3. Extract frames with ffmpeg.
4. Write ODM layout: `images/` + `geo.txt` (always — georef is not optional).

### Output

`odm-results/{timestamp:YYYYMMDDHHMMSS}_{slug}` e.g 
```text
odm-results/20260616123251_0088_20260122_eitan_1/
├── run.json
└── odm_input/
    ├── images/
    │   ├── frame_0001.jpg
    │   └── ...
    └── geo.txt
```

---

## Step 2 — ODM

Stock image for now: `opendronemap/odm:3.5.1`. Pin the tag in `docker-compose.yml`. If we later need pinned system libs or plugins, we build a thin wrapper image — still no ODM fork.

ODM reads `odm_input/` and writes into the same run folder:

```text
odm-results/20260616123251_0088_20260122_eitan_1/
├── odm_input/          # input
├── odm_orthophoto/
├── odm_dem/
├── odm_report/
└── odm_texturing/
```

ODM flags pass through the wrapper:

```bash
./scripts/run_odm.sh 0088_20260122_eitan_1 --fast-orthophoto
./scripts/run_odm.sh 0088_20260122_eitan_1 --pc-quality high --skip-3dmodel
```

### GPU in compose

Use Compose v2 `gpus: all` on the `odm` service — this is the reliable path for `docker compose run` on a single node:

```yaml
odm:
  image: opendronemap/odm:3.5.1
  gpus: all
```

`runtime: nvidia` is the older Docker CLI approach; `deploy.resources.reservations.devices` targets Swarm and is easy to get wrong with `compose run`. Stick with `gpus: all` plus NVIDIA Container Toolkit on the host. Builder stays CPU-only.

---

## Docker: dev mounts and EC2

### One compose file, live mounts everywhere (recommended for now)

| Mount | Purpose |
|-------|---------|
| `.` → `/app` | live code — edit in VS Code, rerun without rebuild |
| `${DATA_ROOT}` → `/data` | `raw/` + `odm-results/` |
| `configs/` → `/configs` | shared `builder.yaml` |
| `~/.aws` → `/root/.aws:ro` | local S3 credentials only |

**Why live mounts on EC2 too?** You're still experimenting. VS Code Remote SSH + mounted source means one `docker-compose.yml`, no rebuild cycle, same ergonomics as laptop. The only host deps are Docker + NVIDIA toolkit.

**When to bake an image later:** runs are reproducible, CI publishes to ECR, or you stop editing code on the instance. Optional `docker-compose.deploy.yml` drops the `.:/app` mount and copies code into the image at build time. Not needed until then.

---

## Local vs EC2 — same experience

There is no separate "S3 workflow" vs "local workflow". There are two **hosts**, one layout:

| | Laptop | EC2 |
|---|--------|-----|
| `DATA_ROOT` | `/home/bfialkoff/s3` | `/data` (EBS volume) |
| Populate `raw/` | already synced / s3 mount | `aws s3 sync` or S3 mount — outside pipeline |
| Commands | `./scripts/build.sh <slug>` | identical |
| Credentials | `AWS_PROFILE` + `~/.aws` mount | IAM instance role |
| Code | live mount | live mount (for now) |

The pipeline reads and writes under `{data_root}/raw/` and `{data_root}/odm-results/`. Whether `data_root` is a local disk path or backed by S3 sync is your choice **before** you run the script — the builder doesn't care.

```bash
# Laptop
export DATA_ROOT=/home/bfialkoff/s3
./scripts/build.sh 0088_20260122_eitan_1

# EC2 — same commands after bootstrap
export DATA_ROOT=/data
./scripts/build.sh 0088_20260122_eitan_1
```

Optional: sync results back to S3 after ODM (manual, outside the critical path):

```bash
aws s3 sync "$DATA_ROOT/odm-results/20260616123251_0088_20260122_eitan_1/" \
  "s3://my-bucket/odm-results/20260616123251_0088_20260122_eitan_1/"
```

Mirror structure, same prefixes — `raw/` and `odm-results/` on disk match S3.

---

## AWS

### Credentials

| Host | Mechanism |
|------|-----------|
| Laptop | `AWS_PROFILE` in `.env`; `~/.aws` mounted read-only into builder |
| EC2 | IAM instance role — no keys, no profile |

```bash
aws sts get-caller-identity    # confirm account
```

Never put access keys in repo, Dockerfile, or config.

### IAM (EC2 instance role)

- `s3:GetObject`, `s3:ListBucket` on `arn:aws:s3:::BUCKET/raw/*`
- `s3:PutObject`, `s3:ListBucket` on `arn:aws:s3:::BUCKET/odm-results/*`

Only needed if you sync to/from S3 on that instance. Pure local `DATA_ROOT` on laptop needs no IAM.

### S3 mirror (optional, manual)

```text
s3://bucket/raw/0088_20260122_eitan_1/...
s3://bucket/odm-results/20260616123251_0088_20260122_eitan_1/...
```

Same tree as on disk. Sync is a separate step you run when you want it — not built into the job loop yet.

---

## EC2 bootstrap

Host installs **only** Docker + NVIDIA Container Toolkit:

```bash
git clone <repo> ~/scene_reconstruction
cd ~/scene_reconstruction
./scripts/bootstrap-ec2.sh
# set DATA_ROOT in .env, mount or sync data into /data/raw/
docker compose build
./scripts/build.sh 0088_20260122_eitan_1
```

Recommended: `g4dn.xlarge`, 500GB EBS, IAM role attached. Instance is disposable; `raw/` and `odm-results/` are the durable paths (on EBS or synced to S3).

---

## Project layout (to implement)

```text
scene_reconstruction/
├── docs/PLAN.md
├── configs/
│   └── builder.yaml              # pipeline params + data_root default
├── src/scene_recon/
│   ├── cli.py
│   ├── datum.py                  # slug resolve + auto-discovery
│   ├── build_dataset.py
│   ├── frame_select.py
│   └── paths.py                  # {slug}_{timestamp} naming
├── scripts/
│   ├── bootstrap-ec2.sh
│   ├── build.sh                  # ./build.sh <slug>
│   ├── run_odm.sh                # ./run_odm.sh <slug> [odm flags]
│   └── run_all.sh
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Implementation order

1. `datum.py` — slug → discover mp4, gt csv, intrinsics; validate layout.
2. `paths.py` — stamp `{slug}_{timestamp}`, write `run.json`, resolve latest run.
3. `frame_select.py` + `build_dataset.py` — selection + ODM layout + `geo.txt`.
4. `Dockerfile` + `docker-compose.yml` — builder + odm (`gpus: all`).
5. `scripts/build.sh` + `run_odm.sh` — slug-only interface.
6. End-to-end on `0088_20260122_eitan_1`.
7. EC2 bootstrap + verify same commands with `DATA_ROOT=/data`.

---

## Explicitly later

| Item | Why wait |
|------|----------|
| Job queue / worker loop | Manual slug trigger is enough |
| Terraform | Hand-launch EC2 |
| Custom ODM image | Stock works until it doesn't |
| `docker-compose.deploy.yml` / ECR | Live mounts fine while experimenting |
| GCSV/BIN sync | Upstream; slug folder is already prepared |
| Auto S3 sync in pipeline | Manual `aws s3 sync` keeps v0 simple |

---

## Quick reference

| Question | Answer |
|----------|--------|
| Per-dataset identifier? | Slug folder name under `raw/` |
| Video / poses in config? | No — auto-discovered from slug folder |
| Output folder name? | `{timestamp}_{slug}/` under `odm-results/` — automatic |
| `geo.txt`? | Always written |
| Docker for builder? | Yes |
| Docker for ODM? | Yes — stock image, pinned tag |
| GPU? | `gpus: all` on `odm` service |
| Local vs EC2 commands? | Identical; only `DATA_ROOT` differs |
| Config file? | One `configs/builder.yaml` for pipeline params |
| How to run? | `./scripts/build.sh <slug>` then `./scripts/run_odm.sh <slug>` |
