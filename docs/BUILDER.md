# Builder Design — Record, Scores-First, Then Selection

Implemented in `src/scene_recon/`. See README for usage.

---

## CLI

```bash
scene-recon build /home/bfialkoff/s3/raw/0088_20260122_eitan_1
```

Canonical input is **abspath** to the slug folder. Shell scripts may expand slug → `$DATA_ROOT/raw/$slug`.

---

## Record

```python
Record.from_path(abspath) -> Record
```

| Field | Source |
|-------|--------|
| `path` | abspath to slug folder |
| `slug` | `path.name` |
| `video` | `*.mp4` with matching `_derived/gt_<stem>.csv` |
| `poses_path` | `_derived/gt_<stem>.csv` |
| `intrinsics` | `intrinsicK.csv` |
| `stream_id` | mp4 stem |

`data_root` inferred as parent of `raw/` for `odm-results/` output.

---

## Core principle: score every frame, then select

**Do not** prefilter poses and decode a subset.

**Do** decode and score **every** candidate frame, write metrics back into the DataFrame, **then** run keyframe selection using scores + poses together.

```text
Record
  → load_poses()           # one row per FrameNumber
  → init_candidates()      # same rows, empty score columns
  → score_all_frames()     # full video pass, fill quality columns
  → select_keyframes()     # spatial bins + quality + pose diversity + coverage
  → export_selected()      # PNG {FrameNumber:06d}.png + geo.txt
```

Selection is a pure table operation on a complete candidate set. No selection logic runs before scores exist.

---

## Tables (principled DataFrames)

### Poses (from gt CSV)

Fixed schema: `FrameNumber`, `TimeUS`, `easting`, `northing`, `altamsl`, `utm_zone`, `roll_rad`, `pitch_rad`, `yaw_rad`, …

Indexed by `FrameNumber`. Sorted.

### Candidates (poses + scores + selection flags)

Same rows as poses. Additional columns:

| Column | When set |
|--------|----------|
| `feature_count` | `score_all_frames` |
| `sharpness` | `score_all_frames` |
| `quality_score` | `score_all_frames` |
| `cell_x`, `cell_y` | `select_keyframes` (spatial bin) |
| `selected` | `select_keyframes` |
| `reject_reason` | `select_keyframes` (optional, for debugging) |

Vectorized groupby / idxmax / masks — no row-object iteration for bulk ops.

Small helpers for pose geometry (translation m, rotation deg from full RPY) used inside selection.

---

## Video: perfect frame semantics

Single **forward** pass through the video. No random seek.

```text
targets = sorted(candidates.FrameNumber)
idx = 0
for each grab/retrieve:
    if idx in targets:
        score_image(bgr) → write row[idx] feature_count, sharpness, quality_score
    idx += 1
```

- Frame index must match `FrameNumber` in poses (validate on first rows).
- **Threading:** decode stays sequential; hand `bgr` arrays to a worker pool for ORB + Laplacian.
- Second export pass only if needed; prefer retaining decoded arrays for selected rows in memory or a temp cache if RAM allows — otherwise re-decode selected indices only (small set).

---

## Image quality scoring (every frame)

Per decoded frame:

```text
feature_count  = ORB keypoint count (or similar)
sharpness      = variance of Laplacian
quality_score  = w_feat * norm(features) + w_sharp * norm(sharpness)
```

Weights are **constants in `frame_select.py`** for now; optional `tuning.yaml` later.

Purpose: rank competing observations of the same region, not pick a global “best image.”

---

## Keyframe selection (after scores exist)

Runs only on the fully scored `candidates` DataFrame.

### 1. Spatial grouping

Bin by `(easting, northing)` into cells (e.g. 5 m × 5 m; constant for now).

```python
candidates["cell_x"] = (easting // BIN_SIZE_M).astype(int)
candidates["cell_y"] = (northing // BIN_SIZE_M).astype(int)
```

### 2. Best observation per bin

Within each cell, prefer highest `quality_score`. Removes hover / slow-flight redundancy.

### 3. Pose diversity

Among survivors (or when adding to the set), require meaningful viewpoint change:

- translation difference
- rotation difference from **full** roll/pitch/yaw (not yaw alone)

Keep a high-quality frame even when spatially close if rotation delta is large (parallax).

### 4. Coverage

Avoid large spatial gaps between selected keyframes. Balance quality, coverage, and novelty — not any single metric alone.

### 5. Cap

`MAX_KEYFRAMES` constant (or yaml later).

### Information gain (target framing)

```text
information_gain ≈ quality × coverage_gain × pose_novelty
```

Implement first as explicit gates; refactor to scored gain when constants stabilize.

### Smoke mode (step 1 implementation)

Until real rules land, `select_keyframes` can still use `FRAME_SKIP` on the scored table — proves pipeline with `quality_score` populated but ignored.

---

## Export

| Artifact | Rule |
|----------|------|
| Images | `{FrameNumber:06d}.png` (8-bit BGR PNG) |
| `geo.txt` | one line per selected frame, filename matches |
| `run.json` | record paths, `n_candidates`, `n_selected`, policy snapshot, `selected_frame_numbers` |

Output run dir: `{data_root}/odm-results/{timestamp}_{slug}/odm_input/`.

---

## Config

**Now:** all thresholds in `frame_select.py` / `scoring.py` as module constants.

**Later:**

```bash
scene-recon build /path/to/record --config tuning.yaml
```

YAML tunes selection weights/thresholds only — not paths, not slug.

---

## Module layout

```text
record.py       Record.from_path
schema.py       column defs, validators
poses.py        load_poses
candidates.py   init_candidates
video.py        sequential decode
scoring.py      ORB, Laplacian, quality_score
frame_select.py select_keyframes + constants
export.py       PNG, geo.txt, run.json
build.py        pipeline wire-up
cli.py          build <abspath>
```

---

## Implementation order

1. `Record` + `load_poses` + `init_candidates`
2. `video.py` + `scoring.py` → **score all frames** into DataFrame
3. `export.py` — PNG + geo.txt with `{FrameNumber:06d}.png`
4. `select_keyframes` smoke (`FRAME_SKIP`) on scored table
5. CLI `build <abspath>`, update scripts
6. Real selection: spatial bin → best per bin → diversity → coverage
7. Optional `tuning.yaml`

---

## Unchanged

- Docker two-step (builder + stock ODM)
- `HOST_UID` / `HOST_GID` file ownership
- ODM: `--project-path <run_dir> odm_input`
- Manual trigger, no job queue
