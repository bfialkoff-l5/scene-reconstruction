#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_common.sh"
source "$ROOT/scripts/_paths.sh"

_cli_data_root="${DATA_ROOT:-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
[[ -n "${_cli_data_root}" ]] && DATA_ROOT="${_cli_data_root}"

: "${DATA_ROOT:?Set DATA_ROOT in .env or environment}"
export DATA_ROOT HOST_UID HOST_GID AWS_CREDENTIALS_MOUNT

ensure_data_dirs

RECORD="${1:?usage: $0 <record-abspath-or-slug> [-- odm args...]}"
shift

if [[ "$RECORD" != /* ]]; then
  RECORD="$DATA_ROOT/raw/$RECORD"
fi

RUN_ARGS=()
ODM_ARGS=()
PARSING_ODM=0
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    PARSING_ODM=1
    continue
  fi
  if [[ $PARSING_ODM -eq 1 ]]; then
    ODM_ARGS+=("$arg")
  else
    RUN_ARGS+=("$arg")
  fi
done

ODM_DATASET="odm_input"

CONTAINER_RECORD="$(record_container_path "$RECORD")"

RUN_DIR="$(docker compose run --rm -T builder resolve-run "$CONTAINER_RECORD" "${RUN_ARGS[@]}" | tail -n1)"
CONTAINER_PROJECT_PATH="$RUN_DIR"

echo "ODM project path: $CONTAINER_PROJECT_PATH (dataset: $ODM_DATASET)"
if [[ ${#ODM_ARGS[@]} -eq 0 ]]; then
  ODM_ARGS=(--fast-orthophoto)
fi

OPTS_HOST="$(container_to_host_path "$CONTAINER_PROJECT_PATH")/$ODM_DATASET/odm_options.json"

# Auto-scale --matcher-neighbors to hold a fixed matching baseline reach for this
# selection's frame density (build/prepare-odm wrote the value into odm_options.json).
# A fixed neighbour count shrinks the baseline as spacing densifies, which silently
# wrecks triangulation -- pin the reach instead. Skipped if the caller set it explicitly.
if [[ ! " ${ODM_ARGS[*]} " == *" --matcher-neighbors "* ]]; then
  MN="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('matcher_neighbors',0))" "$OPTS_HOST" 2>/dev/null || echo 0)"
  if [[ -n "$MN" && "$MN" != "0" ]]; then
    ODM_ARGS+=(--matcher-neighbors "$MN")
    echo "auto matcher-neighbors: $MN (co-visibility-derived; covers cross-track / loop-closure pairs)"
  else
    echo "WARN: no auto matcher-neighbors (odm_options.json missing/0); re-run build to regenerate" >&2
  fi
fi

# Keep OpenSfM's GPS position-prior weight pinned to the prepared baseline. ODM can preserve
# stale/loose dop values in opensfm/exif when reusing an input tree, so the patched OpenSfM
# also receives this as an explicit metadata override.
GPS_ACCURACY="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('gps_accuracy',''))" "$OPTS_HOST" 2>/dev/null || true)"
if [[ -n "$GPS_ACCURACY" ]]; then
  if [[ ! " ${ODM_ARGS[*]} " == *" --gps-accuracy "* ]]; then
    ODM_ARGS+=(--gps-accuracy "$GPS_ACCURACY")
  fi
  echo "gps accuracy: $GPS_ACCURACY m (also exported as SCENE_RECON_GPS_DOP)"
else
  echo "WARN: no gps_accuracy in odm_options.json; OpenSfM will use EXIF/ODM defaults" >&2
fi

# Bind our lab calibration (intrinsicK.csv -> cameras.json). Without a seed ODM defaults
# to a 0.85 focal prior on our EXIF-less frames and the bundle adjust diverges (principal
# point off-image, OpenMVS fuses 0 points). cameras.json is keyed to ODM's detected camera
# id so the override binds; see Camera.camera_id().
#
# By default the calibration is *locked* (--use-fixed-camera-params). Set OPTIMIZE_CAMERAS=1
# to instead use cameras.json only as the BA *initial guess* and let bundle adjustment
# refine the intrinsics -- worth doing now that strong cross-track matching constrains the
# solve (the historic 0-point divergence was a weak-matching symptom). Watch the point
# count and opensfm/stats residuals_v2 to confirm it converged.
CAMERAS_JSON="$CONTAINER_PROJECT_PATH/$ODM_DATASET/cameras.json"
CAMERA_ARGS=(--cameras "$CAMERAS_JSON")
if [[ "${OPTIMIZE_CAMERAS:-0}" == "1" ]]; then
  echo "OPTIMIZE_CAMERAS=1: seeding cameras.json, letting bundle adjustment refine intrinsics (no --use-fixed-camera-params)"
else
  CAMERA_ARGS+=(--use-fixed-camera-params)
fi

# Hand the patched OpenSfM our exact per-image attitude priors (build wrote
# rotation_priors.json next to geo.txt). extract_metadata reads this path, sets each
# image's opk, and bundle adjustment anchors rotation to it -- the fix for the global
# rotation flip. Only set on the patched image (ODM_IMAGE); stock ODM ignores the env var.
ENV_ARGS=()
if [[ -n "${GPS_ACCURACY:-}" ]]; then
  ENV_ARGS+=(-e "SCENE_RECON_GPS_DOP=$GPS_ACCURACY")
fi
PRIORS_CONTAINER="$CONTAINER_PROJECT_PATH/$ODM_DATASET/rotation_priors.json"
PRIORS_HOST="$(container_to_host_path "$CONTAINER_PROJECT_PATH")/$ODM_DATASET/rotation_priors.json"
if [[ -f "$PRIORS_HOST" ]]; then
  ENV_ARGS+=(-e "SCENE_RECON_ROTATION_PRIORS=$PRIORS_CONTAINER")
  echo "rotation priors: $PRIORS_CONTAINER (BA absolute rotation prior; requires patched ODM_IMAGE)"
else
  echo "WARN: no rotation_priors.json (re-run build); BA rotation will be unconstrained" >&2
fi

docker compose run --rm "${ENV_ARGS[@]}" odm \
  --project-path "$CONTAINER_PROJECT_PATH" \
  "$ODM_DATASET" \
  "${CAMERA_ARGS[@]}" \
  "${ODM_ARGS[@]}"
