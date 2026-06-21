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

# Auto-scale --matcher-neighbors to hold a fixed matching baseline reach for this
# selection's frame density (build/prepare-odm wrote the value into odm_options.json).
# A fixed neighbour count shrinks the baseline as spacing densifies, which silently
# wrecks triangulation -- pin the reach instead. Skipped if the caller set it explicitly.
if [[ ! " ${ODM_ARGS[*]} " == *" --matcher-neighbors "* ]]; then
  OPTS_HOST="$(container_to_host_path "$CONTAINER_PROJECT_PATH")/$ODM_DATASET/odm_options.json"
  MN="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('matcher_neighbors',0))" "$OPTS_HOST" 2>/dev/null || echo 0)"
  if [[ -n "$MN" && "$MN" != "0" ]]; then
    ODM_ARGS+=(--matcher-neighbors "$MN")
    echo "auto matcher-neighbors: $MN (holds fixed matching baseline reach for this selection)"
  else
    echo "WARN: no auto matcher-neighbors (odm_options.json missing/0); re-run build to regenerate" >&2
  fi
fi

# Bind our lab calibration (intrinsicK.csv -> cameras.json) and stop ODM from
# self-calibrating. Without this ODM defaults to a 0.85 focal prior on our
# EXIF-less frames and the bundle adjust diverges (principal point off-image,
# OpenMVS fuses 0 points). cameras.json is keyed to ODM's detected camera id so
# the override actually binds; see Camera.camera_id().
CAMERAS_JSON="$CONTAINER_PROJECT_PATH/$ODM_DATASET/cameras.json"
docker compose run --rm odm \
  --project-path "$CONTAINER_PROJECT_PATH" \
  "$ODM_DATASET" \
  --cameras "$CAMERAS_JSON" \
  --use-fixed-camera-params \
  "${ODM_ARGS[@]}"
