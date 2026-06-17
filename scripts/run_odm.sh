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
docker compose run --rm odm \
  --project-path "$CONTAINER_PROJECT_PATH" \
  "$ODM_DATASET" \
  "${ODM_ARGS[@]}"
