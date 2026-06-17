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

RECORD="${1:?usage: $0 <record-abspath-or-slug> [--run YYYYMMDDHHMMSS]}"
shift

if [[ "$RECORD" != /* ]]; then
  RECORD="$DATA_ROOT/raw/$RECORD"
fi

CONTAINER_RECORD="$(record_container_path "$RECORD")"

docker compose run --rm builder prepare-odm "$CONTAINER_RECORD" "$@"
