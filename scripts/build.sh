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

RECORD="${1:?usage: $0 <record-abspath-or-slug>}"
shift

if [[ "$RECORD" != /* ]]; then
  RECORD="$DATA_ROOT/raw/$RECORD"
fi

CONTAINER_RECORD="$(record_container_path "$RECORD")"

# Stream logs/progress (stderr) live; capture the run dir the CLI echoes on stdout
# and translate the container /data path back to the host DATA_ROOT path.
CONTAINER_RUN_DIR="$(docker compose run --rm -T builder build "$CONTAINER_RECORD" "$@" | tail -n1)"
container_to_host_path "$CONTAINER_RUN_DIR"
