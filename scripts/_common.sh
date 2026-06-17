#!/usr/bin/env bash
# Host user/group for container file ownership (must match volume writer).
# UID/GID are readonly in bash — use HOST_UID/HOST_GID for compose.
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

# Credentials mount: use host ~/.aws when present, else empty stub (EC2 uses IAM role).
_common_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${HOME}/.aws" ]]; then
  export AWS_CREDENTIALS_MOUNT="${HOME}/.aws"
else
  export AWS_CREDENTIALS_MOUNT="${_common_root}/.aws-empty"
  mkdir -p "${AWS_CREDENTIALS_MOUNT}"
fi

ensure_data_dirs() {
  : "${DATA_ROOT:?Set DATA_ROOT in .env or environment}"
  mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/odm-results"
  if [[ ! -w "${DATA_ROOT}/odm-results" ]]; then
    echo "ERROR: ${DATA_ROOT}/odm-results is not writable by $(id -un) (uid $(id -u))." >&2
    echo "Likely created by an earlier root-owned container run. Remove once, then rerun:" >&2
    echo "  sudo rm -rf ${DATA_ROOT}/odm-results && mkdir -p ${DATA_ROOT}/odm-results" >&2
    exit 1
  fi
}
