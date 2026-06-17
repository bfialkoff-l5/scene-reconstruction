#!/usr/bin/env bash
# Shared path helpers for docker compose (host DATA_ROOT mounted at /data).

record_container_path() {
  local record="$1"
  if [[ "$record" == "$DATA_ROOT"/* ]]; then
    echo "/data/${record#"$DATA_ROOT"/}"
  elif [[ "$record" == /data/* ]]; then
    echo "$record"
  else
    echo "ERROR: record path must be under DATA_ROOT ($DATA_ROOT): $record" >&2
    return 1
  fi
}

container_to_host_path() {
  local container_path="$1"
  if [[ "$container_path" == /data/* ]]; then
    echo "$DATA_ROOT/${container_path#/data/}"
  else
    echo "$container_path"
  fi
}
