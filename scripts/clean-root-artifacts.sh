#!/usr/bin/env bash
# Remove root-owned artifacts left by pre-fix Docker runs.
# Safe to run anytime; only deletes root-owned paths under the repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mapfile -t ROOT_OWNED < <(find "$ROOT" -user root 2>/dev/null || true)

if [[ ${#ROOT_OWNED[@]} -eq 0 ]]; then
  echo "No root-owned files under $ROOT"
  exit 0
fi

echo "Root-owned paths under repo (${#ROOT_OWNED[@]}):"
printf '  %s\n' "${ROOT_OWNED[@]}"

if ! rm -rf "${ROOT_OWNED[@]}" 2>/dev/null; then
  echo "Some paths need elevated removal:"
  sudo rm -rf "${ROOT_OWNED[@]}"
fi

REMAINING="$(find "$ROOT" -user root 2>/dev/null | wc -l)"
echo "Done. Root-owned remaining: $REMAINING"
