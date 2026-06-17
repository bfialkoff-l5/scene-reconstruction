#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD="${1:?usage: $0 <record-abspath-or-slug> [-- odm args...]}"
shift

"$ROOT/scripts/build.sh" "$RECORD"
"$ROOT/scripts/run_odm.sh" "$RECORD" -- "$@"
