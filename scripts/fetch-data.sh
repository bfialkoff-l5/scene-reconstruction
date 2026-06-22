#!/usr/bin/env bash
# Pull a dataset onto the box into the layout the builder expects. Run after `aws sso login`.
# Usage: ./scripts/fetch-data.sh <slug> <map.gpkg>     e.g. ./scripts/fetch-data.sh 0088_20260122_eitan_1 Eitan.gpkg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; source .env; set +a; }   # DATA_ROOT, AWS_PROFILE, AWS_REGION

SLUG="${1:?usage: $0 <slug> <map.gpkg>}"
MAP="${2:?usage: $0 <slug> <map.gpkg>}"
: "${DATA_ROOT:?set DATA_ROOT in .env}"
: "${AWS_PROFILE:?set AWS_PROFILE in .env}"   # exported above, so aws CLI picks it up

BUCKET="s3://line5-localization-evaluation-data-939103584914-eu-north-1-an"
FLIGHT_PREFIX="Flight logs/Flight 000"        # slug folders live under this prefix in-bucket

echo "==> raw flight -> $DATA_ROOT/raw/$SLUG  (skipping unused AvatarG* stream)"
aws s3 sync "$BUCKET/$FLIGHT_PREFIX/$SLUG/" "$DATA_ROOT/raw/$SLUG/" --exclude "AvatarG*"

echo "==> DTM -> $DATA_ROOT/geo-resources/DSM/israelDTM.gpkg"
mkdir -p "$DATA_ROOT/geo-resources/DSM"
aws s3 cp "$BUCKET/israelDTM.gpkg" "$DATA_ROOT/geo-resources/DSM/israelDTM.gpkg"

echo "==> reference ortho -> $DATA_ROOT/geo-resources/orthophoto/$MAP"
mkdir -p "$DATA_ROOT/geo-resources/orthophoto"
aws s3 cp "$BUCKET/Maps/$MAP" "$DATA_ROOT/geo-resources/orthophoto/$MAP"

echo "done. next: ./scripts/build.sh $SLUG --terrain-gpkg /geo/DSM/israelDTM.gpkg --ray-grid 48 27"
