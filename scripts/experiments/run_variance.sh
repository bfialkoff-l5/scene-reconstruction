#!/usr/bin/env bash
# Variance / determinism experiment runner.
#
# Runs the SAME byte-identical input through OpenSfM N times with a given set of extra ODM
# flags, then reports the CE90 distribution. Use it to test determinism levers:
#
#   run_variance.sh flann       3 --                                          # baseline (FLANN)
#   run_variance.sh bruteforce  3 -- --matcher-type bruteforce
#   run_variance.sh single      2 -- --matcher-type bruteforce --max-concurrency 1 --no-gpu
#   run_variance.sh triangulate 3 -- --sfm-algorithm triangulation
#
# CE90 comes from opensfm/stats/stats.json (we --end-with opensfm to save time).
# A "flip" = low reprojection error but high CE90 (internally consistent, globally rotated).
set -uo pipefail

REPO=/home/bfialkoff/projects/scene_reconstruction/flip-determinism   # run scripts from here
SLUG=/home/bfialkoff/s3/odm-results/0088_20260122_eitan_1
RECORD=0088_20260122_eitan_1

# Source input dir (md5-identical to the good run 20260624193806). Override with SRC=...
# If unset/missing, build a fresh one from clean main's pipeline first.
SRC="${SRC:-$SLUG/runs/20260628040841/odm_input}"

LABEL="${1:?usage: run_variance.sh <label> <N> -- <extra odm args>}"; shift
N="${1:?need N}"; shift
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

export ODM_IMAGE=opendronemap/odm:gpu   # force STOCK image (a stale env var can leak the patched one)
cd "$REPO"

LOG="/tmp/var_${LABEL}.log"; : > "$LOG"
echo "label=$LABEL N=$N image=$ODM_IMAGE src=$SRC extra='${EXTRA[*]}'" | tee -a "$LOG"
[[ -d "$SRC/images" ]] || { echo "SRC missing ($SRC). Build a fresh input first (see docs/FLIP_DETERMINISM.md)." | tee -a "$LOG"; exit 1; }

for i in $(seq 1 "$N"); do
  TS="$(date -u -d "+$((i*3)) minute" +%Y%m%d%H%M%S)"
  DST="$SLUG/runs/$TS/odm_input"; mkdir -p "$DST"
  for f in cameras.json geo.txt images.json img_list.txt matching_profile.json odm_options.json; do
    [[ -f "$SRC/$f" ]] && cp -a "$SRC/$f" "$DST/$f"
  done
  cp -al "$SRC/images" "$DST/images"
  echo "===== [$LABEL] run $i/$N TS=$TS $(date -u +%H:%M:%SZ) =====" | tee -a "$LOG"
  ./scripts/run_odm.sh "$RECORD" --run "$TS" -- \
      --auto-boundary --gps-accuracy 3 --fast-orthophoto --end-with opensfm "${EXTRA[@]}" \
      > "/tmp/var_${LABEL}_${TS}.log" 2>&1
  ST="$DST/opensfm/stats/stats.json"
  if [[ -f "$ST" ]]; then
    python3 - "$ST" "$LABEL" "$i" "$TS" <<'PY' | tee -a "$LOG"
import json,sys
s=json.load(open(sys.argv[1])); g=s.get('gps_errors',{}); rs=s.get('reconstruction_statistics',{})
ce=g.get('ce90'); ce=ce if isinstance(ce,(int,float)) else float('nan')
print("RESULT [%s] run %s TS=%s  ce90=%.2f  le90=%.2f  reproj=%.3f  shots=%s/%s  %s"%(
  sys.argv[2], sys.argv[3], sys.argv[4], ce, g.get('le90',float('nan')),
  rs.get('reprojection_error_normalized',float('nan')),
  rs.get('reconstructed_shots_count'), rs.get('initial_shots_count'),
  "GOOD" if ce<15 else "FLIP"))
PY
  else
    echo "RESULT [$LABEL] run $i TS=$TS FAILED (no stats) -- see /tmp/var_${LABEL}_${TS}.log" | tee -a "$LOG"
  fi
done
echo "===== [$LABEL] DONE =====" | tee -a "$LOG"
grep -E "^RESULT" "$LOG"
