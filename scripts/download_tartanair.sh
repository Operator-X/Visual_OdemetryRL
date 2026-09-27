#!/usr/bin/env bash
# Download + unzip TartanAir (v1) image_left zips for chosen scenes from the AirLab (CMU) server.
# Usage: scripts/download_tartanair.sh [Easy|Hard] scene1 scene2 ...
#   e.g. scripts/download_tartanair.sh Easy japanesealley carwelding westerndesert
# Scene sizes: see castacks/tartanair_tools download_training_zipfiles.txt. Resumable (curl -C -).
set -euo pipefail

LEVEL="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/TartanAir"
ZIPS="$ROOT/data/_zips"
BASE="https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair"
mkdir -p "$OUT" "$ZIPS"

for scene in "$@"; do
  zip="$ZIPS/${scene}_${LEVEL}_image_left.zip"
  echo "== $scene/$LEVEL: downloading"
  # Abort if <10 KB/s for 30 s (server connections sometimes stall silently), then resume where it stopped.
  for attempt in $(seq 1 50); do
    if curl -fL --retry 5 --speed-limit 10000 --speed-time 30 -C - -o "$zip" "$BASE/$scene/$LEVEL/image_left.zip"; then
      break
    fi
    echo "   stalled/failed (attempt $attempt), resuming in 5 s"; sleep 5
  done
  echo "== $scene/$LEVEL: verifying + unzipping"
  unzip -tq "$zip"
  unzip -q -o "$zip" -d "$OUT"
done
echo "done -> $OUT"
