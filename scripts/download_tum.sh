#!/usr/bin/env bash
# Download + extract the 9 TUM-RGBD freiburg1 sequences used by the authors (reference tum_loader test_split)
# into data/TUM-RGBD/. Resumable; stalls are detected and resumed.
# Usage: scripts/download_tum.sh [seq ...]     (default: all 9, e.g. 'xyz desk')
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/TUM-RGBD"
ZIPS="$ROOT/data/_zips/tum"
BASE="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
mkdir -p "$OUT" "$ZIPS"

SEQS=("$@"); [ ${#SEQS[@]} -eq 0 ] && SEQS=(360 desk desk2 floor plant room rpy teddy xyz)

for s in "${SEQS[@]}"; do
  name="rgbd_dataset_freiburg1_$s"; tgz="$ZIPS/$name.tgz"
  if [ -f "$OUT/$name/rgb.txt" ] && [ -f "$OUT/$name/groundtruth.txt" ]; then
    echo "$(date +%T) $name: already extracted"; continue
  fi
  for attempt in $(seq 1 30); do
    if curl -fL --retry 5 --speed-limit 10000 --speed-time 30 -C - -o "$tgz" "$BASE/$name.tgz"; then break; fi
    echo "$(date +%T) $name: stalled/failed (attempt $attempt), resuming in 5 s"; sleep 5
  done
  echo "$(date +%T) $name: extracting"
  tar -xzf "$tgz" -C "$OUT" && echo "$(date +%T) $name: OK ($(ls "$OUT/$name/rgb" | wc -l | tr -d ' ') images)"
done
echo "$(date +%T) done -> $OUT"
