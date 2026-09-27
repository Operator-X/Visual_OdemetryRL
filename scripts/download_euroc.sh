#!/usr/bin/env bash
# Download the EuRoC MAV sequence archives from the ETH Research Collection (official host since the old
# robotics.ethz.ch server was retired). The server rate-limits (HTTP 429), so this backs off politely and resumes.
# Usage: scripts/download_euroc.sh [vicon_room1 vicon_room2 machine_hall]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS="$ROOT/data/_zips/euroc"
API="https://www.research-collection.ethz.ch/server/api/core/bitstreams"
mkdir -p "$ZIPS"

# DSpace bitstream ids (record doi:10.3929/ethz-b-000690084)
id_for() {
  case "$1" in
    machine_hall) echo 7b2419c1-62b5-4714-b7f8-485e5fe3e5fe ;;
    vicon_room1)  echo 02ecda9a-298f-498b-970c-b7c44334d880 ;;
    vicon_room2)  echo ea12bc01-3677-4b4c-853d-87c7870b8c44 ;;
    *) echo "unknown archive $1" >&2; exit 1 ;;
  esac
}

NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=(vicon_room1 vicon_room2 machine_hall)

for name in "${NAMES[@]}"; do
  out="$ZIPS/$name.zip"; id="$(id_for "$name")"; wait=60
  while :; do
    code=$(curl -sL -o /dev/null -w "%{http_code}" -r 0-0 --max-time 60 "$API/$id/content")
    if [ "$code" = 206 ] || [ "$code" = 200 ]; then
      echo "$(date +%T) $name: downloading"
      if curl -fL --retry 5 --speed-limit 10000 --speed-time 60 -C - -o "$out" "$API/$id/content"; then
        break
      fi
      echo "$(date +%T) $name: transfer interrupted, resuming in ${wait}s"
    else
      echo "$(date +%T) $name: HTTP $code (rate limited?), waiting ${wait}s"
    fi
    sleep "$wait"; wait=$(( wait * 2 > 600 ? 600 : wait * 2 ))   # cap backoff at 10 min
  done
  echo "$(date +%T) $name: verifying"
  unzip -tq "$out" && echo "$(date +%T) $name: OK ($(du -h "$out" | cut -f1))"
done
echo "$(date +%T) all done -> $ZIPS"
