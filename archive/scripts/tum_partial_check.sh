#!/usr/bin/env bash
# Wait until >= N TUM sequences are fully extracted, link them into data/TUM-RGBD-ready/, then evaluate
# SVO's heuristics (3 repeats) and one pilot policy (1 repeat) on them.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; N="${1:-3}"; LOG="$ROOT/data/tum_download.log"
until [ "$(grep -c ': OK (' "$LOG")" -ge "$N" ] || ! pgrep -f download_tum.sh >/dev/null; do sleep 30; done
READY="$ROOT/data/TUM-RGBD-ready"; rm -rf "$READY"; mkdir -p "$READY"
for name in $(grep ': OK (' "$LOG" | awk '{print $2}' | tr -d ':'); do ln -s "$ROOT/data/TUM-RGBD/$name" "$READY/$name"; done
echo "sequences: $(ls "$READY" | tr '\n' ' ')"
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/scripts/evaluate.py" --heuristic --dataset tum --data-dir data/TUM-RGBD-ready --repeats 3 --out-tag tum_check 2>&1 | grep -E "ATE|Traceback|Error|SystemExit"
"$PY" "$ROOT/scripts/evaluate.py" --runs "$ROOT/runs/pilot2_baseline_s0" --dataset tum --data-dir data/TUM-RGBD-ready --repeats 1 --out-tag tum_check_policy 2>&1 | grep -E "ATE|Traceback|Error|SystemExit"
