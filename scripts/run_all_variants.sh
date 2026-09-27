#!/usr/bin/env bash
# Train the baseline + every variant in configs/variants/ one after another, then evaluate all of them.
# Usage: scripts/run_all_variants.sh <tag> <total_timesteps> [repeats] [extra --set overrides...]
#   e.g. scripts/run_all_variants.sh pilot 20000 1          (pipeline test, ~15 min on the M3 Pro)
#        scripts/run_all_variants.sh r1m 1000000 3          (real pilot, ~6 h)
set -uo pipefail

TAG="$1"; STEPS="$2"; REPEATS="${3:-1}"; shift 3 || shift $#
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/runs/${TAG}_log.txt"
mkdir -p "$ROOT/runs"; : > "$LOG"

VARIANTS=(baseline $(ls "$ROOT/configs/variants" | sed 's/\.yaml$//'))
RUNS=()
for v in "${VARIANTS[@]}"; do
  start=$(date +%s)
  if [ "$v" = baseline ]; then vflag=(); else vflag=(--variant "$v"); fi
  caffeinate -i "$PY" "$ROOT/scripts/train.py" ${vflag[@]+"${vflag[@]}"} --name "${TAG}_${v}" \
      --set total_timesteps="$STEPS" val_interval=100000 "$@" > "$ROOT/runs/${TAG}_${v}.out" 2>&1
  rc=$?
  echo "$v exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$LOG"
  [ $rc -eq 0 ] && RUNS+=("$ROOT/runs/${TAG}_${v}_s0")
done

"$PY" "$ROOT/scripts/evaluate.py" --heuristic --runs ${RUNS[@]+"${RUNS[@]}"} --repeats "$REPEATS" --out-tag "$TAG" 2>&1 \
  | grep -v -E "^[IWE][0-9]{4}|^objc|^  |^loaded|WARNING: Logging" | tee -a "$LOG"
"$PY" "$ROOT/scripts/compare.py" --tag "$TAG" | tee -a "$LOG"
