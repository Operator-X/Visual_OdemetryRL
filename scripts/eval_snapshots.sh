#!/usr/bin/env bash
# Evaluate the intermediate policy snapshots of finished runs (training trend without new training).
#   scripts/eval_snapshots.sh            -> results/eval/<dataset>/snapshots/<iter>/<run>.csv (1 repeat = seed 0)
# Summarize with scripts/snapshot_trends.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNS=${RUNS:-"bc_ppo_s0 bc_ppo_s1 bc_ppo_s2 constrained_lw_s0 constrained_lw_s1 constrained_lw_s2 residual_s0 residual_s1
residual_s2 bc_constrained_lw_s0 ppo_v2_s0 bc_ppo_v2_s0 constrained_v2_s0 shadow_rel_v2_s0 ppo15_s0"}
for ds in ${DATASETS:-tum tartanair}; do
  for it in $(ls runs/ppo15_s0/Policy | sed -n 's/^\(iter_[0-9]*\)\.pth$/\1/p'); do
    sel=()
    for r in $RUNS; do [ -f "runs/$r/Policy/$it.pth" ] && sel+=("runs/$r"); done
    [ ${#sel[@]} -eq 0 ] && continue
    echo "[snapshots] $ds $it: ${#sel[@]} runs" >&2
    $PY scripts/evaluate.py --runs "${sel[@]}" --checkpoint "$it" --dataset "$ds" --repeats 1 \
      --out-tag "snapshots/$it" 2>&1 | grep -E "ATE|snapshots" || true
  done
done
