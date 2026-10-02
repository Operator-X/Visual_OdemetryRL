#!/usr/bin/env bash
# Authors' PPO (= reward1_baseline), 3 seeds x 400k, now with policy snapshots every 30 iterations.
# Run: caffeinate -i scripts/experiments/ppo_snap.sh > runs/logs/ppo_snap.log 2>&1   (~55 min on the M3 Pro)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for s in 0 1 2; do
  echo "[ppo_snap] train seed $s $(date +%T)"
  $PY scripts/train.py --name ppo_snap --set total_timesteps=400000 val_interval=30 obs_rms_warmup_steps=200 seed=$s "data.train_include=[japanesealley/Easy,carwelding/Easy,westerndesert/Easy]" \
    > runs/logs/ppo_snap_s$s.out 2>&1 || echo "[ppo_snap] TRAIN FAILED seed $s"
done
R="runs/ppo_snap_s0 runs/ppo_snap_s1 runs/ppo_snap_s2"
for ds in tum tartanair; do
  echo "[ppo_snap] eval final $ds $(date +%T)"
  $PY scripts/evaluate.py --runs $R --dataset $ds --repeats 3 --out-tag ppo_snap 2>&1 | grep "ATE"
done
echo "[ppo_snap] eval snapshots $(date +%T)"
RUNS="ppo_snap_s0 ppo_snap_s1 ppo_snap_s2" ./scripts/eval_snapshots.sh 2>&1 | grep -E "ATE|snapshots"
echo "[ppo_snap] DONE $(date +%T)"
