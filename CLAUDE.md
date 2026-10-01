# RL Visual Odometry — project context

> **Continue here:** `docs/research_plan.md` (next steps, comparative-study plan, method catalog, open decisions).
> History of every decision and finding: `docs/log.md`. All changes vs the authors: `docs/modifications.md`.

## Goal
Replicate **"Reinforcement Learning Meets Visual Odometry"** (Messikommer, Cioffi, Gehrig, Scaramuzza — ECCV 2024),
then extend it. A **course project** aimed at a **publishable paper** (possibly a comparative study of RL techniques,
publishable even if nothing beats the baseline). Results must be reproducible and comparable to the paper's tables, and
every design choice needs a reason we can write up.

## Current state (2026-09-29)
- **Setup validated on real data:** SVO's own rules on TUM-RGBD reproduce the paper's Table 2 SVO row (7/8 sequences
  finished, 8% mean deviation) with `tum_tuned.yaml`. Reference: `results/eval/tum/svo_rules_std`.
- **PPO (authors' setup + our fixes):** as robust as the tuned SVO rules with ~half the keyframes, but ~5-8% LESS
  accurate. Longer training (1.5M) does not help: reward/step is flat, the keyframe rate keeps falling.
- **Reward diagnosis:** the authors' reward is informative but weak (99.7% of steps positive on TUM), measures local
  (5-frame) error rather than drift, and a keyframe's benefit comes 4-5 frames later (discounted away at gamma=0.6).
- **Reward experiment (reward1, 3 seeds for baseline/long_window):** no variant beats the baseline robustly. Every reward
  change pushes the keyframe rate to an extreme (penalty 5e-3 -> ~no keyframes; long_window / gamma 0.9-0.99 -> keyframe
  every frame). The keyframe penalty weight decides the extreme. **Next idea: control the keyframe budget directly**
  (constrained RL / Lagrangian penalty to a target rate ~0.25-0.3) or a penalty sweep.
- **Behavior cloning -> PPO (3 seeds): beats PPO from scratch (-9%) and SVO's rules (-4%) on the paper's metric in
  every seed** (TUM ATE 0.488 +- 0.019 vs rules 0.510), no keyframe collapse. Gain concentrated on desk (-32%); worse on
  plant (+27%) and xyz (+90%, inherited from the clone's extra keyframes); geometric-mean per-seq ratio ~ PPO scratch.
  BUT on held-out TartanAir BC+PPO is LESS robust than PPO scratch and the rules (all 3 seeds) -> dataset-dependent.
  A fixed higher keyframe penalty (5e-4) overshoots (keyframe rate 0.20, more failures). Next: target-rate control
  (constrained RL) or residual RL on top of the rules. Related work: docs/related_work.md. `scripts/bc.py`, then `train.py --set init_policy=runs/bc_rules_s0 critic_warmup_iters=5`.
- **Residual RL over the rules (3 seeds):** learns to always follow the rules (identical results); overrides don't pay
  with the authors' reward. Reward signal = the bottleneck. Next: residual + long_window / gamma 0.9.
- **Constrained PPO (PI-Lagrangian keyframe-rate target 0.30, 20-frame window, 3 seeds): best all-rounder.** TUM
  0.518 +- 0.019 (rules 0.510, within noise; PPO scratch 0.539), TartanAir 2.3/6 finished, 1.00 failures/traj (rules 1, 1.50).
- **BC + constrained (1 seed): failed.** The rate target held in training, but on TUM the policy keyframes ~0.20 ->
  poor robustness. A fixed keyframe-rate target doesn't transfer across datasets. Next idea: constrain the OUTCOME
  (failure rate) instead, or a motion-dependent target.
- **More data (7 scenes, 1 seed):** no clear accuracy gain; PPO from scratch learns to keyframe EVERY frame and becomes
  the most robust on hard held-out TartanAir (7/13 vs rules 2/13) at ~14% more SVO time; constrained v2 matches the
  rules on TUM (0.509) but is less robust on hard motion. Next: failure-rate (outcome) constraint.
- **Shadow-SVO relative reward (new, 1 seed, 7 scenes):** reward = agent minus a shadow SVO running the rules on the same
  images (exact mirror verified). Learns to beat the rules on training data and has the fewest failures on TartanAir,
  but by keyframing much more -> worse on slow TUM seqs. Next: shadow reward + keyframe budget taken from the shadow.
- EuRoC not downloaded: the ETH host rate-limits our IP (see research_plan.md for options).

## Layout
```
paper/        paper PDF (local only, gitignored) + notes.md (summary, hyperparameters, paper-vs-code discrepancies)
reference/    official code (uzh-rpg/rl_vo @ c273182). READ-ONLY.
third_party/svo-lib/  our patched SVO; every change marked `[rlvo]`
src/rlvo/     env.py (RLVOEnv = reference env + switches), data.py, train.py, evaluate.py
configs/      base.yaml (authors' setup scaled to 12 envs + our fixes) + variants/<name>.yaml (one change each)
scripts/      build, data prep, train, evaluate, compare, plot, svo_param_search, reward_diagnosis
tests/        test_env.py: env == reference with all switches off; sanity checks
docs/         research_plan.md, log.md, modifications.md
results/      current tables/figures/evals (tracked). archive/ = superseded results (tracked, see archive/README.md)
runs/         training runs (gitignored); runs/archive/ old runs, runs/logs/ run logs
data/         datasets (gitignored): TartanAir (7 scenes after the 2026-10-01 download), TUM-RGBD (9 seqs), calibration/, logs/, _zips/
notebooks/, writeup/   empty for now
```

## Commands
- Train: `.venv/bin/python scripts/train.py [--variant <name>] [--set key=value ...]` -> `runs/<name>_s<seed>/`
- Resume / extend: `scripts/train.py --resume runs/X [--set total_timesteps=N]` (LR schedule restarts from the new total)
- Many runs: `VARIANTS="baseline long_window" SEEDS="0 1 2" EVAL_DATASETS="tartanair tum" scripts/run_all_variants.sh <tag> <steps> <repeats> obs_rms_warmup_steps=200`
- Evaluate: `scripts/evaluate.py --runs runs/X [--heuristic] --dataset tum|tartanair|tum_default|euroc --repeats 3 --out-tag <tag> [--stochastic]`
- Compare: `scripts/compare.py --tag <tag> --dataset <ds>` -> `results/tables/<tag>_<ds>.md` (merges seeds)
- Plots: `scripts/plot_curves.py --tag <tag>`. Tests: `.venv/bin/python -m pytest -q tests/`
- SVO: `scripts/build_svo_mac.sh`, `scripts/smoke_test_svo.py --envs 12`. Data: `download_tartanair.sh` + `tartan_to_gray.py`, `download_tum.sh`, `download_euroc.sh`

## Conventions (what "standard" means now)
- **Headline metrics:** `finished` (sequences with zero failures, the paper's criterion) and `ate_common` (ATE on the
  sequences EVERY method/seed finishes). The authors' first-segment ATE and our `ate_all` can both be gamed by failing
  (early failure -> short easy segment; many failures -> many easy short segments). Always report failures/coverage.
- **TartanAir evaluation set is FIXED** (`data.val_include` in base.yaml = the 6 held-out Easy trajectories of our first
  3 scenes), even as more scenes are downloaded. Other held-out trajectories (DPVO test split in new scenes / Hard) are
  never trained on; `evaluate.py --tartan-val all` evaluates on all of them (results tag `<tag>_valall`).
- **TUM:** `--dataset tum` = `tum_tuned.yaml` (quality_min_fts 25), for SVO rules AND agents. `tum_default` = authors' file.
- **base.yaml** = authors' setup scaled to 12 envs, PLUS: `env.fix_reset_gt_indexing: true` (our fix of a reference
  bug), `svo_threads: 12`, 6 held-out TartanAir trajectories (18 train). `EnvConfig()` defaults = pure reference
  (used by tests). Use `obs_rms_warmup_steps=200` for runs shorter than ~1M steps.
- **Seeds:** single-seed results were misleading (long_window). Use >= 3 seeds before claiming anything.
- Training is deterministic for a fixed seed + config (reward1 baseline reproduced ppo15 exactly).
- Speed: ~500-600 training steps/s on the M3 Pro (400k steps = 13.5 min incl. evals; 1M = ~30 min).

## Gotchas (macOS build and environment)
- Python 3.12 venv at `.venv/`. **Re-run `scripts/fix_torch_libomp.sh` after every torch (re)install** (two OpenMP
  runtimes abort with "OMP: Error #15"). Never use `KMP_DUPLICATE_LIB_OK`.
- Homebrew: `cmake eigen@3 opencv@4 boost yaml-cpp libomp suite-sparse glew`. Use eigen@3 and opencv@4 (pinned in the
  build script). Never add `/opt/homebrew/include` globally (Homebrew glog/gflags would shadow the bundled ones).
- SVO pose buffer is **column-major**: `T = poses[i].reshape(4,4).T`. Stages: 0 paused, 1 init, 2 tracking, 3 reloc.
- #envs must be <= #training trajectories (TartanLoader starts env i on trajectory i).
- The reference predates NumPy 2: `rlvo/__init__.py` sets `np.Inf`. Harmless warning: `objc: Class ... implemented in both`.
- Reference PPO starts obs normalization only at iteration 10 -> `obs_rms_warmup_steps`.
- The ETH EuRoC host rate-limits (HTTP 429); don't hammer it.

## The reference implementation (key facts)
- Only **SVO** is released (no DSO / ORB-SLAM3). PPO = modified SB3 copy (`rl_algorithms/`) with a masked rollout
  buffer (learn only from tracking steps) and a privileged critic (GT during training). Policy: Perceiver-style encoder.
- Train on TartanAir; evaluate on EuRoC, TUM-RGBD (+ KITTI). ATE after Umeyama; a sequence counts as finished only
  without any failure ("-" in the tables). They also report an "intersection" metric (TODO: add to compare.py).
- License: GPL-3.0. Cite the paper and SVO.

## Working rules
- Never modify `reference/`. When paper and code disagree, note it in `paper/notes.md` and ask which to follow.
- Every experiment gets a config file and a run folder; the config, seed and git commit are logged with results.
- Keep large files (datasets, checkpoints, videos) out of git. New findings go to `docs/log.md`; keep this file current.
- **The user prefers small tests first. Ask before any run longer than ~15 min**, before installing system packages,
  and before deleting data. Commit/push only when the user says so.
