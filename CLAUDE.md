# RL Visual Odometry — project context

> **Continue here:** `docs/research_plan.md` (next steps, comparative-study plan, method catalog, open decisions).
> History of every decision and finding: `docs/log.md`. All changes vs the authors: `docs/modifications.md`.

## Goal
Replicate **"Reinforcement Learning Meets Visual Odometry"** (Messikommer, Cioffi, Gehrig, Scaramuzza — ECCV 2024),
then extend it. A **course project** aimed at a **publishable paper** (possibly a comparative study of RL techniques,
publishable even if nothing beats the baseline). Results must be reproducible and comparable to the paper's tables, and
every design choice needs a reason we can write up.

## Current state (2026-10-02) — full history in docs/log.md
- **Setup validated:** SVO's own rules on TUM-RGBD reproduce the paper's Table 2 SVO row (7/8 finished, 8% mean
  deviation) with `tum_tuned.yaml`. Reference: `results/eval/tum/svo_rules_std`.
- **Headline (checkpoint-averaged, `results/tables/main_results.md` part 2): no method beats SVO's tuned rules beyond
  noise.** Final-checkpoint scores are +-7% random draws; re-scoring overturned 3 earlier claims (PPO "5-8% worse",
  BC -> PPO "-4%", PPO "half the keyframes").
- **Authors' PPO (ppo_snap_s0-2, 3 seeds, 3 Easy scenes, 400k):** = rules (TUM ATE +0.1%, per-seq 1.08; keyframe rate
  seed-dependent 0.22-0.49; TartanAir failures within noise). Flat from 270k to 1.5M (ppo15).
- **Reward diagnosis:** weak (99.7% of steps positive on TUM), local (5-frame error, not drift), delayed (keyframe benefit
  4-5 frames later, discounted at gamma=0.6). Reward changes push the keyframe rate to an extreme (penalty 5e-3 -> none;
  long window / gamma 0.9-0.99 -> every frame).
- **Fixes tried (all ~= rules or worse, checkpoint-averaged):** BC -> PPO (3 seeds; -1.4% mean but 11% worse per seq, less
  robust on TartanAir); residual RL (3 seeds; always follows the rules); constrained PPO with a PI-Lagrangian keyframe
  target 0.30 (3 seeds; = rules, no collapse, the only method still improving at 400k, in robustness); BC + constrained
  (1 seed; worse); 7 scenes (1 seed each; PPO keyframes every frame); shadow-SVO relative reward (1 seed; beats the rules
  on training data but by keyframing ~85%, worse on TUM).
- **Candidate paper framing:** replication + evaluation study (RL converges to the tuned rules at small scale; evaluation
  practice can manufacture +-5-10% gains). Open: EuRoC (clean test set; ETH host rate-limits us, browser download),
  shadow reward + keyframe budget, scale test (authors' batch size).

## Layout
```
paper/        paper PDF (local only, gitignored) + notes.md (summary, hyperparameters, paper-vs-code discrepancies)
reference/    official code (uzh-rpg/rl_vo @ c273182). READ-ONLY.
third_party/svo-lib/  our patched SVO; every change marked `[rlvo]`
src/rlvo/     env.py (RLVOEnv = reference env + switches), data.py, train.py, evaluate.py
configs/      base.yaml (authors' setup scaled to 12 envs + our fixes) + variants/<name>.yaml (one change each)
scripts/      build, data prep, train, evaluate, compare, plot, svo_param_search, reward_diagnosis, bc, main_tables,
              eval_snapshots + snapshot_trends; experiments/ = one runner script per experiment (reproducible)
tests/        test_env.py: env == reference with all switches off; sanity checks
docs/         research_plan.md, log.md, modifications.md
results/      current tables/figures/evals (tracked; index + protocol in results/README.md). archive/ = superseded results
runs/         training runs (gitignored); runs/archive/ old runs, runs/logs/ run logs
data/         datasets (gitignored): TartanAir (7 scenes after the 2026-10-01 download), TUM-RGBD (9 seqs), calibration/, logs/, _zips/
writeup/      outline.md (paper/course-report outline, contributions, planned figures, open items)
notebooks/    empty for now
```

## Commands
- Train: `.venv/bin/python scripts/train.py [--variant <name>] [--set key=value ...]` -> `runs/<name>_s<seed>/`
- Resume / extend: `scripts/train.py --resume runs/X [--set total_timesteps=N]` (LR schedule restarts from the new total)
- Many runs: `VARIANTS="baseline long_window" SEEDS="0 1 2" EVAL_DATASETS="tartanair tum" scripts/run_all_variants.sh <tag> <steps> <repeats> obs_rms_warmup_steps=200`
- Evaluate: `scripts/evaluate.py --runs runs/X [--heuristic] --dataset tum|tartanair|tum_default|euroc --repeats 3 --out-tag <tag> [--stochastic]`
- Compare: `scripts/compare.py --tag <tag> --dataset <ds>` -> `results/tables/<tag>_<ds>.md` (merges seeds)
- Main results table for the write-up: `scripts/main_tables.py` -> `results/tables/main_results.md` (never hand-edit)
- Plots: `scripts/plot_curves.py --tag <tag>`. Tests: `.venv/bin/python -m pytest -q tests/`
- SVO: `scripts/build_svo_mac.sh`, `scripts/smoke_test_svo.py --envs 12`. Data: `download_tartanair.sh` + `tartan_to_gray.py`, `download_tum.sh`, `download_euroc.sh`

## Conventions (what "standard" means now)
- **Score a run by the mean over its last >=3 policy snapshots (and >=3 seeds), never the final checkpoint alone.**
  Report the geometric-mean per-sequence ratio next to the paper-style mean. TUM is a development set (we tuned on it).
- **Headline metrics:** `finished` (sequences with zero failures, the paper's criterion) and `ate_common` (ATE on the
  sequences EVERY method/seed finishes). The authors' first-segment ATE and our `ate_all` can both be gamed by failing
  (early failure -> short easy segment; many failures -> many easy short segments). Always report failures/coverage.
- **TartanAir training data:** `data.train_include` (null = all 89 non-held-out trajectories on disk, 7 scenes incl.
  Hard). To reproduce 3-scene experiments set `data.train_include=[japanesealley/Easy,carwelding/Easy,westerndesert/Easy]`.
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
