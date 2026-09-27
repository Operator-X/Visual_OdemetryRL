# RL Visual Odometry — project context

## Goal
Replicate **"Reinforcement Learning Meets Visual Odometry"** (Messikommer, Cioffi, Gehrig, Scaramuzza — ECCV 2024),
then extend it with new contributions. This is a **course project**, and the aim is to turn it into a **publishable paper**.
That means results must be reproducible and comparable to the paper's tables, and every design choice needs a reason we can write up.

## Layout
```
paper/        Paper PDF (local only, gitignored) + notes.md (summary, hyperparameters, known paper/code discrepancies)
reference/    Official code (uzh-rpg/rl_vo @ c273182). READ-ONLY. Do not edit; copy what we need into src/.
third_party/svo-lib/  Our patched copy of SVO (from reference/). Every change is marked with a `[rlvo]` comment.
src/rlvo/     Our package: env.py (RLVOEnv = reference env + switchable modifications), data.py, train.py, evaluate.py
docs/         modifications.md: every change to the authors' setup (why, how, where) + bugs found
configs/      base.yaml (authors' setup scaled to 12 envs) + variants/<name>.yaml (one modification each)
tests/        test_env.py: our env == reference when all switches are off; sanity checks
scripts/      Entry points: train / eval / data prep / plotting
notebooks/    Exploration + Colab notebooks
data/         Datasets (gitignored) — see data/README.md
runs/         Training logs, checkpoints, trajectories (gitignored)
results/      Final tables and figures that go into the report/paper (tracked)
writeup/      Course report + paper draft
```

## Compute plan
- **Mac is the main training machine** (M3 Pro, 12 CPU cores = 6 performance + 6 efficiency, 18 GB RAM, ~630 GB free disk).
  The bottleneck is the SVO environment stepping on the CPU, not the network (the agent has only ~300k params). Expect about 12 parallel envs, not 100.
  Use `caffeinate` for long runs.
- **Google Colab:** backup and extra runs. Free Colab has only ~2 vCPUs, so it is slow for SVO rollouts.
- Colab sessions are time-limited and disconnect: training must **checkpoint regularly and resume** from Drive.
  Data and checkpoints go on Google Drive, not the Colab VM disk.
- Paper scale is 100 parallel SVO envs, 25M timesteps on an A100. We will almost certainly need a scaled-down setup. Record every deviation.

## Environment
- Plain venv at `.venv/` using **Python 3.12** (matches Colab). Activate: `source .venv/bin/activate`
- `requirements.txt` = direct deps (unpinned). `requirements-lock-mac.txt` = exact pinned versions from the Mac (`pip freeze`).
- Jupyter kernel: "Python (rlvo)".
- On Colab: `pip install -r requirements.txt` (torch comes preinstalled with CUDA; don't reinstall it).
- Local torch uses the MPS backend (Apple GPU). The agent is tiny, so CPU is fine too.

## SVO build (macOS, native arm64) — WORKING
- Build: `./scripts/build_svo_mac.sh` → `third_party/svo-lib/build/svo_env/svo_env.cpython-312-darwin.so`
- Test: `.venv/bin/python scripts/smoke_test_svo.py --envs 12` (synthetic images, no dataset needed)
- Import in Python: add `third_party/svo-lib/build/svo_env` to `sys.path`, then `import svo_env`
- Homebrew deps: `cmake eigen@3 opencv@4 boost yaml-cpp libomp suite-sparse glew`.
  Use **eigen@3** (not Eigen 5) and **opencv@4** (4.14, keg-only). Both are pinned in the build script.
- Gotchas we fixed (all marked `[rlvo]` in the CMake files):
  - x86/GCC-only flags removed on arm64; NEON paths enabled (`HAVE_FAST_NEON`); C++17.
  - Eigen found through CMake, not `/usr/include/eigen3`.
  - pybind11 3.x needs `Development.Module`.
  - Link only the needed OpenCV modules.
  - `-DWITH_GFLAGS=OFF` so the bundled glog doesn't pick up Homebrew gflags (otherwise flags get registered twice at import).
  - Never add `/opt/homebrew/include` globally: Homebrew's glog/gflags headers would shadow the bundled ones.
- The pose output buffer (16 values per env) is **column-major**: `T = poses[i].reshape(4,4).T`
- SVO stages: 0 paused, 1 initializing, 2 tracking, 3 relocalization.
- **OpenMP:** Homebrew OpenCV -> OpenBLAS -> Homebrew libomp, and pip torch bundles its own libomp. Two runtimes
  abort with "OMP: Error #15". Fix: `scripts/fix_torch_libomp.sh` symlinks torch's libomp to Homebrew's.
  **Re-run it after every torch install/upgrade.** Never use `KMP_DUPLICATE_LIB_OK`.
- `pybind_wrapper.cpp` is patched so `gt_init_pose` accepts non-contiguous arrays (svo_wrapper.py passes a strided view).
- Harmless warning: `objc: Class CVWindow/AVF... implemented in both` (pip cv2 + Homebrew OpenCV GUI/video classes; unused).

## Running the reference env on our data
- `.venv/bin/python scripts/run_reference_env.py --envs N --steps K [--actions heuristic|random]`
- Data prep: `scripts/download_tartanair.sh Easy <scenes...>` then `scripts/tartan_to_gray.py`.
- **N envs must be <= the number of training trajectories.** TartanLoader starts env i on trajectory i.
- The reference code predates NumPy 2. Our scripts set `np.Inf = np.inf` before importing it (reference/ stays untouched).
- Measured on the M3 Pro, TartanAir japanesealley + carwelding Easy, 10 envs, GT init:
  **~1000 env-steps/s** (9.8 ms per vec-step) with SVO heuristics, ~840/s with random actions.
  This is about the paper's rollout throughput (100 envs x 104 ms = ~960/s). 25M steps is ~7-8 h of rollout
  (network forward and PPO updates not included yet).

## Training / evaluation workflow
- Train: `.venv/bin/python scripts/train.py [--variant <name>] [--set key=value ...]` -> `runs/<name>_s<seed>/`
  (config.yaml, meta.json with git commit, train.csv per rollout, eval.csv, Policy/*.pth + *_rms.npz)
- All variants: `scripts/run_all_variants.sh <tag> <steps> [repeats]`. Compare: `scripts/compare.py --tag <tag>`
  -> `results/tables/<tag>_<dataset>.md`
- Evaluate: `scripts/evaluate.py --runs runs/X --heuristic --repeats 3 [--dataset euroc] --out-tag <tag>`
- Tests: `.venv/bin/python -m pytest -q tests/`
- Speed on the M3 Pro: ~570 training steps/s including PPO updates (1M steps is ~30 min). Torch thread count makes no difference.
- The user wants **small-scale tests only for now** (minutes, not hours). Ask before launching anything long.
- **Metric pitfall:** the authors' ATE uses only the segment before the FIRST tracking failure. Policies that fail
  early get a tiny, flattering ATE (pilot: extra_obs ATE 0.17 m at 6% coverage, 31 failures/traj). Always report
  `ate_coverage`, tracked fraction and failures next to ATE. Consider a coverage-aware headline metric for the paper.
- Per-variant training cost is about the same as the baseline, except `augment` (~2x slower: per-pixel noise generation, can be optimized).
- SVO calls `omp_set_num_threads(8)` process-wide in its constructor. torch shares the same OpenMP runtime.

## The reference implementation (key facts)
- Only **SVO** is released (no DSO / ORB-SLAM3 code). SVO is C++ (`reference/rl_vo/svo-lib`) with pybind11 bindings (`svo_env`),
  built with CMake. Deps: Eigen, OpenCV, Boost, yaml-cpp, glog/gflags, SuiteSparse, GLEW. Their setup is Docker + Ubuntu 20.04 + CUDA 12.1.
- Building svo-lib natively on macOS may not work. Try it, but the fallback is to build on Colab (Ubuntu) or in Docker.
- RL: PPO (a modified copy of Stable-Baselines3 in `rl_algorithms/`) with a masked rollout buffer and a privileged critic.
  Policy is in `policies/attention_policy.py` (Perceiver-style encoder). Config uses Hydra (`config/config.yaml`).
- Train on TartanAir. Evaluate on EuRoC, TUM-RGBD (+ KITTI in the supplementary). Metric: ATE [m] after Umeyama alignment, averaged over 5 runs.
- License: GPL-3.0. Any code we copy from reference/ keeps that license. Cite the paper and SVO.

## Working rules
- Never modify `reference/`. It is the ground truth for "what the authors actually did".
- When the paper and code disagree, note it in `paper/notes.md` and ask which one to follow.
- Every experiment gets a config file and a run folder. Log the config, seed and git commit with results.
- Seed everything. Report mean over multiple runs (the paper uses 5).
- Keep large files (datasets, checkpoints, videos) out of git.
- Ask before installing system-level packages (brew) or starting long-running jobs.

## Status / decisions log
- 2026-09-27: Project scaffolded. Paper + reference code added. - 2026-09-27: Environment: plain venv, Python 3.12.
- 2026-09-27: svo-lib builds natively on the M3 Pro. Smoke test passes: 12 envs, initializes then tracks on synthetic images. The Mac is the main training machine.
  Calibration yamls are in `data/calibration/`.
- 2026-09-27: Real TartanAir data runs through the authors' VecSVOEnv (~1000 env-steps/s, 10 envs). Local subset: japanesealley,
  carwelding, westerndesert (Easy).
- 2026-09-27: Implemented all proposed modifications as config switches (docs/modifications.md). Fixed an ARM-only SVO
  crash (warpAffine OOB read). Pipeline test: every variant trained 20k steps + evaluated (results/tables/pilot_*).
  EuRoC not downloaded yet (ETH Research Collection rate-limited us; 3 large zips). The TartanAir held-out set
  (3 trajs) is too noisy for real comparisons.
