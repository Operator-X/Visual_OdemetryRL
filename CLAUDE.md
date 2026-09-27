# RL Visual Odometry — project context

> **Continue here:** `docs/research_plan.md` has the next steps (first real training experiment, then a comparative
> study of RL techniques) and the open decisions for the user.

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
              research_plan.md: next steps, comparative-study plan, open decisions
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
- Evaluate: `scripts/evaluate.py --runs runs/X --heuristic --repeats 3 [--dataset euroc] [--stochastic] --out-tag <tag>`
  (`--stochastic` samples actions instead of argmax -> results in `<tag>_stoch`)
- Resume / extend: `scripts/train.py --resume runs/X [--set total_timesteps=N]`. `checkpoint.pt` is written every
  `checkpoint_every` PPO updates (atomic). Env state (SVO map) can't be saved, so all envs restart their sequences after a resume.
- Plots: `scripts/plot_curves.py --tag <tag>` -> `results/figures/<tag>_curves_<metric>.png` (small multiples vs baseline)
- Held-out TartanAir set: the authors' test split + `data.extra_val_trajs` (base.yaml) = 6 val / 18 train on our subset.
  Runs trained before 2026-09-27 evening used only the 3-trajectory split (their config lacks the key).
- Metrics: `ate` (authors', first segment only) + `first_sub_frac`; `ate_all` (every segment between failures aligned
  separately, >=10 poses, length-weighted RMSE) + `ate_all_cov`; tracked fraction; failures. Short segments align
  easily, so read `ate_all` with its coverage and the failure count too.
- Tests: `.venv/bin/python -m pytest -q tests/`
- Speed on the M3 Pro: ~570 training steps/s including PPO updates (1M steps is ~30 min) with 8 SVO threads. Torch thread count makes no difference.
  `svo_threads: 12` (base.yaml -> env var RLVO_SVO_THREADS) gives ~20% faster rollouts (1272 vs 1060 env-steps/s).
  Profile of a short run: SVO step ~45%, PPO update ~26%, image loading ~13%, policy forward ~12%, reward <2%.
- **Observation normalization starts late:** the reference PPO only activates obs normalization at PPO iteration 10
  (`update_rms` every 10 iterations). Until then the policy sees raw values (e.g. #features ~140). Runs shorter than
  10 iterations (30k steps at 12 envs) never normalize, and variants with more large inputs (extra_obs, frame_stack,
  threshold_action) start with biased keyframe decisions (20-30% instead of 50%).
  **Fix (option):** `obs_rms_warmup_steps: N` runs N vec-steps of SVO heuristics before training to estimate the stats
  and activates them immediately. Default 0 = authors. Use ~200 for short tests
  (`scripts/run_all_variants.sh <tag> <steps> <repeats> obs_rms_warmup_steps=200`). Verified on extra_obs:
  first-rollout keyframe rate 0.49 with warm-up vs 0.21 without.
- The user wants **small-scale tests only for now** (minutes, not hours). Ask before launching anything long.
- **Metric pitfall:** the authors' ATE uses only the segment before the FIRST tracking failure. Policies that fail
  early get a tiny, flattering ATE (pilot: extra_obs ATE 0.17 m at 6% coverage, 31 failures/traj). Always report
  `ate_coverage`, tracked fraction and failures next to ATE. Consider a coverage-aware headline metric for the paper.
- Augmentation cost: 2.7 ms per 12-image batch (OpenCV LUT + precomputed noise bank; was ~8 ms+ with per-pixel RNG).
- SVO's parallel loops now use an explicit `num_threads(...)` clause (no process-wide omp_set_num_threads), so torch
  and SVO no longer overwrite each other's thread counts.

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
- 2026-09-27: pilot2 = all variants, 20k steps, obs_rms_warmup_steps=200, 6 val trajs, 1 seed, 1 repeat, both argmax and
  stochastic eval (results/tables/pilot2_*, results/figures/pilot2_*). Still far too short for conclusions. Early signals:
  keyframe_paper (5e-3) collapses to ~no keyframes again (2/2 pilots) -> 34 failures/traj; failure_penalty,
  normalized_error and gamma_0.99 learn MORE keyframes (0.65-0.72), and failure_penalty has the fewest failures
  (6.0 vs 9.7/traj stochastic). Mean ATE is dominated by 2 westerndesert trajectories (~10 m); use per-trajectory/median.
- 2026-09-27: TUM-RGBD download (TUM server ~150 KB/s at first, faster later; `scripts/download_tum.sh`). EuRoC blocked:
  the ETH Research Collection rate-limits our IP (HTTP 429 even for a single browser-like request). `scripts/download_euroc.sh`
  is ready; retry later or download via a browser into data/_zips/euroc/.
- 2026-09-27: **Validation on real data:** our SVO heuristics on TUM (GT init, 3 repeats) match the paper's Table 2 SVO
  row within 5%: 360 0.196 vs 0.186, desk 0.647 vs 0.681, desk2 0.880 vs 0.898 (paper RL-SVO: 0.189 / 0.556 / 0.755).
  TUM uses `tartan_test.yaml` params (as the authors' config_eval.yaml). `scripts/evaluate.py --dataset tum [--data-dir]`.
- 2026-09-27: **Full TUM-RGBD check** (SVO heuristics, 9 seqs x 3 repeats, results/eval/tum/tum_full): on the 5 fully
  tracked seqs we are within -19..+16% of the paper's SVO row (desk -5%, desk2 -2%, teddy -9%, plant -19%, xyz +16% = 9 mm).
  floor fails in both. But our SVO LOSES TRACKING on 360 (1x), room (23x), rpy (15x), where the paper's SVO finishes.
  Likely the paper's per-dataset grid-searched SVO params (we use tartan_test.yaml defaults). Fine for RL-vs-heuristic
  (same SVO settings), but not for matching the paper's absolute robustness.
- Authors' evaluation (reference/rl_vo/evaluation/evaluate_runs.py): ATE = first sub-trajectory; a seq counts as finished
  only without any failure ("-" in the tables = not finished). They also report an "intersection" metric: all methods
  compared on the shortest common first segment (coverage-fair). TODO: add it to our compare.py.
- 2026-09-27: **FAST-9 (ARM) vs FAST-10 (x86) corner detector: tested, NO effect.** x86 builds use FAST-10 (SSE2), the ARM
  build FAST-9 (NEON). But SVO re-scores every candidate with fast_corner_score_10 and keeps a corner per grid cell only
  if its score > threshold. FAST-9-only candidates score <= threshold and can never be selected, so the final features
  are identical: TUM 9 seqs x 3 repeats gave byte-identical results, and speed was the same (within noise, <=3%).
  Runtime switch kept: RLVO_FAST_VARIANT=10 (verified with lldb that it calls fast_corner_detect_10). Default FAST-9.
  So the remaining gap to the paper is NOT the detector. Likely causes: grid-searched params, nondeterminism, library versions.
- 2026-09-27: **Reference bug found: GT re-initialization after a tracking failure is mis-indexed** (svo_wrapper.py
  reset_dones: GT flags/poses passed for all envs, other arrays packed; C++ uses the packed index for all). After a real
  failure SVO re-initializes without/with wrong GT and reports a cascade of spurious failures. TUM (SVO rules, 3 reps):
  46.3 failures/run -> 4.0 with `env.fix_reset_gt_indexing: true` (room 22.7 -> 0.7, rpy 14.7 -> 0.3); first-segment ATE
  and coverage identical. Pilot failure counts and the failure_penalty signal were inflated by this. Also: GT init is
  now never done on frames without GT (TUM floor; previously a fatal quaternion check). Recommendation: enable the fix
  for all our experiments. **User decision 2026-09-27: ON by default** (base.yaml `env.fix_reset_gt_indexing: true`;
  EnvConfig default stays False = reference, so tests/test_env.py still checks exact equivalence). Runs before this
  (pilot, pilot2) used the buggy reset.
- 2026-09-27: **SVO settings search on TUM** (scripts/svo_param_search.py, fix_reset_gt_indexing on, 16 configs over
  kfselect_min_disparity/min_angle/min_dist_metric and quality_min_fts). Only quality_min_fts matters (40 -> 25). The keyframe
  thresholds barely matter (FORWARD criterion is dominated by the #tracked-features bounds 90/180). Saved as
  `svo_env/param/tum_tuned.yaml`: 7/8 paper seqs finished in 3/3 runs, mean
  deviation 8% from the paper's SVO row; room 0.817 (+1%), rpy 0.055 (+3%). 360 still fails once per run. Tuned on the
  test set, like the paper's baseline. RL must be evaluated with the SAME SVO params.
- 2026-09-27: **User decision: tum_tuned is the standard for TUM.** `evaluate.py --dataset tum` now uses tum_tuned.yaml
  (for SVO rules AND RL agents); the authors' untuned file is `--dataset tum_default`. Earlier TUM results
  (results/eval/tum/tum_check*, tum_full) were produced with the untuned file and the buggy re-init.
