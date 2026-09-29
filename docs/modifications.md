# Modifications to the authors' setup

Every modification is a config switch. `configs/base.yaml` is the authors' released setup (scaled to one Mac).
`configs/variants/<name>.yaml` turns on one change at a time. With every switch off, `src/rlvo/env.py` gives
**exactly** the reference reward and critic observations. `tests/test_env.py` checks this on real data (tolerance 1e-12).

How to run: `scripts/train.py --variant <name>`, or all of them with `scripts/run_all_variants.sh <tag> <steps> [repeats]`.
Compare: `scripts/compare.py --tag <tag>` writes `results/tables/<tag>_<dataset>.md`.

## Baseline (authors) and forced deviations

| | Authors | Ours (base.yaml) | Why |
|---|---|---|---|
| Parallel envs | 100 | 12 | 12 CPU cores |
| Total steps | 25M (config) | pilot budget | compute |
| PPO batch | full batch, 250 x 100 = 25,000 | full batch, 250 x 12 = 3,000 | keeps n_steps (the GAE horizon) and full-batch updates |
| Training data | all of TartanAir (337 seqs) | 3 Easy scenes, 21 train trajectories | disk and download time |
| Keyframe penalty | 1e-4 in code, 5e-3 in paper | 1e-4 (code) | replicate the released code; the paper value is a variant |
| GT re-initialization after a failure | mis-indexed (bug, see below) | **fixed** (`env.fix_reset_gt_indexing: true`) | the bug creates cascades of spurious failures (TUM: 46 -> 4 per run); first-segment ATE unaffected |

## Reward

### `failure_penalty` — explicit penalty for losing tracking
- **Why:** when SVO loses tracking, the reference code only resets the episode. There is no negative reward, so the
  agent only loses future positive reward, and with gamma=0.6 it barely sees that. Robustness is a headline claim of the paper.
- **How:** `reward -= failure_penalty` on steps where SVO reports done (no last frame, i.e. the map was lost).
  0.02 is about 13 steps of maximum position reward.
- **Where:** `RLVOEnv.compute_reward`.

### `gamma_0.9`, `gamma_0.99` — longer credit assignment
- **Why:** gamma=0.6 gives an effective horizon of about 2.5 frames. Keyframe decisions shape the map for tens of
  frames. The paper never states gamma; it only appears in the config.
- **How:** `agent.gamma`.

### `normalized_error` — scale-invariant position error
- **Why:** the reference threshold is 0.2 **m**. TartanAir mixes indoor and large outdoor scenes, so the same metric
  error means very different things in different scenes.
- **How:** `score = tau - e_tran / L`, where L is the GT path length inside the 5-frame window (floored at 0.05 m).
  tau=0.25 is calibrated so the mean reward under SVO's heuristics (0.138) equals the absolute version.
  Measured under heuristics: e/L has median 0.053, p90 0.27.

### `rotation_reward` — orientation error term
- **Why:** the reward uses position only, but rotation drift causes failures like the fast-rotation EuRoC V103.
- **How:** relative rotation error over the window, comparing est vs GT from the oldest buffered pose to now:
  `-w * min(angle, cap)/cap`, with w=0.005 and cap=2 deg.
- **Pitfall avoided:** first version used the Umeyama rotation. It is fit on only 3 near-collinear positions, so it is
  unconstrained about the motion direction (median error 64 deg, up to 177 deg). The relative version is alignment-free:
  median 0.16 deg under heuristics (see test).

### `keyframe_paper` — keyframe penalty from the paper
- `keyframe_reward: 5e-3` (paper) vs 1e-4 (released code), 50x apart.

## Critic

### `critic_horizon_5` — more privileged information
- **Why:** the privileged critic sees only the current position error and the next GT motion. More future GT motion
  may reduce value variance further. The critic is only used during training.
- **How:** `TartanLoaderK` returns poses t..t+K. The critic gets 1 + 6K dims (translation + rotation vector per future
  step). K=1 reproduces the reference exactly.

## Observations

### `extra_obs` — SVO quality signals
- **Why:** the agent only sees keypoints (position, depth) and pose statistics, but SVO internally knows how well it is
  tracking.
- **How (C++, `[rlvo]` in `frame_handler_base.cpp`):** 5 extra values written after the default 564 only if the caller
  allocates them: tracking quality (0/1/2), #features in the frame, #keyframes in the map, current grid cell size,
  current FAST threshold. `RLVOEnv` moves them into the "fixed" block, so they get running-mean normalization.

### `frame_stack_3` — memory without recurrence
- **Why:** the policy is memoryless, but slow degradations (feature count dropping over frames) need history.
- **How:** the policy sees the normalized fixed block of the last 3 steps. History is cleared at sequence starts and failures.
- **Not done:** a recurrent (LSTM) policy. It would need recurrent PPO combined with the authors' masked rollout buffer,
  which is a larger change. Frame stacking is the cheap first test of whether memory helps at all.

## Actions

### `threshold_action` — FAST detector threshold as an action
- **Why:** only keyframe and grid size are learned. The corner-detector threshold is another hand-tuned heuristic
  (10 in `tartan_train.yaml`).
- **How (C++):** optional 3rd action column in {5, 10, 15, 20, 25}, applied to the detector each frame when RL actions
  are active, otherwise the configured value. Also turns on `extra_obs`, so the agent sees its current setting.

## Data

### `augment` — photometric domain randomization
- **Why:** training is synthetic only (TartanAir), evaluation is real (EuRoC, TUM).
- **How:** per sequence: gain U(0.7, 1.3), gamma U(0.7, 1.4), noise sigma U(0, 5). Per frame: ±3% brightness flicker.
  Kept mild because SVO is a direct method (photometric consistency).

## Infrastructure improvements (not experiment variables)
| Change | Effect |
|---|---|
| `svo_threads: 12` (env var `RLVO_SVO_THREADS`, explicit OpenMP clause in `svo_vec_env.cpp` and `utils.cpp`) | ~20% faster rollouts; torch no longer changes SVO's thread count and vice versa |
| Augmentation with OpenCV LUT + precomputed noise bank | 8.2 -> 2.7 ms per 12-image batch |
| Checkpoint + resume (`--resume`) | runs survive sleep, crashes and Colab disconnects, and can be extended |
| 3 extra held-out TartanAir trajectories | 6 val trajectories instead of 3 (harder: heuristic fails ~10x per trajectory) |
| `ate_all` coverage-aware metric + stochastic evaluation mode | the authors' ATE can be gamed by failing early; argmax exaggerates early policies |
| `scripts/plot_curves.py` | learning curves as small multiples vs baseline |

| `obs_rms_warmup_steps` (off by default) | the reference PPO activates observation normalization only at PPO iteration 10, so short runs never normalize and variants with large extra inputs start biased (extra_obs: 21% keyframes before any learning). A warm-up of 200 vec-steps of SVO heuristics fixes it (49%). Use it for all short tests. |

## Not implemented (documented future work)
- **A different VO backend** (DSO, ORB-SLAM3, DPVO). RL-SVO is far behind modern learned VO (EuRoC 0.97 m vs DPVO
  0.105 m). This is probably the most publishable direction, and a project of its own.
- **Recurrent policy** (see frame stacking).

## Bugs found while doing this (all fixed)
| Bug | Symptom | Fix |
|---|---|---|
| Two OpenMP runtimes (Homebrew OpenCV -> OpenBLAS -> libomp, plus torch's bundled libomp) | `OMP: Error #15` abort | `scripts/fix_torch_libomp.sh` |
| Strided `gt_init_pose` rejected by pybind11 | `TypeError` in `step()` during GT init | `pybind_wrapper.cpp` accepts any layout (copies) |
| **ARM-only out-of-bounds read in `warpAffine`** during relocalization: float->int conversion of inf saturates to INT_MAX on ARM (x86 gives INT_MIN, which the check catches), then `xi+1` overflows and passes the bounds check | SIGBUS mid-training | NaN/inf-safe float bounds check before the conversion (`patch_warp.cpp`) |
| NumPy 2 removed `np.Inf` | import error | shim in `rlvo/__init__.py` |
| TartanLoader needs #envs <= #train trajectories | bare `StopIteration` | checked with a clear message in our scripts |
| **Reference bug: GT re-initialization after a failure uses the wrong env's GT.** `reset_dones` passes images/actions for the reset envs only (packed) but GT flags/poses for ALL envs; C++ indexes all with the packed index, so the i-th reset env reads env i's GT | after a real failure SVO re-initializes without (or with wrong) GT, emitting a cascade of spurious "failures". TUM, SVO rules: 46.3 failures/run -> 4.0 with the fix. First-segment ATE unaffected | `env.fix_reset_gt_indexing: true` packs flags/poses correctly (off = reference behaviour) |
| GT init on frames without GT (TUM floor marks them -1) | SVO aborts: non-unit quaternion check | always on: such frames are never used for GT init |

## macOS / Apple Silicon build of SVO (moved here from CLAUDE.md on 2026-09-29)

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
