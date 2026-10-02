# RL Visual Odometry

Replication and extension of **Reinforcement Learning Meets Visual Odometry**
(Messikommer, Cioffi, Gehrig, Scaramuzza — ECCV 2024, [paper](https://rpg.ifi.uzh.ch/docs/ECCV24_Messikommer.pdf),
[official code](https://github.com/uzh-rpg/rl_vo)). An RL agent (PPO) makes the heuristic decisions inside the
SVO visual-odometry pipeline, such as keyframe selection and feature-grid size, instead of hand-tuned rules.

The project is a course project, with the goal of a paper. It runs **natively on an Apple Silicon Mac**
(M3 Pro, 12 CPU cores). The authors used Docker, Ubuntu and a GPU server.

## Status

| Stage | State |
|---|---|
| SVO built natively on macOS arm64 (patched, see below) | ✅ |
| Authors' training environment running on real TartanAir data | ✅ ~1,270 env-steps/s rollout, ~600 steps/s training |
| Training / evaluation / comparison / plotting pipeline | ✅ |
| 11 proposed modifications, each a config switch | ✅ implemented, tested at small scale |
| Small-scale pilot comparisons (20k steps) | ✅ pipeline validated (archived: superseded by later runs) |
| TUM-RGBD: SVO rules reproduce the paper's SVO row | ✅ 7/8 sequences finished, 8% mean deviation (tuned SVO settings) |
| EuRoC evaluation | ⏳ download blocked by the host's rate limit |
| PPO, authors' setup (3 seeds, 400k; 1 seed 1.5M) | ✅ matches SVO's tuned rules (checkpoint-averaged); no consistent gain |
| Reward diagnosis + reward experiment | ✅ reward is weak/local/delayed; reward changes push the keyframe rate to extremes |
| Fixes: behavior cloning -> PPO, residual RL, constrained PPO, shadow-SVO relative reward, more data | ✅ none beats the rules beyond noise (3 seeds where available) |
| Bias audit + checkpoint-averaged re-scoring | ✅ final-checkpoint scores are +-7% random draws; 3 earlier claims overturned |
| Write-up | ⏳ outline in `writeup/outline.md` |

## What we changed relative to the authors

All modifications are **off by default**. With every switch off, our environment reproduces the authors' reward and
critic observations exactly (`tests/test_env.py`, tolerance 1e-12). Details and motivation:
[`docs/modifications.md`](docs/modifications.md).

| Variant (`configs/variants/`) | Idea |
|---|---|
| `failure_penalty` | explicit penalty when SVO loses tracking (the reference gives none) |
| `gamma_0.9`, `gamma_0.99` | longer credit assignment (the authors use gamma=0.6, a horizon of ~2.5 frames) |
| `normalized_error` | scale-invariant position error (error / distance travelled) instead of a fixed 0.2 m threshold |
| `rotation_reward` | relative rotation error in the reward (the authors use position only) |
| `keyframe_paper` | keyframe penalty from the paper (5e-3) instead of the released code (1e-4) |
| `critic_horizon_5` | privileged critic sees 5 future GT motions instead of 1 |
| `extra_obs` | SVO quality signals as observations (tracking quality, #features, #keyframes, ...) |
| `frame_stack_3` | memory: the policy sees the last 3 observations |
| `threshold_action` | new action: FAST corner-detector threshold |
| `augment` | photometric domain randomization of training images |

### Bugs fixed along the way
- **ARM-only crash in SVO** (`warpAffine`): converting `inf` to `int` gives `INT_MIN` on x86 (caught by the bounds check)
  but `INT_MAX` on ARM (overflows past it). The result was an out-of-bounds read and a SIGBUS during relocalization.
- Two OpenMP runtimes in one process (PyTorch + Homebrew OpenCV), which aborted at import.
- pybind11 rejected a strided pose array that the reference code passes during GT initialization.
- **Reference bug: GT re-initialization after a tracking failure used another environment's ground truth.** It created
  cascades of spurious failures (TUM, SVO rules: 46 -> 4 failures per run with the fix). Fixed, on by default.
- The reference PPO only switches on observation normalization after 10 PPO iterations, so short runs never normalize.
  Fixed with an optional warm-up (`obs_rms_warmup_steps`).

## Validation on real data (TUM-RGBD)
SVO with its own rules (no RL), GT initialization, 3 repeats, vs the paper's Table 2 SVO row. One setting changed from
the authors' TUM file (`quality_min_fts` 40 -> 25, found by `scripts/svo_param_search.py`; the paper's baseline was
grid-searched too). The result is saved as `svo_env/param/tum_tuned.yaml` and is our standard for TUM.

| seq | ours [m] | paper SVO [m] | | seq | ours [m] | paper SVO [m] |
|---|---|---|---|---|---|---|
| desk | 0.647 | 0.681 | | room | 0.817 | 0.805 |
| desk2 | 0.880 | 0.898 | | rpy | 0.055 | 0.053 |
| plant | 0.259 | 0.320 | | teddy | 0.698 | 0.769 |
| xyz | 0.066 | 0.057 | | 360 | fails once/run | 0.186 |
| floor | fails | fails | | | | |

## Training results so far
Checkpoint-averaged (mean over the last 3 policy snapshots per seed, then over seeds). TUM-RGBD, same tuned SVO
settings for everyone; ATE on the 5 sequences every method finishes; "per seq" = geometric mean of the per-sequence ATE
ratio to the rules (1.00 = equal). Full table: [`results/tables/main_results.md`](results/tables/main_results.md).

| Method (3 Easy scenes, 400k steps) | Seeds | TUM ATE vs rules | Per seq | Keyframe rate | TartanAir failures/traj |
|---|---|---|---|---|---|
| SVO rules (tuned) | - | 0.523 m | 1.00 | 0.30 | 1.50 |
| PPO, authors' setup | 3 | +0.1% | 1.08 ± 0.06 | 0.34 ± 0.14 | 1.35 ± 0.33 |
| Behavior cloning -> PPO | 3 | -1.4% ± 5% | 1.11 ± 0.05 | 0.40 | 1.87 |
| Residual RL over the rules | 3 | 0.0% | 1.00 | 0.30 | 1.52 |
| Constrained PPO (keyframe rate 0.30) | 3 | +0.2% ± 3% | 0.99 ± 0.05 | 0.27 | 1.20 ± 0.45 |

**Bottom line so far: at ~1.6% of the paper's training steps, RL keyframe selection converges to (but does not beat)
SVO's hand-tuned rules.** The paper reports -14.5% ATE vs its SVO at 25M steps and 8x our batch size.
## Findings so far (small scale)
- **Evaluation noise can manufacture gains.** One run's TUM ATE moves +-7% between consecutive policy snapshots with
  no trend; seeds differ as much. Final-checkpoint, single-seed scores produced three claims we later overturned
  (PPO "5-8% worse", BC -> PPO "4% better", PPO "half the keyframes").
- **The authors' ATE can be gamed by failing early** (it only covers the segment before the first tracking failure:
  0.17 m at 6% coverage in a pilot). We also report finished sequences, failures and coverage.
- **A bug in the reference re-initialization after tracking failures** inflated failure counts ~11x on TUM (fixed by default).
- **The reward is weak, local and delayed:** 99.7% of steps get a positive reward on TUM; it measures 5-frame error, not
  drift; a keyframe's benefit arrives 4-5 frames later, discounted away at gamma=0.6.
- **Reward changes push the keyframe rate to an extreme** (paper's 5e-3 penalty -> almost no keyframes; longer
  windows/horizons -> a keyframe every frame). Constraining the rate (Lagrangian) prevents the collapse.
- **Snapshot trends:** the authors' PPO is flat from 270k to 1.5M steps; residual RL stays exactly at the rules;
  only constrained PPO kept improving (in robustness) up to 400k.

Throughput on the M3 Pro: 400k training steps take ~13 minutes; the paper's 25M steps would take ~11.5 hours.
## Quick start (macOS, Apple Silicon)

```bash
# 1. Python environment (Python 3.12)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/fix_torch_libomp.sh           # one OpenMP runtime; re-run after every torch (re)install

# 2. Build SVO (Homebrew: cmake eigen@3 opencv@4 boost yaml-cpp libomp suite-sparse glew)
./scripts/build_svo_mac.sh
.venv/bin/python scripts/smoke_test_svo.py --envs 12      # synthetic images, no data needed

# 3. Data: a few TartanAir scenes (Easy), converted to grayscale
./scripts/download_tartanair.sh Easy japanesealley carwelding westerndesert
.venv/bin/python scripts/tartan_to_gray.py

# 4. Train / evaluate / compare
.venv/bin/python scripts/train.py --variant failure_penalty --set total_timesteps=200000 obs_rms_warmup_steps=200
.venv/bin/python scripts/train.py --resume runs/failure_penalty_s0 --set total_timesteps=400000   # continue / extend
.venv/bin/python scripts/evaluate.py --heuristic --runs runs/failure_penalty_s0 --repeats 3 --out-tag mytest [--stochastic]
./scripts/run_all_variants.sh <tag> <steps> <repeats> obs_rms_warmup_steps=200            # everything, then compare
.venv/bin/python scripts/compare.py --tag <tag> && .venv/bin/python scripts/plot_curves.py --tag <tag>

# Tests
.venv/bin/python -m pytest -q tests/
```

Throughput on the M3 Pro: 1M training steps take about 28 minutes. The paper's 25M steps would take about 11.5 hours.

## Repository layout

```
reference/rl_vo/      official code, read-only (uzh-rpg/rl_vo @ c273182)
third_party/svo-lib/  our patched SVO (every change marked [rlvo])
src/rlvo/             env (reference env + switches), data, train, evaluate
configs/              base.yaml (authors' setup scaled to 12 envs) + variants/
scripts/              build, data prep, train, evaluate, compare, plots; experiments/ = one runner per experiment
tests/                equivalence with the reference + sanity checks
docs/                 modifications.md (every change + bugs), research_plan.md (next steps), log.md (history)
paper/notes.md        paper summary, hyperparameters, paper-vs-code discrepancies
results/              current tables, figures, evaluations (tracked; index in results/README.md); runs/, data/ not tracked
writeup/              paper / course-report outline
archive/              superseded results, with a README explaining why
CLAUDE.md             working notes and conventions for this project
```

## Deviations from the authors (compute)
100 parallel envs -> 12. PPO batch 25,000 -> 3,000 (still full batch). A bug in the reference re-initialization after
tracking failures is fixed by default (`env.fix_reset_gt_indexing`). All of TartanAir -> 3 Easy scenes
(18 train / 6 held-out trajectories; `data.train_include`), later 7 scenes incl. Hard (89 train) for the `*_v2` runs.
400k steps per run (1.6% of 25M). SVO threads 8 -> 12.

## Next steps
1. EuRoC as a clean test set (the paper's main benchmark; the ETH host rate-limits us, browser download needed).
2. Decide the paper framing: replication + evaluation study (current evidence) vs. a method that beats the rules
   (candidate: shadow-SVO relative reward with a keyframe budget; needs 3+ seeds).
3. Optional scale test: the authors' batch size (25k) and several million steps, to see whether scale explains the
   paper's gain.

## Citation and license
Please cite the original paper and SVO (see `reference/rl_vo/README.md`). The reference code and our derived code
are **GPL-3.0**.
