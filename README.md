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
| First PPO training (400k / 1.5M steps) | ✅ as robust as tuned SVO rules with ~half the keyframes, ~5-8% less accurate |
| Reward diagnosis + reward experiment (3 seeds) | ✅ no reward variant beats the baseline yet; keyframe rate collapses to extremes |
| Keyframe-budget control (constrained RL) | ⏳ next |

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
TUM-RGBD, same tuned SVO settings for everyone, ATE on the 5 sequences every method finishes (3 repeats):

| Method | ATE [m] | Keyframe rate |
|---|---|---|
| SVO rules | **0.510** | 0.29 |
| PPO baseline (400k steps, 3 seeds) | 0.539 ± 0.015 | 0.19-0.40 |
| PPO + 20-frame reward window (3 seeds) | 0.572 ± 0.044 | 0.23-1.00 |

- **Reward diagnosis:** the authors' reward is informative but weak (99.7% of steps get a positive reward on TUM),
  measures local error rather than drift, and a keyframe's benefit arrives 4-5 frames later, heavily discounted at
  gamma=0.6.
- **Every reward change pushes the keyframe rate to an extreme** (the paper's 5e-3 penalty -> almost no keyframes;
  longer horizons or windows -> a keyframe on every frame). The single keyframe-penalty weight decides which, which
  motivates controlling the keyframe budget directly.
- Single-seed results were misleading here: a 1-seed win of the 20-frame window did not hold over 3 seeds.

## Early findings (small scale, not conclusions)
Pilot: 20k training steps per variant (<0.1% of the paper's budget), 1 seed, 6 held-out TartanAir trajectories.
These pilots are archived (superseded): [`archive/`](archive/README.md).

- **The authors' ATE can be gamed by failing early.** It only covers the segment before the first tracking failure.
  A policy that loses tracking immediately gets a tiny, flattering ATE (0.17 m at 6% coverage). We therefore also report
  `ate_all` (every tracked segment), coverage, tracked fraction and failure count.
- **Keyframe selection dominates robustness.** Policies that stop keyframing lose tracking constantly. Keyframing on
  every frame is as robust as SVO's rules, but slower.
- **The paper's keyframe penalty (5e-3) drives the policy to almost no keyframes** in both pilots (34 failures per
  trajectory vs 9.7 for SVO's rules). This supports the smaller value in the released code.
- The robustness-oriented rewards (`failure_penalty`, `normalized_error`, `gamma_0.99`) learn *more* keyframes, and
  `failure_penalty` had the fewest failures (6.0 vs 9.7 per trajectory). This is one seed, so it's a hint only.

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
scripts/              build, data prep, train, evaluate, compare, plots
tests/                equivalence with the reference + sanity checks
docs/                 modifications.md (every change + bugs), research_plan.md (next steps), log.md (history)
paper/notes.md        paper summary, hyperparameters, paper-vs-code discrepancies
results/              current tables and figures (tracked); runs/ and data/ are not tracked
archive/              superseded results, with a README explaining why
CLAUDE.md             working notes and conventions for this project
```

## Deviations from the authors (compute)
100 parallel envs -> 12. PPO batch 25,000 -> 3,000 (still full batch). A bug in the reference re-initialization after
tracking failures is fixed by default (`env.fix_reset_gt_indexing`). All of TartanAir -> 3 Easy scenes
(18 train / 6 held-out trajectories). Short pilot budgets so far. SVO threads 8 -> 12.

## Next steps
1. Download EuRoC (the paper's main real-world benchmark; the ETH host currently rate-limits us) and evaluate there.
2. Choose a coverage-aware headline metric for the paper.
3. Multi-seed runs (200k–1M steps) of the baseline and the most promising variants.
4. Larger extensions: recurrent policy, other VO backends.

## Citation and license
Please cite the original paper and SVO (see `reference/rl_vo/README.md`). The reference code and our derived code
are **GPL-3.0**.
