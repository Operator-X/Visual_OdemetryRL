# Research plan (written 2026-09-27, to continue from here)

## Where we are
- The foundation is **validated against the paper on real data**. SVO's own rules on TUM-RGBD reproduce the paper's
  Table 2 SVO row: 7/8 sequences finished, 8% mean deviation (`tum_tuned.yaml`, the standard for `--dataset tum`).
- Pipeline ready: 11 modifications as config switches, training with resume, evaluation (argmax + stochastic,
  coverage-aware `ate_all`), comparison tables, learning-curve plots. Details in `docs/modifications.md`.
- Bugs found in the reference and fixed: GT re-initialization mis-indexing (on by default), obs normalization starting
  late (warm-up option), GT init on frames without GT, plus the ARM-only SVO crash.
- **No real training yet.** Only 20k-step pilots (<0.1% of the paper's budget); their numbers are hints, not results.
- EuRoC: download blocked by the ETH host's rate limit (`scripts/download_euroc.sh` ready; or download
  machine_hall.zip, vicon_room1.zip, vicon_room2.zip in a browser from
  https://www.research-collection.ethz.ch/handle/20.500.11850/690084 into `data/_zips/euroc/`).

## Step 1 (next): first real training experiment, ~1.5 h unattended
Question: does the RL agent beat SVO's rules (the paper: 10-19% better on desk, desk2, plant, teddy), and does any of
our changes beat the authors' setup?

| Setup | Why |
|---|---|
| baseline (authors + our bug fix) | reference, "replicating the paper" |
| failure_penalty | fewest failures in the pilots |
| normalized_error | the other reward that learned more robust keyframing |
| gamma_0.99 | does a longer planning horizon help? |

- 200k steps per run (~6 min), **3 seeds** each, `obs_rms_warmup_steps=200`.
- Evaluate on TUM (standard tuned SVO settings, vs SVO rules with the same settings) and the 6 held-out TartanAir
  trajectories; argmax and stochastic; 3 repeats.
- Report ATE **with** finished sequences, failures and coverage (the ATE-only metric can be gamed by failing early).

| Outcome | Next |
|---|---|
| agent beats the rules on TUM | longer runs (1-5M) for the best setups, then the paper's tables |
| learns but doesn't beat them yet | extend the best runs with `--resume` |
| a variant beats the baseline across seeds | first contribution: ablations, EuRoC, write-up |
| nothing learns | debug the training setup before spending more compute |

Optional before long runs (not needed for step 1): image prefetch while SVO steps (~10-13% faster training).
Needs a seed loop in `run_all_variants.sh` and mean ± std across seeds in `compare.py`.

## Step 2: comparative study of RL techniques
The paper can be a comparative study ("What matters when using RL inside a VO pipeline?"), which is publishable even if
nothing beats the baseline, as long as it's rigorous: same conditions for every method, several seeds, learning curves
(sample efficiency matters at our budget), and an explanation of *why*.

### A. Non-RL baselines ("do we even need RL?")
| Method | Question | Effort |
|---|---|---|
| SVO rules, default vs tuned settings | what does simple tuning give? | done |
| Per-sequence optimization of SVO thresholds (Bayesian optimization) | can a well-tuned static rule match a learned adaptive one? | small |
| Imitation learning from SVO's rules, then RL fine-tuning | does starting from the rules speed up learning? | medium |

### B. RL algorithms (the paper uses only PPO)
| Method | Why | Effort |
|---|---|---|
| Contextual bandit (gamma = 0, no planning) | the authors' gamma=0.6 is almost that: does planning matter? | small (config) |
| Off-policy DQN-style (flatten keyframe x grid = 10 actions) | reuses experience; likely the biggest sample-efficiency gain with an expensive simulator | medium |
| A2C | how much does PPO's machinery help? | small |
| Recurrent PPO (sb3-contrib) | real memory, instead of frame stacking | medium-large |

The reference's masked rollout buffer (learn only from steps where the agent acted) and privileged critic (GT during
training) must be carried over to every algorithm. That is most of the implementation work in B.

### C. Design choices (mostly built already)
Reward (failure penalty, normalized error, rotation, keyframe penalty), critic horizon, extra observations, frame
stacking, augmentation: all implemented. To add: plain MLP vs the authors' attention encoder (small).

### Suggested study
1. Contextual bandit
2. Imitation warm-start
3. Off-policy DQN
4. MLP vs attention encoder
5. Static per-sequence tuning (non-RL baseline)
6. The reward and critic variants already built

Cost: ~20 min per method at 200k steps x 3 seeds, so 10-12 methods is ~4 h of compute (can be spread over nights).
Implementation of 1-5: ~1-2 days. Caveat to state in the paper: small budgets favor fast learners, so report learning
curves and frame the study as "at a limited compute budget".

## Already-publishable side findings
- The authors' ATE (first segment before any failure) can be gamed by failing early. We propose coverage-aware reporting.
- Reference bug: GT re-initialization after failures mis-indexed (46 -> 4 failures/run on TUM when fixed).
- The paper's keyframe penalty (5e-3) collapses the policy to no keyframes (2/2 pilots); the released 1e-4 doesn't.
- Obs normalization in the reference PPO only starts at iteration 10 (biases short runs).
- Native Apple Silicon build, including an ARM-only out-of-bounds read in SVO (x86 hides it).
- SVO tuning on TUM: only `quality_min_fts` matters; keyframe thresholds barely do (FORWARD criterion).

## Open decisions (ask the user)
1. Go-ahead for step 1 (~1.5 h unattended) and the 4 setups.
2. Which techniques from step 2 to include (suggested: 1-6 above).
3. Course deadline and target venue (workshop / RA-L / ICRA-IROS). This sets how deep and long the runs should be.
4. EuRoC: retry the script, or download via a browser.
