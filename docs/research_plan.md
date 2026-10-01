# Research plan (written 2026-09-27, to continue from here)

## Where we are
- The foundation is **validated against the paper on real data**. SVO's own rules on TUM-RGBD reproduce the paper's
  Table 2 SVO row: 7/8 sequences finished, 8% mean deviation (`tum_tuned.yaml`, the standard for `--dataset tum`).
- Pipeline ready: 11 modifications as config switches, training with resume, evaluation (argmax + stochastic,
  coverage-aware `ate_all`), comparison tables, learning-curve plots. Details in `docs/modifications.md`.
- Bugs found in the reference and fixed: GT re-initialization mis-indexing (on by default), obs normalization starting
  late (warm-up option), GT init on frames without GT, plus the ARM-only SVO crash.
- **First real training done (2026-09-28/29):** PPO baseline at 400k and 1.5M steps, a reward diagnosis, and a reward
  experiment (5 setups; baseline and long_window with 3 seeds). Result: PPO is as robust as the tuned SVO rules with
  ~half the keyframes, but ~5-8% less accurate; no reward variant beats the baseline robustly, because every change
  pushes the keyframe rate to an extreme. Details: `docs/log.md`, `results/tables/reward1_*`, `results/reward_diagnosis/`.
- **Current next idea:** control the keyframe budget directly (constrained RL / Lagrangian penalty to a target rate
  ~0.25-0.3), or a keyframe-penalty sweep with long_window. Step 1 below is done (kept for the record).
- EuRoC: download blocked by the ETH host's rate limit (`scripts/download_euroc.sh` ready; or download
  machine_hall.zip, vicon_room1.zip, vicon_room2.zip in a browser from
  https://www.research-collection.ethz.ch/handle/20.500.11850/690084 into `data/_zips/euroc/`).

## Step 1 (DONE 2026-09-29, see log): first real training experiment
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

## Catalog of viable AI methods (added 2026-09-28)
The problem: at every frame, choose SVO settings (keyframe yes/no, grid size, optionally the detector threshold) from
what SVO observes, to maximize accuracy and robustness.

**Properties of our setup that decide viability:**
- Few discrete actions: 2 x 5 = 10 combinations (50 with the threshold).
- Slow simulator: ~1,200 SVO steps/s on the Mac, so sample efficiency matters.
- **SVO's internal state can't be saved/restored.** No trying both actions from the same moment: this rules out
  search/planning (MCTS) and counterfactual "what if" labels.
- Ground truth is available during training (privileged critic).
- Synthetic training (TartanAir), real evaluation (TUM, EuRoC).
- The agent only acts while SVO tracks (masked rollout buffer).

Legend: ✅ good fit, 🟡 possible with caveats, ❌ poor fit here.

### 1. RL, on-policy
| Method | Fit | Notes |
|---|---|---|
| PPO (authors) | ✅ | baseline; masked buffer + privileged critic built |
| A2C | ✅ | simpler PPO; small effort |
| TRPO | 🟡 | rarely better than PPO; low value |
| Recurrent PPO (LSTM/GRU) | ✅ | real memory; medium-large effort (masking + memory) |
| Phasic Policy Gradient | 🟡 | more stable value learning; medium effort |

### 2. RL, off-policy (reuses experience, key with a slow simulator)
| Method | Fit | Notes |
|---|---|---|
| DQN + Double/Dueling + n-step | ✅ | small discrete action set; likely much more sample-efficient than PPO |
| Branching DQN (one head per action type) | ✅ | keyframe x grid x threshold without enumerating combinations |
| Rainbow / distributional (QR-DQN, IQN) | ✅ | QR-DQN in sb3-contrib; robust to noisy rewards |
| Discrete SAC | 🟡 | often unstable with discrete actions |
| IMPALA / V-trace | ❌ | built for many machines; no gain on one Mac |

### 3. Bandits (no planning)
| Method | Fit | Notes |
|---|---|---|
| Contextual bandit (neural, LinUCB, Thompson sampling) | ✅ | gamma=0.6 is almost a bandit already; tests whether planning matters; small effort |

### 4. Learning from demonstrations
| Method | Fit | Notes |
|---|---|---|
| Behavior cloning of SVO's rules, then RL fine-tuning | ✅ | starts as good as SVO instead of random; small-medium effort |
| DAgger | ❌ | needs an expert labeling arbitrary states; the only expert is SVO's rules |
| GAIL / inverse RL | ❌ | no better-than-rules demonstrations |

### 5. Offline RL (log data once, train many policies without SVO)
| Method | Fit | Notes |
|---|---|---|
| CQL / IQL on logged SVO runs (rules + random actions) | ✅ | dataset once (hours), then each method trains in minutes; scales well for a comparative study |
| Decision Transformer | 🟡 | same data; weak when logged behavior is weak |

### 6. Gradient-free / black-box optimization
| Method | Fit | Notes |
|---|---|---|
| Bayesian optimization / CMA-ES of SVO thresholds | ✅ | strongest non-RL baseline (a well-tuned fixed rule); our grid search was a mini version |
| Evolution strategies on policy weights | 🟡 | parallel, noise-robust; practical only for small (linear/MLP) policies |
| Genetic programming of keyframe rules | 🟡 | readable evolved rules; harder to tune |

### 7. Supervised learning (predict, then decide with a rule)
| Method | Fit | Notes |
|---|---|---|
| Failure predictor + threshold rule | ✅ | classifier "tracking fails within N frames" from SVO signals -> keyframe when risk is high; simple, interpretable |
| Error regressor (predict near-future drift) | 🟡 | same idea, noisier target |
| Hindsight labels from trying both actions | ❌ | impossible: SVO state can't be restored |

### 8. Problem reformulations (combine with any algorithm)
| Method | Fit | Notes |
|---|---|---|
| Constrained RL (Lagrangian PPO: max accuracy s.t. keyframe rate / runtime <= budget) | ✅ | cleaner than hand-tuning the keyframe penalty, which collapsed in the pilots |
| Multi-objective RL (policy conditioned on an accuracy/speed weight) | ✅ | one run gives the whole accuracy-vs-runtime curve; strong for a paper |
| Hierarchical RL (choose a setting regime every N frames) | 🟡 | fewer decisions; may lose fast reactions |
| Safe RL / shielding (fall back to SVO's rules when unsure) | ✅ | "never worse than the rules"; practical for deployment |

### 9. Model-based RL / planning
| Method | Fit | Notes |
|---|---|---|
| Dreamer-style world models | ❌ | SVO's internal state (map, depth filters) too complex to model |
| MCTS / MuZero | ❌ | needs simulator save/restore |
| Learned failure model + short-horizon planning | 🟡 | light version of 7 |

### 10. Policy architecture
| Method | Fit | Notes |
|---|---|---|
| Plain MLP on summary features | ✅ | is the authors' attention encoder needed? |
| Perceiver / attention over keypoints (authors) | ✅ | baseline |
| Transformer / graph network over keypoints | 🟡 | richer; small gains likely at our budget |
| Recurrent (LSTM/GRU) | ✅ | see 1 |
| CNN on raw images | 🟡 | more information, larger sim-to-real gap, slower |

### 11. Generalization and robustness
| Method | Fit | Notes |
|---|---|---|
| Domain randomization | ✅ | built (`augment`) |
| Curriculum (easy -> hard trajectories) | ✅ | cheap; may help early learning |
| Meta-RL / fast adaptation to new scenes | 🟡 | research-grade, expensive |
| Test-time adaptation | 🟡 | adapts online to real data; fragile |

### 12. Interpretability
| Method | Fit | Notes |
|---|---|---|
| Distill the trained policy into a decision tree (VIPER) | ✅ | "what did RL learn?" as readable rules, possibly a better hand-written SVO heuristic |
| Symbolic regression of a keyframe rule | 🟡 | similar goal, harder |

### 13. Not viable here
- LLM/VLM agents: far too slow for per-frame decisions at 20-30 FPS; not suited to numeric control.
- Replacing SVO with learned VO (DPVO, DROID-SLAM): a different problem (the "other backend" direction).

### Strongest mix for a comparative study (value for effort)
1. PPO (baseline)
2. DQN-family (sample efficiency)
3. Contextual bandit (does planning matter?)
4. BC from SVO's rules, then RL (warm start)
5. Offline RL (IQL/CQL) on logged data (cheap to compare many methods)
6. Bayesian optimization of static rules ("do we need learning?")
7. Supervised failure predictor (simplest learned alternative)
8. Constrained or multi-objective PPO (accuracy vs runtime)
9. Decision-tree distillation (interpretability)

This covers every major family, each answering one clear question. It supersedes the shorter "Suggested study" list
above. The user still has to choose what to include (see open decisions).

## Already-publishable side findings
- The authors' ATE (first segment before any failure) can be gamed by failing early. We propose coverage-aware reporting.
- Reference bug: GT re-initialization after failures mis-indexed (46 -> 4 failures/run on TUM when fixed).
- The paper's keyframe penalty (5e-3) collapses the policy to no keyframes (2/2 pilots); the released 1e-4 doesn't.
- Obs normalization in the reference PPO only starts at iteration 10 (biases short runs).
- Native Apple Silicon build, including an ARM-only out-of-bounds read in SVO (x86 hides it).
- SVO tuning on TUM: only `quality_min_fts` matters; keyframe thresholds barely do (FORWARD criterion).

## Open decisions (ask the user)
1. Go-ahead for step 1 (~1.5 h unattended) and the 4 setups.
2. Which methods to include in the comparative study (see "Catalog of viable AI methods"; suggested: the 9-item mix).
3. Course deadline and target venue (workshop / RA-L / ICRA-IROS). This sets how deep and long the runs should be.
4. EuRoC: retry the script, or download via a browser.

## Newer RL techniques that fit our problems (web search, 2026-09-30)
Our problems: the keyframe rate slides to extremes; the accuracy reward is weak and delayed; the simulator is slow;
BC -> PPO lost robustness on TartanAir.

| Priority | Technique | Addresses | Effort | Source |
|---|---|---|---|---|
| 1 | **Residual RL over SVO's rules:** per frame choose "follow rule" / "force keyframe" / "force no keyframe" (SVO already has a per-env rules-vs-RL switch) | robustness loss, collapse, exploration | small-medium, Python only | [overview](https://www.emergentmind.com/topics/residual-reinforcement-learning-residual-rl), [residual off-policy RL for BC (2025)](https://residual-offpolicy-rl.github.io/) |
| 2 | **Constrained PPO** (PPO-Lagrangian / RCPO; PID or predictive Lagrangian against oscillation): keyframe rate ~ target, penalty becomes a learned price | the hand-tuned penalty, collapse | medium | [predictive Lagrangian](https://arxiv.org/abs/2501.15217), [PCPO](https://arxiv.org/abs/2508.01883) |
| 3 | **PQN** (DQN without replay/target net, LayerNorm, lambda-returns, parallel envs), then **IBRL** (BC policy proposes actions for exploration + value targets) | slow simulator, better use of BC | medium each | [PQN](https://arxiv.org/abs/2407.04811), [IBRL](https://arxiv.org/abs/2311.02198) |
| 4 | **Q-chunking / action chunking** (decide the keyframe pattern for the next k frames; offline-to-online) | delayed reward | medium-large | [Q-chunking, NeurIPS 2025](https://arxiv.org/abs/2507.07969) |
| 5 | **Return decomposition** (RUDDER and successors: move delayed reward back to the causing decision) | delayed reward (analysis) | large | [RUDDER](https://arxiv.org/abs/1806.07857) |
