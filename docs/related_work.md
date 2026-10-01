# Related work and novelty check (first pass, 2026-09-30)

Method: web search + abstracts (not exhaustive). Next step for thoroughness: go through the papers citing the ECCV 2024
paper on Google Scholar / Semantic Scholar, and read the full texts of the entries below.

## Directly related: RL controlling decisions inside a VO / VIO pipeline

| Paper | What the RL agent decides | Training | Relevance to us |
|---|---|---|---|
| Messikommer, Cioffi, Gehrig, Scaramuzza. *Reinforcement Learning Meets Visual Odometry*. ECCV 2024 ([arXiv 2407.15626](https://arxiv.org/abs/2407.15626)) | keyframe + grid size in SVO / DSO (+ ORB-SLAM3) | PPO from scratch, privileged critic, TartanAir -> EuRoC/TUM | the paper we replicate |
| Nascivera, Bauersfeld, Delaune, Scaramuzza. *Online Adaptation of Visual Odometry Frontends with Image-Conditioned Reinforcement Learning* ([arXiv 2603.21785](https://arxiv.org/abs/2603.21785), Mar 2026) | FAST threshold, KLT patch size, RANSAC threshold; policy sees image embeddings + frontend statistics | RL from scratch, privileged critic, TartanAirV2 -> zero-shot EuRoC, TUM-VI, UZH-FPV; **no imitation / warm start** | same lab follow-up. **Overlaps our `threshold_action` and `extra_obs` variants: those are NOT novel.** Up to 8% lower ATE, 57% less runtime vs static parameters |
| Dai, Su, Kong, Ming, Kong. *Keyframe-Based Feed-Forward Visual Odometry* ([arXiv 2601.16020](https://arxiv.org/abs/2601.16020), Jan 2026) | keyframe selection for feed-forward (foundation-model) VO, e.g. VGGT-Long style | RL, trained on TartanAir; no mention of warm start | RL keyframe selection on a learned backend (the "other backend" direction) |
| Pan, Zheng, Yin, Dou. *Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry* ([arXiv 2511.21083](https://arxiv.org/abs/2511.21083), CVPR 2026) | when to run the visual frontend (from IMU only) + how much to trust VO in the fusion | RL (details not in the abstract) | cost-aware decisions in VIO; EuRoC, TUM-VI; up to 1.77x faster |

## Imitation -> RL (the general technique)

| Paper | Notes |
|---|---|
| Xing, Romero, Bauersfeld, Scaramuzza. *Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight* ([arXiv 2403.12203](https://arxiv.org/abs/2403.12203)) | same lab; imitation of a privileged RL teacher, then RL fine-tuning; learns where RL from scratch fails. Different domain (drone control, not VO decisions) |
| BC + RL fine-tuning in general ([2509.26605](https://arxiv.org/abs/2509.26605), [2509.19301](https://arxiv.org/html/2509.19301v1), [NeurIPS 2024](https://neurips.cc/virtual/2024/101647)) | well-established. Known issue: BC gives narrow, overconfident policies that limit RL improvement. Residual RL on top of BC is one fix |

## Novelty assessment for our work

- **Behavior cloning of the VO system's OWN hand-written rules, then RL fine-tuning, for in-pipeline VO decisions:**
  not found in this search. The technique itself (imitation -> RL) is standard, so the contribution would be the
  application plus the analysis: in RL-for-VO, rewards are weak and delayed, every reward change collapses the
  keyframe rate to an extreme, and warm-starting from the rules avoids the collapse and beats both PPO from scratch
  and the rules on the paper's metric (3 seeds, TUM). **Claim cautiously** until the full-text check is done.
- **Our analysis findings look new** (not reported in the abstracts found): the first-segment ATE can be gamed by
  failing early; the reference GT re-initialization bug; the reward diagnosis (weak, local, delayed signal); the
  keyframe-collapse mechanism; the SVO quality_min_fts sensitivity; the ARM-only SVO bug.
- **Not novel:** RL control of FAST threshold / frontend parameters (Nascivera et al. 2026), RL keyframe selection per
  se (the ECCV paper; Dai et al. 2026). Our `threshold_action` variant must cite Nascivera et al.
- **Worth borrowing:** image-conditioned observations (Nascivera et al.); residual RL on top of a BC/heuristic policy
  (could fix the "BC is overconfident" issue and the xyz/plant regressions).
