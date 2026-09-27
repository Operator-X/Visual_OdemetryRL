# Paper notes — Reinforcement Learning Meets Visual Odometry (ECCV 2024)

- Authors: Nico Messikommer*, Giovanni Cioffi*, Mathias Gehrig, Davide Scaramuzza (UZH RPG)
- Code: https://github.com/uzh-rpg/rl_vo (copy in `../reference/rl_vo`)  ·  Video: https://youtu.be/pt6yPTdQd6M

## Core idea
VO is framed as an MDP. The environment is the VO system (SVO / DSO / ORB-SLAM3) together with a fixed image sequence.
At every frame, an RL agent picks VO hyperparameters online instead of using hand-tuned heuristic rules.

## Agent
- **Observations:** tracked keypoints (image position + estimated depth, variable count N×3) + local map statistics (relative poses to the newest and oldest keyframe).
- **Network:** Perceiver-style "Variable Encoder". M learned query tokens (dim D) cross-attend to the projected keypoints (multi-head attention),
  then a 2-layer MLP (ReLU) with the map stats. About 296k parameters in total.
- **Actions** (independent categoricals):
  - keyframe: yes/no
  - grid size (SVO only): {20, 25, 30, 35, 40}

## Reward
`r = λ1 · max(-1, 0.2 − e_tran) − λ2 · a_keyframe`
- e_tran: position error at t_i over a sliding window of 5 poses. The first 3 poses are Umeyama-aligned (sim3) to ground truth.
- Paper: λ1 = 0.01, λ2 = 5e-3

## Training
- PPO (Stable-Baselines3), **privileged critic** (sees current + future GT poses, used only during training).
- **Masked rollout buffer:** only "valid" states (previous state was in tracking mode) are used for policy updates. All states are used for returns.
- VO initialization uses GT poses for triangulation (faster, and removes init variance).
- Data: TartanAir, 337 sequences, 279,987 frames. Uses grayscale `image_left`.
- Runtime: 100 parallel SVO envs; one rollout step takes about 104 ms.

## Evaluation
- EuRoC (11 seqs), TUM-RGBD (9 seqs), KITTI (supplementary). ATE [m] after Umeyama alignment.
- Baselines: SVO/DSO with grid-searched parameters. SOTA comparisons (TartanVO, DROID-VO, DPVO) are taken from the DPVO paper.
- Headline numbers to reproduce (EuRoC avg ATE): RL SVO 0.696, RL DSO 0.483 (vs DSO 0.594).
  TUM avg: RL SVO 0.422 vs SVO 0.471.
- Ablations (Tab. 5): w/o keypoints, w/o privileged critic, w/o keyframe action, w/o grid-size action.

## Paper vs released code (verify these)
| Item | Paper | `reference/rl_vo/config/config.yaml` |
|---|---|---|
| Keyframe penalty λ2 | 5e-3 | `keyframe_reward: 0.0001` |
| Align reward λ1 | 0.01 | `align_reward: 0.01` ✓ |
| Window / align points | 5 / 3 | `traj_length: 5`, `nr_points_for_align: 3` ✓ |
| γ (discount) | not stated | 0.6 |
| Total timesteps | not stated | 25e6 |
| Other PPO params | — | n_steps 250, batch 25000, n_epochs 10, gae_λ 0.95, ent_coef 0.0025 |
| DSO / ORB-SLAM3 | reported | **not released** (SVO only) |

## Weaknesses in the authors' setup -> our variants
See `docs/modifications.md` for details. In short:
- Losing tracking is not penalized; gamma=0.6 means a ~2.5-frame horizon -> `failure_penalty`, `gamma_0.9/0.99`
- 0.2 m absolute error threshold across scenes of very different scale -> `normalized_error`
- Rotation ignored in the reward -> `rotation_reward` (relative rotation error; Umeyama rotation from 3 points is ill-posed)
- Keyframe penalty differs 50x between paper and code -> `keyframe_paper`
- Critic sees only 1 future GT step -> `critic_horizon_5`
- Memoryless policy -> `frame_stack_3` (LSTM = future work)
- Only keyframe + grid size are learned -> `threshold_action` (FAST threshold); SVO quality signals unused -> `extra_obs`
- Synthetic-only training -> `augment` (photometric domain randomization)
- Biggest limit is SVO itself (EuRoC 0.97 m vs DPVO 0.105 m) -> other backends = future work
