"""RLVOEnv: the reference VecSVOEnv plus switchable modifications (all OFF by default = authors' setup).

Modifications (see docs/modifications.md for motivation):
  reward.failure_penalty   extra negative reward when SVO loses tracking (svo done)
  reward.error_mode        'absolute' (authors: 0.2 m threshold) or 'normalized' (error / GT path length in window)
  reward.rotation_weight   penalty on orientation error after the same Umeyama alignment
  reward.keyframe_reward   keyframe penalty lambda2 (code: 1e-4, paper: 5e-3)
  critic_horizon K         privileged critic sees K future GT relative motions (authors: K=1)
  extra_obs                5 extra SVO quality stats in the agent observation
  frame_stack k            agent sees the fixed observations of the last k steps (memory without recurrence)
  threshold_action         3rd action: FAST detector threshold in {5, 10, 15, 20, 25}
  augment                  photometric domain randomization on training images
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

import rlvo  # noqa: F401
from env.svo_wrapper import VecSVOEnv
from env.utils.running_mean_std import RunningMeanStd
from env.utils.trajectory_alignment import align_umeyama

from rlvo.data import PhotometricAugmenter, TartanLoaderK

N_KEYPOINT_OBS = 180 * 3
N_BASE_FIXED = 24
N_EXTRA_OBS = 5  # [tracking quality, #features, #keyframes, grid cell size, FAST threshold]; see svo C++ [rlvo]


@dataclass
class RewardConfig:
    align_reward: float = 0.01          # lambda1
    keyframe_reward: float = 0.0001     # lambda2 (released code value; paper says 5e-3)
    traj_length: int = 5                # sliding window
    nr_points_for_align: int = 3
    error_mode: str = "absolute"        # 'absolute' | 'normalized'
    error_threshold: float = 0.2        # metres (absolute) or fraction of window path length (normalized)
    min_path_length: float = 0.05       # metres, floor for normalized error
    rotation_weight: float = 0.0        # 0 = off
    rotation_cap_deg: float = 10.0
    failure_penalty: float = 0.0        # 0 = off


@dataclass
class EnvConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    critic_horizon: int = 1
    extra_obs: bool = False
    frame_stack: int = 1
    threshold_action: bool = False
    augment: bool = False


class RLVOEnv(VecSVOEnv):
    def __init__(self, params_yaml_path, calib_yaml_path, dataset_dir, num_envs, mode, cfg: EnvConfig,
                 initialize_glog=False, val_traj_ids=None, dataset='tartanair', seed=0, extra_val=()):
        self.cfg = cfg
        r = cfg.reward
        ref_reward = SimpleNamespace(align_reward=r.align_reward, keyframe_reward=r.keyframe_reward,
                                     traj_length=r.traj_length, nr_points_for_align=r.nr_points_for_align)
        ref_dataset = dataset
        if dataset == 'tartanair':
            # Our loader: K future poses (identical to reference for K=1) + extra held-out trajectories.
            # Built BEFORE the reference __init__, which is told an unknown dataset name so it keeps this loader
            # (the reference loader would assert on the authors' fixed val split).
            self.dataloader = TartanLoaderK(dataset_dir, mode, num_envs, val_traj_ids,
                                            n_future=cfg.critic_horizon, extra_val=extra_val)
            ref_dataset = '__rlvo_prebuilt__'
        super().__init__(params_yaml_path, calib_yaml_path, dataset_dir, num_envs, mode, ref_reward,
                         initialize_glog=initialize_glog, val_traj_ids=val_traj_ids, dataset=ref_dataset)

        # Observation layout seen by the policy: [fixed_t, fixed_{t-1}, ..., keypoints(540), critique]
        self.n_extra = N_EXTRA_OBS if cfg.extra_obs else 0
        self.agent_obs_dim_fixed = N_BASE_FIXED + self.n_extra          # used by the reference normalization
        self.agent_obs_dim = self.agent_obs_dim_fixed + N_KEYPOINT_OBS  # also the size handed to C++
        self.critique_dim = 1 + 6 * cfg.critic_horizon
        self.obs_rms = RunningMeanStd(shape=(1, self.agent_obs_dim_fixed))
        self.obs_rms_new = RunningMeanStd(shape=[1, self.agent_obs_dim_fixed])
        self.stack = cfg.frame_stack
        self.policy_obs_dim_fixed = self.agent_obs_dim_fixed * self.stack
        self.obs_dim = self.policy_obs_dim_fixed + N_KEYPOINT_OBS + self.critique_dim
        self.observation_space = spaces.Box(-np.inf * np.ones(self.obs_dim), np.inf * np.ones(self.obs_dim),
                                            dtype=np.float64)
        self._fixed_hist = np.zeros([num_envs, self.stack, self.agent_obs_dim_fixed])

        if cfg.threshold_action:
            self.action_space = spaces.MultiDiscrete([2, 5, 5])
            self.action_space_scale = np.asarray([[1, 0], [5, 20], [5, 5]])
            self.action_dim = 3

        # Rotation buffers for the relative rotation error (same window semantics as the reference positions buffer)
        W = cfg.reward.traj_length
        self.rot_est = np.tile(np.eye(3), (num_envs, W, 1, 1))
        self.rot_gt = np.tile(np.eye(3), (num_envs, W, 1, 1))

        self.augmenter = PhotometricAugmenter(num_envs, seed=seed) if (cfg.augment and mode == 'train') else None
        self.last_info_extra = {}
        self.last_raw_since_kf = np.ones(num_envs)

    # ---------------------------------------------------------------- images
    def get_images_pose(self):
        images, poses, new_seq = super().get_images_pose()
        if self.augmenter is not None:
            images = self.augmenter(images, new_seq)
        return images, poses, new_seq

    # ---------------------------------------------------------------- observations
    def _reorder(self, obs):
        """C++ layout [24 base, 540 keypoints, extra] -> [24 base, extra, 540 keypoints]."""
        if self.n_extra == 0:
            return obs
        base = N_BASE_FIXED
        return np.concatenate([obs[:, :base], obs[:, base + N_KEYPOINT_OBS:], obs[:, base:base + N_KEYPOINT_OBS]], 1)

    def svo_step(self, images, action, timestamps, use_RL_actions, use_gt_init_poses, gt_init_poses=None):
        if gt_init_poses is not None and gt_init_poses.ndim == 3:
            gt_init_poses = gt_init_poses[:, 0, :]
        poses, observations, dones = super().svo_step(images, action, timestamps, use_RL_actions,
                                                      use_gt_init_poses, gt_init_poses)
        self.last_raw_since_kf = observations[:, 1].copy()  # raw "frames since last keyframe" (0 = new keyframe)
        return poses, self._reorder(observations), dones

    def reset_dones(self, dones, images, action, poses, observations, info, use_gt_initialization, gt_init_poses=None):
        poses, observations, info = super().reset_dones(dones, images, action, poses, observations, info,
                                                        use_gt_initialization, gt_init_poses)
        m = dones.astype(bool)
        if self.n_extra and m.any():  # rows written by reset_dones come straight from C++
            observations[m] = self._reorder(observations[m])
        return poses, observations, info

    def extract_next_poses(self, gt_poses):
        K = self.cfg.critic_horizon
        if gt_poses.ndim == 3:
            return gt_poses[:, 0, :], gt_poses[:, 1:1 + K, :]
        return gt_poses, np.zeros([gt_poses.shape[0], K, 7])

    def add_critique_observations(self, observations, gt_poses, next_poses, position_error):
        critique_obs = np.zeros([self.num_envs, self.critique_dim])
        if self.mode == 'train':
            critique_obs[:, 0] = position_error
            for k in range(self.cfg.critic_horizon):  # k=0 reproduces the reference exactly
                nxt = next_poses[:, k, :]
                translation = nxt[:, :3] - gt_poses[:, :3]
                critique_obs[:, 1 + 6 * k:4 + 6 * k] = translation
                rot_mask = np.abs(translation).sum(1) != 0
                if rot_mask.any():
                    diff = Rotation.from_quat(gt_poses[rot_mask, 3:]) * Rotation.from_quat(nxt[rot_mask, 3:]).inv()
                    critique_obs[rot_mask, 4 + 6 * k:7 + 6 * k] = diff.as_rotvec()
        return np.concatenate([observations, critique_obs], axis=1)

    def _stack(self, obs, starts):
        """Frame stacking of the (normalized) fixed block. History is cleared at sequence starts / failures."""
        if self.stack == 1:
            return obs
        f = self.agent_obs_dim_fixed
        self._fixed_hist[starts] = 0.0
        self._fixed_hist = np.roll(self._fixed_hist, 1, axis=1)
        self._fixed_hist[:, 0] = obs[:, :f]
        return np.concatenate([self._fixed_hist.reshape(self.num_envs, -1), obs[:, f:]], axis=1)

    def step(self, action, use_RL_actions_bool=True, use_gt_initialization=False):
        obs, reward, dones, info, valid = super().step(action, use_RL_actions_bool, use_gt_initialization)
        obs = self._stack(obs, np.asarray(dones, dtype=bool))
        self.last_observations = obs
        return obs, reward, dones, info, valid

    def reset(self, seed=None, options=None, use_gt_initialization=False):
        obs = super().reset(seed, options, use_gt_initialization)
        self._fixed_hist[:] = 0.0
        obs = self._stack(obs, np.ones(self.num_envs, dtype=bool))
        self.last_observations = obs
        return obs

    # ---------------------------------------------------------------- reward
    def update_alignment_buffer(self, pos_svo_mask, poses, gt_poses):
        # Mirror the reference position-buffer update for rotations (before it increments env_steps).
        full = np.logical_and(self.env_steps >= self.reward_traj_length, pos_svo_mask)
        if full.any():
            self.rot_est[full, :-1] = self.rot_est[full, 1:]
            self.rot_gt[full, :-1] = self.rot_gt[full, 1:]
        if pos_svo_mask.any():
            idx = np.minimum(self.env_steps[pos_svo_mask], self.reward_traj_length - 1).astype(int)
            rows = np.nonzero(pos_svo_mask)[0]
            self.rot_est[rows, idx] = poses[pos_svo_mask].reshape(-1, 4, 4).swapaxes(-1, -2)[:, :3, :3]
            self.rot_gt[rows, idx] = Rotation.from_quat(gt_poses[pos_svo_mask, 3:]).as_matrix()
        super().update_alignment_buffer(pos_svo_mask, poses, gt_poses)

    def compute_reward(self, dones, poses, gt_poses, svo_dones, valid_stages, info, action):
        r = self.cfg.reward
        n = self.num_envs
        pos_mask = np.logical_and(~dones.astype(bool), self.env_steps > self.reward_nr_points_for_align)
        position_reward = np.zeros(n)
        rotation_reward = np.zeros(n)
        position_error = np.zeros(n)
        if pos_mask.any():
            k = self.reward_nr_points_for_align
            s, R, t = align_umeyama(self.gt_positions[pos_mask, :k, :], self.positions[pos_mask, :k, :])
            est_pos = poses[pos_mask, -4:-1]
            aligned = s[:, None] * np.matmul(R, est_pos[:, :, None]).squeeze(2) + t
            err = np.sqrt(((aligned - gt_poses[pos_mask, :3]) ** 2).sum(1))
            position_error[pos_mask] = err

            if r.error_mode == 'absolute':
                score = r.error_threshold - err
            elif r.error_mode == 'normalized':
                buf = self.gt_positions[pos_mask]                       # [m, W, 3], valid rows = env_steps
                n_valid = np.minimum(self.env_steps[pos_mask], self.reward_traj_length).astype(int)
                seg = np.linalg.norm(np.diff(buf, axis=1), axis=2)      # [m, W-1]
                seg_mask = np.arange(seg.shape[1])[None, :] < (n_valid - 1)[:, None]
                path = np.maximum((seg * seg_mask).sum(1), r.min_path_length)
                score = r.error_threshold - err / path
            else:
                raise ValueError(r.error_mode)
            position_reward[pos_mask] = np.maximum(score, -1) * r.align_reward

            if r.rotation_weight > 0:
                # Relative rotation error over the window (oldest buffered pose -> now). Alignment-free: the
                # Umeyama rotation from 3 near-collinear points is ill-conditioned about the motion direction.
                rows = np.nonzero(pos_mask)[0]
                cur = np.minimum(self.env_steps[pos_mask], self.reward_traj_length).astype(int) - 1
                d_est = np.matmul(self.rot_est[rows, 0].swapaxes(-1, -2), self.rot_est[rows, cur])
                d_gt = np.matmul(self.rot_gt[rows, 0].swapaxes(-1, -2), self.rot_gt[rows, cur])
                ang = np.degrees(Rotation.from_matrix(np.matmul(d_gt.swapaxes(-1, -2), d_est)).magnitude())
                self.last_info_extra['rotation_error_deg'] = ang
                rotation_reward[pos_mask] = -r.rotation_weight * np.minimum(ang, r.rotation_cap_deg) / r.rotation_cap_deg

        keyframe_reward = -action[:, 0] * r.keyframe_reward * valid_stages
        failure_reward = -r.failure_penalty * svo_dones.astype(bool)

        reward = position_reward + rotation_reward + keyframe_reward + failure_reward
        for i in range(n):
            info[i]['position_reward'] = position_reward[i]
            info[i]['keyframe_reward'] = keyframe_reward[i]
            info[i]['rotation_reward'] = rotation_reward[i]
            info[i]['failure_reward'] = failure_reward[i]
            info[i]['svo_failure'] = bool(svo_dones[i])
        return reward, info, position_error
