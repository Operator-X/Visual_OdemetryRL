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
  residual                 residual RL over SVO's rules: action per frame = follow rule / force keyframe / force no
                           keyframe (grid size stays at the rules' value); keyframe penalty on ACTUAL keyframes
  shadow_reward            reward RELATIVE to SVO's rules: a second "shadow" SVO runs the rules on the same images;
                           reward = (agent position reward - shadow position reward) - lambda*(agent kf - shadow kf).
                           Cancels scene difficulty; the privileged critic also sees the shadow's error.
  kf_target                constrained RL: the keyframe penalty becomes a Lagrange multiplier (env.kf_lambda) that a
                           PI controller in train.py adjusts after every rollout so the keyframe rate tracks the target
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
    fix_reset_gt_indexing: bool = False  # reference bug: GT re-init after a failure reads env i's GT for the i-th reset env
    residual: bool = False               # residual RL over SVO's rules (see module docstring)
    residual_grid_idx: int = 2           # grid size used when overriding (2 -> 30 px = the rules' tartan/tum value)
    override_penalty: float = 0.0        # optional cost per override (prefers the rules when in doubt)
    kf_target: object = None             # None = fixed penalty; float = target keyframe rate (constrained RL)
    shadow_reward: bool = False          # reward relative to a shadow SVO running the rules (train mode only)


def _invalid_gt(gt):
    """Rows marked as 'no ground truth' (all -1, TUM floor) or with a non-unit quaternion."""
    q = np.linalg.norm(gt[:, 3:7], axis=1)
    return np.all(gt == -1, axis=1) | (np.abs(q - 1.0) > 1e-3)


class ShadowSVO:
    """A second SVO instance per env that always runs SVO's own rules on the same images (train mode).

    Mirrors the reference bookkeeping for its own resets, GT initialization and the sliding-window position error,
    so its per-step position reward is computed exactly like the agent's.
    """

    def __init__(self, params, calib, n, reward_cfg, delta_time):
        import svo_env
        self.env = svo_env.SVOEnv(params, calib, n, False)
        self.n, self.r, self.dt = n, reward_cfg, delta_time
        W = reward_cfg.traj_length
        self.positions = np.zeros([n, W, 3])
        self.gt_positions = np.zeros([n, W, 3])
        self.env_steps = np.zeros(n)
        self.timestamps = np.zeros(n)
        self.stage = np.ones(n, dtype=int)
        self.prev_valid = np.zeros(n, dtype=bool)
        self.reset(np.ones(n, dtype=bool))

    def reset(self, mask):
        if mask.any():
            self.env.reset(np.nonzero(mask)[0].astype(np.float64))
            for b in (self.positions, self.gt_positions):
                b[mask] = 0
            self.env_steps[mask] = 0
            self.timestamps[mask] = 0
            self.stage[mask] = 1
            self.prev_valid[mask] = False

    def step(self, images, gt_poses, new_seq):
        """One frame for all envs; returns (position reward, position error, keyframe made, tracking valid)."""
        n, r = self.n, self.r
        self.reset(np.asarray(new_seq, dtype=bool))
        use_gt = (self.stage == 1) & ~_invalid_gt(gt_poses)
        poses, obs = np.zeros([n, 16]), np.zeros([n, N_BASE_FIXED + N_KEYPOINT_OBS])
        dones, stages, runtime = np.zeros(n), np.zeros(n), np.zeros(n)
        self.env.step(images, self.timestamps.copy(), np.zeros([n, 2]), np.zeros(n), poses, obs, dones, stages, runtime,
                      use_gt.astype(np.float64), np.ascontiguousarray(gt_poses))
        self.stage = stages.astype(int)
        kf = obs[:, 1] == 0
        failed = dones.astype(bool)
        if failed.any():
            # like RLVOEnv.reset_dones (fixed indexing): restart and re-initialize on the SAME frame from GT
            self.reset(failed)
            k = int(failed.sum())
            gt_k = np.ascontiguousarray(gt_poses[failed])
            st_k = np.zeros(k)
            self.env.env_step(np.nonzero(failed)[0].astype(np.float64), np.ascontiguousarray(images[failed]),
                              self.timestamps[failed].copy(), np.zeros([k, 2]), np.zeros(k), np.zeros([k, 16]),
                              np.zeros([k, N_BASE_FIXED + N_KEYPOINT_OBS]), np.zeros(k), st_k, np.zeros(k),
                              (~_invalid_gt(gt_k)).astype(np.float64), gt_k)
            self.stage[failed] = st_k.astype(int)
        self.timestamps += self.dt
        valid = (self.stage == 2) & ~failed                 # = reference svo_valid_stage (used for the buffer)
        acted = valid & self.prev_valid                     # = reference valid_stages (tracking now and before)
        self.prev_valid = valid.copy()
        # sliding-window buffer, same semantics as the reference update_alignment_buffer
        W = r.traj_length
        full = (self.env_steps >= W) & valid
        if full.any():
            self.positions[full, :-1] = self.positions[full, 1:]
            self.gt_positions[full, :-1] = self.gt_positions[full, 1:]
        if valid.any():
            idx = np.minimum(self.env_steps[valid], W - 1).astype(int)
            rows = np.nonzero(valid)[0]
            self.positions[rows, idx] = poses[valid, -4:-1]
            self.gt_positions[rows, idx] = gt_poses[valid, :3]
            self.env_steps[valid] += 1
        pos_reward, err = np.zeros(n), np.zeros(n)
        m = ~failed & (self.env_steps > r.nr_points_for_align)   # reference: not done (even while relocalizing)
        if m.any():
            k = r.nr_points_for_align
            sc, R, t = align_umeyama(self.gt_positions[m, :k], self.positions[m, :k])
            aligned = sc[:, None] * np.matmul(R, poses[m, -4:-1, None]).squeeze(2) + t
            err[m] = np.sqrt(((aligned - gt_poses[m, :3]) ** 2).sum(1))
            pos_reward[m] = np.maximum(r.error_threshold - err[m], -1) * r.align_reward
        return pos_reward, err, kf & acted, acted


class RLVOEnv(VecSVOEnv):
    def __init__(self, params_yaml_path, calib_yaml_path, dataset_dir, num_envs, mode, cfg: EnvConfig,
                 initialize_glog=False, val_traj_ids=None, dataset='tartanair', seed=0, extra_val=(), val_include=None,
                 train_include=None):
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
                                            n_future=cfg.critic_horizon, extra_val=extra_val, val_include=val_include,
                                            train_include=train_include)
            ref_dataset = '__rlvo_prebuilt__'
        super().__init__(params_yaml_path, calib_yaml_path, dataset_dir, num_envs, mode, ref_reward,
                         initialize_glog=initialize_glog, val_traj_ids=val_traj_ids, dataset=ref_dataset)

        # Observation layout seen by the policy: [fixed_t, fixed_{t-1}, ..., keypoints(540), critique]
        self.n_extra = N_EXTRA_OBS if cfg.extra_obs else 0
        self.agent_obs_dim_fixed = N_BASE_FIXED + self.n_extra          # used by the reference normalization
        self.agent_obs_dim = self.agent_obs_dim_fixed + N_KEYPOINT_OBS  # also the size handed to C++
        self.critique_dim = 1 + 6 * cfg.critic_horizon + (1 if cfg.shadow_reward else 0)
        self.obs_rms = RunningMeanStd(shape=(1, self.agent_obs_dim_fixed))
        self.obs_rms_new = RunningMeanStd(shape=[1, self.agent_obs_dim_fixed])
        self.stack = cfg.frame_stack
        self.policy_obs_dim_fixed = self.agent_obs_dim_fixed * self.stack
        self.obs_dim = self.policy_obs_dim_fixed + N_KEYPOINT_OBS + self.critique_dim
        self.observation_space = spaces.Box(-np.inf * np.ones(self.obs_dim), np.inf * np.ones(self.obs_dim),
                                            dtype=np.float64)
        self._fixed_hist = np.zeros([num_envs, self.stack, self.agent_obs_dim_fixed])

        self._override = np.zeros(num_envs, dtype=bool)
        if cfg.residual:
            assert not cfg.threshold_action, "residual and threshold_action are not combined"
            # Policy action: 0 follow SVO rule, 1 force keyframe, 2 force no keyframe. Internally (action_dim 2) the
            # reference still receives [keyframe, grid_idx] and a per-env rules-vs-RL switch (see create_options).
            self.action_space = spaces.MultiDiscrete([3])
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
        self.kf_lambda = float(cfg.reward.keyframe_reward)   # Lagrange multiplier when cfg.kf_target is set
        self.shadow = None
        if cfg.shadow_reward and mode == 'train':
            assert cfg.reward.error_mode == 'absolute' and cfg.reward.rotation_weight == 0, "shadow: absolute reward only"
            self.shadow = ShadowSVO(params_yaml_path, calib_yaml_path, num_envs, cfg.reward, self.delta_time)
        self._sh_err = np.zeros(num_envs)
        self._stash = None
        self.kf_integral = 0.0

    def seed(self, seed=0):
        super().seed(seed)
        if self.shadow is not None:
            self.shadow.env.setSeed(seed)      # same RNG stream as the agent's SVO -> identical when decisions match

    # ---------------------------------------------------------------- images
    def get_images_pose(self):
        images, poses, new_seq = super().get_images_pose()
        if self.augmenter is not None:
            images = self.augmenter(images, new_seq)
        self._stash = (images, poses, np.asarray(new_seq, dtype=bool))
        return images, poses, new_seq

    def _shadow_step(self, gt_poses, force_new=False):
        images, _, new_seq = self._stash
        if force_new:
            new_seq = np.ones(self.num_envs, dtype=bool)
        return self.shadow.step(np.ascontiguousarray(images), gt_poses, new_seq)

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
        if gt_init_poses is not None:
            # [rlvo] never GT-initialize on a frame without GT (TUM floor marks those with -1; SVO aborts on the
            # resulting non-unit quaternion). The env then waits/initializes on its own, as without GT init.
            use_gt_init_poses = np.logical_and(use_gt_init_poses, ~_invalid_gt(gt_init_poses))
        poses, observations, dones = super().svo_step(images, action, timestamps, use_RL_actions,
                                                      use_gt_init_poses, gt_init_poses)
        self.last_raw_since_kf = observations[:, 1].copy()  # raw "frames since last keyframe" (0 = new keyframe)
        return poses, self._reorder(observations), dones

    def reset_dones(self, dones, images, action, poses, observations, info, use_gt_initialization, gt_init_poses=None):
        """Reimplementation of the reference reset_dones (svo_wrapper.py) with two fixes.

        Always: no GT initialization from a frame without GT (crash guard).
        fix_reset_gt_indexing: the reference passes the GT flags/poses for ALL envs while every other array holds only
        the reset envs; the C++ side indexes all of them with the packed index i, so the i-th reset env reads env i's
        GT. With the switch on, flags/poses are packed like the other arrays. Off = reference behaviour.
        """
        dones_mask = dones.astype("bool")
        nr_resets = int(dones.sum())
        reset_idx = np.nonzero(dones_mask)[0].astype(np.float64)
        self.env.reset(reset_idx)
        for buf in (self.timestamps, self.env_steps):
            buf[dones_mask] = 0
        for buf in (self.positions, self.gt_positions, self.positions_scale, self.gt_positions_scale, self.scale_buffer):
            buf[dones_mask] = 0
        reset_poses = np.zeros([nr_resets, 16], dtype=np.float64)
        reset_observations = np.zeros([nr_resets, self.agent_obs_dim], dtype=np.float64)
        reset_dones_array = np.zeros([nr_resets], dtype=np.float64)
        reset_stages = np.zeros([nr_resets], dtype=np.float64)
        reset_runtime = np.zeros([nr_resets], dtype=np.float64)
        reset_use_RL_actions = np.zeros([nr_resets], dtype=np.float64)

        if not use_gt_initialization or gt_init_poses is None:
            use_gt = np.zeros([self.num_envs], dtype=np.float64)
            gt = -np.ones([self.num_envs, 7], dtype=np.float64)
        elif self.cfg.fix_reset_gt_indexing:
            gt = np.ascontiguousarray(gt_init_poses[dones_mask])                      # packed like the rest
            use_gt = (~_invalid_gt(gt)).astype(np.float64)
        else:                                                                           # reference semantics
            gt = np.ascontiguousarray(gt_init_poses)
            use_gt = np.zeros([self.num_envs], dtype=np.float64)
            use_gt[dones_mask] = 1.0
            use_gt[_invalid_gt(gt)] = 0.0                                               # crash guard (row actually read)

        self.env.env_step(reset_idx, images[dones_mask, :, :, :], self.timestamps[dones_mask], action[dones_mask],
                          reset_use_RL_actions, reset_poses, reset_observations, reset_dones_array, reset_stages,
                          reset_runtime, use_gt, gt)
        poses[dones_mask] = reset_poses
        observations[dones_mask] = self._reorder(reset_observations)   # rows come straight from C++
        self.svo_stages[dones_mask] = reset_stages.astype('int')
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
        if self.cfg.shadow_reward:
            critique_obs[:, -1] = self._sh_err if self.mode == 'train' else 0.0
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

    def create_options(self, use_RL_actions_bool, use_gt_initialization):
        use_RL_actions, use_gt_init_poses = super().create_options(use_RL_actions_bool, use_gt_initialization)
        if self.cfg.residual:
            use_RL_actions = use_RL_actions * self._override   # "follow rule" -> SVO decides this frame itself
        return use_RL_actions, use_gt_init_poses

    def _residual_action(self, action):
        a = np.asarray(action).reshape(self.num_envs, -1)[:, 0].astype(int)
        self._override = a != 0
        full = np.zeros([self.num_envs, 2], dtype=np.int64)
        full[:, 0] = (a == 1)
        full[:, 1] = self.cfg.residual_grid_idx
        return full

    def step(self, action, use_RL_actions_bool=True, use_gt_initialization=False):
        if self.cfg.residual:
            action = self._residual_action(action) if use_RL_actions_bool else np.zeros([self.num_envs, 2], np.int64)
            if not use_RL_actions_bool:
                self._override[:] = False
        obs, reward, dones, info, valid = super().step(action, use_RL_actions_bool, use_gt_initialization)
        obs = self._stack(obs, np.asarray(dones, dtype=bool))
        self.last_observations = obs
        return obs, reward, dones, info, valid

    def reset(self, seed=None, options=None, use_gt_initialization=False):
        obs = super().reset(seed, options, use_gt_initialization)
        if self.shadow is not None:
            gt = self._stash[1][:, 0, :] if self._stash[1].ndim == 3 else self._stash[1]
            self._shadow_step(gt, force_new=True)
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

        lam = self.kf_lambda if self.cfg.kf_target is not None else r.keyframe_reward
        if self.shadow is not None:
            # symmetric with the shadow: charge ACTUAL keyframes on both sides
            keyframe_reward = -(self.last_raw_since_kf == 0).astype(float) * lam * valid_stages
        elif self.cfg.residual:
            # penalize ACTUAL keyframes (rule-made or forced), plus an optional cost per override
            actual_kf = (self.last_raw_since_kf == 0).astype(float)
            keyframe_reward = -actual_kf * lam * valid_stages \
                - self.cfg.override_penalty * self._override * valid_stages
        else:
            keyframe_reward = -action[:, 0] * lam * valid_stages
        failure_reward = -r.failure_penalty * svo_dones.astype(bool)

        if self.shadow is not None:
            sh_pos, sh_err, sh_kf, sh_valid = self._shadow_step(gt_poses)
            self._sh_err = sh_err
            # relative to the rules: position reward minus the shadow's; keyframes charged relative to the shadow's
            position_reward = position_reward - sh_pos
            keyframe_reward = keyframe_reward + lam * sh_kf.astype(float)
            for i in range(n):
                info[i]['shadow_position_reward'] = sh_pos[i]
                info[i]['shadow_keyframe'] = bool(sh_kf[i])
                info[i]['shadow_valid'] = bool(sh_valid[i])

        reward = position_reward + rotation_reward + keyframe_reward + failure_reward
        for i in range(n):
            info[i]['position_reward'] = position_reward[i]
            info[i]['keyframe_reward'] = keyframe_reward[i]
            info[i]['rotation_reward'] = rotation_reward[i]
            info[i]['failure_reward'] = failure_reward[i]
            info[i]['svo_failure'] = bool(svo_dones[i])
            info[i]['override'] = bool(self._override[i]) and bool(valid_stages[i])
            info[i]['keyframe_actual'] = bool(self.last_raw_since_kf[i] == 0) and bool(valid_stages[i])
            info[i]['valid'] = bool(valid_stages[i])
        return reward, info, position_error
