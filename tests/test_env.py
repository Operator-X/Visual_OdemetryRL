"""Checks for src/rlvo/env.py against the reference implementation, on real TartanAir data.

    .venv/bin/python -m pytest -q tests/test_env.py        (needs data/TartanAir + built svo_env)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rlvo  # noqa: E402
from env.svo_wrapper import VecSVOEnv  # noqa: E402
from rlvo.env import EnvConfig, RewardConfig, RLVOEnv  # noqa: E402

PARAMS = str(rlvo.SVO_PARAMS / "tartan_train.yaml")
CALIB = str(rlvo.DATA / "calibration" / "tartan_pinhole.yaml")
DATA = str(rlvo.DATA / "TartanAir")
N = 4

_glog_done = False


def make_env(cfg, n=N):
    global _glog_done
    env = RLVOEnv(PARAMS, CALIB, DATA, n, 'train', cfg, initialize_glog=not _glog_done)
    _glog_done = True
    return env


def run(env, steps, use_rl=True):
    env.reset(use_gt_initialization=True)
    out = []
    for _ in range(steps):
        a = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
        out.append(env.step(a, use_RL_actions_bool=use_rl, use_gt_initialization=True))
    return out


def test_default_matches_reference_reward_and_critic():
    """With all options off, reward + critic observations equal the reference code on identical state."""
    env = make_env(EnvConfig())
    captured = []
    orig = env.compute_reward

    def spy(dones, poses, gt_poses, svo_dones, valid_stages, info, action):
        ours = orig(dones, poses, gt_poses, svo_dones, valid_stages, info, action)
        ref = VecSVOEnv.compute_reward(env, dones, poses, gt_poses, svo_dones, valid_stages,
                                       [dict() for _ in info], action)
        captured.append((ours[0], ref[0], ours[2], ref[2]))
        return ours

    env.compute_reward = spy
    run(env, 80)
    assert len(captured) == 80
    for r_ours, r_ref, e_ours, e_ref in captured:
        np.testing.assert_allclose(r_ours, r_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(e_ours, e_ref, rtol=0, atol=1e-12)
    assert any(np.abs(c[0]).sum() > 0 for c in captured), "reward never non-zero - test not exercising anything"

    # critic observations
    gt = np.random.rand(N, 2, 7)
    gt[..., 3:] /= np.linalg.norm(gt[..., 3:], axis=-1, keepdims=True)
    obs = np.zeros([N, env.agent_obs_dim])
    pe = np.random.rand(N)
    g, nxt = env.extract_next_poses(gt)
    ours = env.add_critique_observations(obs.copy(), g, nxt, pe)
    g_ref, nxt_ref = VecSVOEnv.extract_next_poses(env, gt)
    ref = VecSVOEnv.add_critique_observations(env, obs.copy(), g_ref, nxt_ref, pe)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_observation_dims_all_options():
    cfg = EnvConfig(reward=RewardConfig(error_mode='normalized', rotation_weight=0.005, failure_penalty=0.02),
                    critic_horizon=5, extra_obs=True, frame_stack=3, threshold_action=True, augment=True)
    env = make_env(cfg)
    outs = run(env, 60)
    obs = outs[-1][0]
    assert obs.shape == (N, env.obs_dim) == (N, (24 + 5) * 3 + 540 + 1 + 6 * 5)
    assert np.isfinite(obs).all()
    # extra obs (un-normalized copy is not kept, but the block must be non-constant over time)
    fixed = np.stack([o[0][:, 24:29] for o in outs[20:]])
    assert fixed.std() > 0
    rewards = np.stack([o[1] for o in outs])
    assert np.isfinite(rewards).all()


def test_rotation_error_is_small_with_heuristics():
    """Sanity check of rotation conventions: SVO's own heuristics should give small orientation errors."""
    env = make_env(EnvConfig(reward=RewardConfig(rotation_weight=1e-9)))
    angs = []
    orig = env.compute_reward

    def spy(*a):
        env.last_info_extra.pop('rotation_error_deg', None)
        out = orig(*a)
        if 'rotation_error_deg' in env.last_info_extra:
            angs.append(env.last_info_extra['rotation_error_deg'])
        return out

    env.compute_reward = spy
    run(env, 150, use_rl=False)
    angs = np.concatenate(angs)
    assert len(angs) > 100
    print(f"rotation error deg: median {np.median(angs):.2f}, p90 {np.percentile(angs, 90):.2f}")
    assert np.median(angs) < 3.0
