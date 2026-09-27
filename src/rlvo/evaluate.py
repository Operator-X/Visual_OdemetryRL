"""Evaluation of a policy (or SVO's own heuristics) on held-out sequences.

Per trajectory: ATE [m] of the first sub-trajectory before any tracking failure (the authors' metric, Umeyama sim3),
fraction of frames tracked, number of tracking failures, keyframe rate, and runtime.
"""
import time

import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

import rlvo  # noqa: F401
from env.utils.compute_error import ate_translation
from env.utils.trajectory_alignment import align_umeyama


def evaluate(env, policy=None, max_steps=None, device="cpu"):
    """Run every validation trajectory of `env` (mode='val') once from its start.

    policy=None -> SVO heuristics (use_RL_actions=False). Returns a list of per-trajectory dicts.
    """
    n = env.num_envs
    lengths = env.dataloader.nr_samples_per_traj.astype(int)
    T = int(lengths.max()) if max_steps is None else int(max_steps)
    names = [str(p).split("/")[-3] + "/" + str(p).split("/")[-1] if "TartanAir" in str(p) else str(p).split("/")[-1]
             for p in env.dataloader.trajectories_paths]

    completed = np.zeros(n, bool)
    pos = np.zeros([n, T, 3])
    gt = np.zeros([n, T, 3])
    valid = np.zeros([n, T], bool)
    dones = np.zeros([n, T], bool)
    fails = np.zeros([n, T], bool)
    kf = np.zeros([n, T], bool)
    steps = np.zeros(n, int)
    t_step = []

    obs = env.reset(use_gt_initialization=True)
    for t in range(T):
        if policy is not None:
            with th.no_grad():
                actions, _, _ = policy.forward(obs_as_tensor(obs, device), deterministic=True)
            actions = actions.cpu().numpy()
        else:
            actions = np.zeros([n, env.action_dim], dtype=np.int64)
        t0 = time.perf_counter()
        obs, _, d, infos, vmask = env.step(actions, use_RL_actions_bool=policy is not None, use_gt_initialization=True)
        t_step.append(time.perf_counter() - t0)
        for i, info in enumerate(infos):
            completed[i] |= bool(info['new_seq'])
            if completed[i]:
                continue
            steps[i] += 1
            valid[i, t] = vmask[i]
            dones[i, t] = d[i]
            fails[i, t] = info.get('svo_failure', False)
            kf[i, t] = vmask[i] and env.last_raw_since_kf[i] == 0
            if vmask[i]:
                pos[i, t] = info['position']
                gt[i, t] = info['gt_position']
        if completed.all():
            break

    results = []
    first_sub = np.cumsum(dones, axis=1) == 0
    for i in range(n):
        m = first_sub[i] & valid[i] & (np.abs(pos[i]).sum(1) != 0)
        ate = np.nan
        if m.sum() > 3:
            s, R, tr = align_umeyama(gt[None, i, m], pos[None, i, m])
            aligned = s[0] * np.matmul(R[0], pos[i, m, :, None]).squeeze(2) + tr[0]
            ate = float(np.asarray(ate_translation(gt[None, i, m], aligned[None])).ravel()[0])
        L = max(steps[i], 1)
        results.append(dict(
            trajectory=names[i], frames=int(lengths[i]), steps=int(steps[i]),
            ate=ate,
            first_sub_frac=float(m.sum() / L),        # fraction of the sequence covered by the ATE segment
            tracked_frac=float(valid[i].sum() / L),
            failures=int(fails[i].sum()),
            keyframe_rate=float(kf[i].sum() / max(valid[i].sum(), 1)),
            ms_per_step=1e3 * float(np.mean(t_step)) / n,
        ))
    return results
