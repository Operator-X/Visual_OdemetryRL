"""Behavior cloning of SVO's own rules, as a warm start for PPO.

Demonstrations: SVO runs with its rules (use_RL_actions=False) on the TartanAir training trajectories. The rules'
keyframe decision is not exposed directly, but it can be read back: after each frame SVO reports "frames since last
keyframe", which is 0 when that frame became a keyframe. The rules always use the configured grid size (30 in
tartan_train.yaml = action index 2). A sample is (the observation the agent would have seen before the frame, the
rules' decision for that frame), kept only where the agent would have acted (valid = tracking now and before).
"""
import time

import numpy as np
import torch

import rlvo  # noqa: F401
from policies.attention_policy import CustomActorCriticPolicy

from rlvo.train import make_policy_kwargs, warmup_obs_rms

GRID_IDX_RULES = 2  # grid size 30 = 5*2+20 (tartan_train.yaml grid_size)


def collect_demos(env, vec_steps, warmup_steps=200):
    """Returns obs [N, obs_dim] (normalized exactly like during PPO) and actions [N, action_dim] of SVO's rules."""
    warmup_obs_rms(env, None, warmup_steps)          # normalization stats from SVO rules, then activated
    obs_prev = env.reset(use_gt_initialization=True)
    zeros = np.zeros([env.num_envs, env.action_dim], dtype=np.int64)
    X, Y = [], []
    for _ in range(int(vec_steps)):
        obs, _, dones, _, valid = env.step(zeros, use_RL_actions_bool=False, use_gt_initialization=True)
        kf = (env.last_raw_since_kf == 0).astype(np.int64)
        keep = np.asarray(valid, dtype=bool) & ~np.asarray(dones, dtype=bool)
        if keep.any():
            a = np.zeros([int(keep.sum()), env.action_dim], dtype=np.int64)
            a[:, 0] = kf[keep]
            a[:, 1] = GRID_IDX_RULES
            X.append(obs_prev[keep].copy())
            Y.append(a)
        obs_prev = obs
    return np.concatenate(X), np.concatenate(Y)


def train_bc(env, X, Y, epochs=30, batch=1024, lr=1e-3, val_frac=0.1, seed=0):
    """Supervised training of the (reference) policy to maximize the log-likelihood of the rules' actions."""
    torch.manual_seed(seed)
    policy = CustomActorCriticPolicy(env.observation_space, env.action_space, lambda _: lr, **make_policy_kwargs(env))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        policy.set_training_mode(True)
        perm = rng.permutation(tr)
        losses = []
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            _, logp, _ = policy.evaluate_actions(Xt[b], Yt[b])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        policy.set_training_mode(False)
        with torch.no_grad():
            pred, _, _ = policy.forward(Xt[va], deterministic=True)
        acc_kf = (pred[:, 0] == Yt[va][:, 0]).float().mean().item()
        acc_grid = (pred[:, 1] == Yt[va][:, 1]).float().mean().item()
        hist.append(dict(epoch=ep, train_nll=float(np.mean(losses)), val_acc_keyframe=acc_kf, val_acc_grid=acc_grid))
        print(f"[bc] epoch {ep:2d}  nll {np.mean(losses):.4f}  val acc keyframe {acc_kf:.3f}  grid {acc_grid:.3f}",
              flush=True)
    majority = max(Y[va, 0].mean(), 1 - Y[va, 0].mean())
    print(f"[bc] majority-class keyframe accuracy (baseline): {majority:.3f}  | {time.time() - t0:.0f}s", flush=True)
    return policy, hist, majority
