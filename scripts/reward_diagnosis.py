"""Does the reward tell the agent which keyframe decisions lead to better accuracy? (no training)

A. Strategy level (TUM, 9 seqs, standard tuned SVO params, GT init, fix on): SVO rules and fixed "keyframe with
   probability p" strategies. Per strategy: mean reward per tracked step (authors' absolute-error reward, the
   normalized-error variant computed from the same run, keyframe penalty) vs true accuracy (ATE on sequences finished
   by all strategies) and failures. Informative reward <=> reward ranking matches accuracy ranking.
B. Decision level (TartanAir train, 12 envs, random keyframes p=0.3): position reward in the k=1..5 frames after
   "keyframe" vs "no keyframe" at t, as an effect size (difference of means / pooled std). Tiny effect = the learning
   signal for keyframe decisions is buried in per-step noise.

    .venv/bin/python scripts/reward_diagnosis.py [--probs 0.05 0.1 0.2 0.3 0.5 1.0] [--steps-b 600]
Writes results/reward_diagnosis/{strategies.csv, decision_effect.csv, summary.md}.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("RLVO_SVO_THREADS", "12")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rlvo  # noqa: E402
from rlvo.data import split_trajectories  # noqa: E402
from rlvo.env import EnvConfig, RewardConfig, RLVOEnv  # noqa: E402
from rlvo.evaluate import MIN_SEGMENT, _aligned_sq_errors  # noqa: E402

GRID_IDX = 2          # grid size 30 (5*2+20) for all fixed strategies
TAU_NORM = 0.25       # normalized-error threshold (calibrated, see configs/variants/normalized_error.yaml)
_glog = [True]


class Spy:
    """Wraps env.compute_reward to record the position error and the window path length of every rewarded step."""

    def __init__(self, env):
        self.env, self.orig = env, env.compute_reward
        env.compute_reward = self
        self.records = []

    def __call__(self, dones, poses, gt_poses, svo_dones, valid_stages, info, action):
        out = self.orig(dones, poses, gt_poses, svo_dones, valid_stages, info, action)
        env, err = self.env, out[2]
        n_valid = np.minimum(env.env_steps, env.reward_traj_length).astype(int)
        seg = np.linalg.norm(np.diff(env.gt_positions, axis=1), axis=2)
        mask = np.arange(seg.shape[1])[None, :] < (n_valid - 1)[:, None]
        path = np.maximum((seg * mask).sum(1), 0.05)
        rewarded = np.logical_and(~dones.astype(bool), env.env_steps > env.reward_nr_points_for_align)
        self.records.append(dict(err=err.copy(), path=path, rewarded=rewarded.copy(), valid=np.asarray(valid_stages),
                                 kf_action=np.asarray(action[:, 0]).copy()))
        return out


def make_env(params, calib, data, n, mode, dataset, extra_val=()):
    env = RLVOEnv(str(params), str(calib), str(data), n, mode, EnvConfig(fix_reset_gt_indexing=True, reward=RewardConfig()),
                  initialize_glog=_glog[0], dataset=dataset, extra_val=extra_val)
    _glog[0] = False
    return env


def run_strategy(env, prob, rng):
    """Run every val trajectory once. prob=None -> SVO rules. Returns per-trajectory metrics incl. reward stats."""
    n = env.num_envs
    spy = Spy(env)
    lengths = env.dataloader.nr_samples_per_traj.astype(int)
    T = int(lengths.max())
    names = [str(p).split("/")[-1].replace("rgbd_dataset_freiburg1_", "") for p in env.dataloader.trajectories_paths]
    completed = np.zeros(n, bool)
    pos, gt = np.zeros([n, T, 3]), np.zeros([n, T, 3])
    valid, dones_a, fails = np.zeros([n, T], bool), np.zeros([n, T], bool), np.zeros([n, T], bool)
    active = np.zeros([n, T], bool)
    env.reset(use_gt_initialization=True)
    for t in range(T):
        acts = np.zeros([n, env.action_dim], dtype=np.int64)
        acts[:, 1] = GRID_IDX
        if prob is not None:
            acts[:, 0] = (rng.random(n) < prob).astype(np.int64)
        _, _, d, infos, vm = env.step(acts, use_RL_actions_bool=prob is not None, use_gt_initialization=True)
        for i, info in enumerate(infos):
            completed[i] |= bool(info['new_seq'])
            if completed[i]:
                continue
            active[i, t] = True
            valid[i, t], dones_a[i, t], fails[i, t] = vm[i], d[i], info.get('svo_failure', False)
            if vm[i]:
                pos[i, t], gt[i, t] = info['position'], info['gt_position']
        if completed.all():
            break
    env.compute_reward = spy.orig
    R = spy.records[:T]
    rows = []
    seg_id = np.cumsum(dones_a, axis=1)
    for i in range(n):
        # reward stats over this trajectory's rewarded steps
        m = [(r['err'][i], r['path'][i]) for t_, r in enumerate(R) if t_ < T and active[i, t_] and r['rewarded'][i]]
        e = np.array([a for a, _ in m]) if m else np.array([np.nan])
        L = np.array([b for _, b in m]) if m else np.array([np.nan])
        abs_r = 0.01 * np.maximum(0.2 - e, -1)
        norm_r = 0.01 * np.maximum(TAU_NORM - e / L, -1)
        kf = np.array([r['kf_action'][i] for t_, r in enumerate(R) if active[i, t_] and r['valid'][i]]) if prob is not None \
            else np.array([np.nan])
        has = np.abs(pos[i]).sum(1) != 0
        first = (seg_id[i] == 0) & valid[i] & has
        ate = float(np.sqrt(_aligned_sq_errors(gt[i, first], pos[i, first]).mean())) if first.sum() > 3 else np.nan
        sq = [_aligned_sq_errors(gt[i, mk], pos[i, mk]) for k in np.unique(seg_id[i])
              for mk in [(seg_id[i] == k) & valid[i] & has] if mk.sum() >= MIN_SEGMENT]
        rows.append(dict(seq=names[i], ate=ate, ate_all=float(np.sqrt(np.concatenate(sq).mean())) if sq else np.nan,
                         failures=int(fails[i].sum()), tracked=float(valid[i].sum() / max(active[i].sum(), 1)),
                         pos_reward=float(np.nanmean(abs_r)), norm_reward=float(np.nanmean(norm_r)),
                         mean_err=float(np.nanmean(e)), frac_positive=float(np.nanmean(abs_r > 0)),
                         kf_penalty=float(-1e-4 * np.nanmean(kf)) if prob is not None else 0.0))
    return rows


def decision_effect(steps, prob=0.3, seed=0):
    """B: effect of the keyframe action at t on the position reward at t+k (random policy, TartanAir train)."""
    data = rlvo.DATA / "TartanAir"
    extra = ["japanesealley/Easy/P003", "carwelding/Easy/P005", "westerndesert/Easy/P004"]
    env = make_env(rlvo.SVO_PARAMS / "tartan_train.yaml", rlvo.DATA / "calibration/tartan_pinhole.yaml", data, 12,
                   'train', 'tartanair', extra)
    spy = Spy(env)
    rng = np.random.default_rng(seed)
    env.reset(use_gt_initialization=True)
    for _ in range(steps):
        acts = np.zeros([12, env.action_dim], dtype=np.int64)
        acts[:, 1] = GRID_IDX
        acts[:, 0] = (rng.random(12) < prob).astype(np.int64)
        env.step(acts, use_RL_actions_bool=True, use_gt_initialization=True)
    R = spy.records
    E = np.stack([r['err'] for r in R]); Lp = np.stack([r['path'] for r in R])
    RW = np.stack([r['rewarded'] for r in R]); V = np.stack([r['valid'] for r in R]); K = np.stack([r['kf_action'] for r in R])
    out = []
    for name, rew in (("absolute (authors)", 0.01 * np.maximum(0.2 - E, -1)),
                      ("normalized", 0.01 * np.maximum(TAU_NORM - E / Lp, -1))):
        for k in range(1, 6):
            a, b = [], []
            for t in range(len(R) - k):
                ok = V[t] & RW[t + k]
                a += list(rew[t + k][ok & (K[t] == 1)])
                b += list(rew[t + k][ok & (K[t] == 0)])
            a, b = np.array(a), np.array(b)
            pooled = np.sqrt((a.var() + b.var()) / 2)
            out.append(dict(reward=name, k=k, mean_after_kf=a.mean(), mean_after_no_kf=b.mean(),
                            diff=a.mean() - b.mean(), effect_size_d=(a.mean() - b.mean()) / pooled,
                            n_kf=len(a), n_no_kf=len(b)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--steps-b", type=int, default=600)
    args = ap.parse_args()
    out = ROOT / "results" / "reward_diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- A
    from dataloader.tum_loader import test_split
    env = make_env(rlvo.SVO_PARAMS / "tum_tuned.yaml", rlvo.DATA / "calibration/tum.yaml", rlvo.DATA / "TUM-RGBD",
                   len(test_split), 'val', 'tum')
    rng = np.random.default_rng(0)
    rows = []
    for p in [None] + args.probs:
        env.seed(0)
        label = "SVO rules" if p is None else f"p_kf={p}"
        for r in run_strategy(env, p, rng):
            rows.append(dict(strategy=label, **r))
        print(f"A: {label} done", flush=True)
    del env
    A = pd.DataFrame(rows)
    A.to_csv(out / "strategies.csv", index=False)
    fin = A.groupby("seq")["failures"].max()
    common = list(fin[fin == 0].index)                      # sequences finished by every strategy
    S = A.groupby("strategy", sort=False).agg(
        pos_reward=("pos_reward", "mean"), norm_reward=("norm_reward", "mean"), kf_penalty=("kf_penalty", "mean"),
        frac_positive=("frac_positive", "mean"), mean_err_m=("mean_err", "mean"), failures=("failures", "sum"),
        tracked=("tracked", "mean"))
    S["total_reward"] = S["pos_reward"] + S["kf_penalty"]
    S[f"ATE_common({len(common)})"] = A[A.seq.isin(common)].groupby("strategy", sort=False)["ate"].mean()
    S["ATE_all"] = A.groupby("strategy", sort=False)["ate_all"].mean()

    def rank_corr(x, y):
        return pd.Series(x).rank().corr(pd.Series(y).rank())
    ate_col = f"ATE_common({len(common)})"
    corr = {c: rank_corr(S[c].values, -S[ate_col].values) for c in ["total_reward", "pos_reward", "norm_reward"]}

    # ---------------- B
    B = decision_effect(args.steps_b)
    B.to_csv(out / "decision_effect.csv", index=False)

    md = ["# Reward diagnosis\n", "## A. Strategy level (TUM, 9 seqs, tum_tuned, GT init)\n",
          f"ATE on the {len(common)} sequences every strategy finishes: {', '.join(common)}.\n",
          S.round(5).to_markdown(), "\n\nSpearman rank correlation between reward and accuracy (higher = reward ranks "
          "strategies like true accuracy does; 1 = perfect):\n",
          "\n".join(f"- {k}: {v:+.2f}" for k, v in corr.items()),
          "\n\n## B. Decision level (TartanAir train, random keyframes p=0.3)\n",
          "Position reward k frames after a keyframe vs no keyframe at t. Effect size d = diff / pooled std "
          "(|d| < 0.2 is conventionally 'negligible').\n", B.round(6).to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
