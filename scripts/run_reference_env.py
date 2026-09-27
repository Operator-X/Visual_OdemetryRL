"""Drive the authors' VecSVOEnv (reference/rl_vo/env/svo_wrapper.py) on our local TartanAir data.

Runs N parallel envs for K steps exactly like training does (GT initialization), with either SVO's own
heuristics or uniformly random agent actions, and reports timing, SVO stages and rewards.

    .venv/bin/python scripts/run_reference_env.py --envs 12 --steps 300 [--actions heuristic|random]
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

np.Inf = np.inf  # reference code predates NumPy 2 (np.Inf removed); keep reference/ untouched

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/svo-lib/build/svo_env"))
sys.path.insert(0, str(ROOT / "reference/rl_vo"))
from env.svo_wrapper import VecSVOEnv  # noqa: E402

PARAMS = ROOT / "third_party/svo-lib/svo_env/param/tartan_train.yaml"
CALIB = ROOT / "data/calibration/tartan_pinhole.yaml"
# Same values as reference/rl_vo/config/config.yaml
REWARD = SimpleNamespace(align_reward=0.01, keyframe_reward=0.0001, traj_length=5, nr_points_for_align=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/TartanAir"))
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--actions", choices=["heuristic", "random"], default="heuristic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    # TartanLoader starts env i on trajectory i (traj_idx = arange(num_envs)); with fewer trajectories than
    # envs it hits an IndexError that surfaces as a bare StopIteration. Fail early with a clear message.
    sys.path.insert(0, str(ROOT / "reference/rl_vo"))
    from dataloader.tartan_loader import test_split
    n_train = sum(1 for t in Path(args.data).glob("*/*/P*") if not any(s in str(t) for s in test_split))
    if args.envs > n_train:
        sys.exit(f"--envs {args.envs} > {n_train} training trajectories in {args.data}; use --envs <= {n_train}")

    env = VecSVOEnv(str(PARAMS), str(CALIB), args.data, args.envs, mode="train", reward_config=REWARD,
                    initialize_glog=True)
    n_traj = len(env.dataloader.trajectories_paths)
    print(f"trajectories (train split): {n_traj}, frames: {env.dataloader.nr_samples}")

    env.reset(use_gt_initialization=True)
    use_rl = args.actions == "random"
    step_t, stages, rewards, valid, dones_total = [], [], [], [], 0
    for _ in range(args.steps):
        action = np.stack([env.action_space.sample() for _ in range(args.envs)])
        t0 = time.perf_counter()
        _, reward, dones, _, valid_mask = env.step(action, use_RL_actions_bool=use_rl, use_gt_initialization=True)
        step_t.append(time.perf_counter() - t0)
        stages.append(env.svo_stages.copy())
        rewards.append(reward)
        valid.append(valid_mask)
        dones_total += int(np.sum(dones))

    step_t = np.array(step_t[5:])  # skip warm-up
    stages, rewards, valid = np.array(stages), np.array(rewards), np.array(valid)
    env_steps_per_s = args.envs / step_t.mean()
    print(f"\nactions={args.actions}  envs={args.envs}  steps={args.steps}")
    print(f"vec-step time: median {1e3 * np.median(step_t):.1f} ms, mean {1e3 * step_t.mean():.1f} ms "
          f"-> {env_steps_per_s:.0f} env-steps/s")
    print(f"  => 25M env-steps (paper budget) would take ~{25e6 / env_steps_per_s / 3600:.1f} h (rollout only)")
    frac = {name: float(np.mean(stages == k)) for k, name in enumerate(["paused", "init", "tracking", "reloc"])}
    print("SVO stage fractions:", {k: round(v, 3) for k, v in frac.items()})
    print(f"valid (RL-usable) steps: {valid.mean():.3f}   dones/resets: {dones_total}")
    print(f"reward: mean {rewards[valid].mean() if valid.any() else float('nan'):.5f} over valid steps")


if __name__ == "__main__":
    main()
