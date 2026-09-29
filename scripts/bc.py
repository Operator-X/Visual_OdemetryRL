"""Behavior cloning of SVO's rules (warm start for PPO). Output looks like a training run, so evaluate.py works on it.

    .venv/bin/python scripts/bc.py [--name bc_rules] [--vec-steps 2500] [--epochs 30] [--set key=value ...]
    .venv/bin/python scripts/evaluate.py --runs runs/bc_rules_s0 --heuristic --dataset tum --repeats 3 --out-tag bc
    # PPO fine-tuning from it:  scripts/train.py --name bc_ppo --set init_policy=runs/bc_rules_s0

Writes runs/<name>_s<seed>/: config.yaml, meta.json, bc_history.csv, Policy/final.pth, Policy/final_rms.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rlvo.bc import collect_demos, train_bc  # noqa: E402
from rlvo.train import git_commit, load_config, make_envs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bc_rules")
    ap.add_argument("--vec-steps", type=int, default=2500, help="demo collection steps (x n_envs samples, before filtering)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. seed=1")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", overrides=args.set)
    run = ROOT / "runs" / f"{args.name}_s{cfg.seed}"
    (run / "Policy").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    env, _ = make_envs(cfg, val=False)
    env.seed(cfg.seed)
    X, Y = collect_demos(env, args.vec_steps)
    print(f"[bc] {len(X)} demo samples, keyframe rate of the rules {Y[:, 0].mean():.3f}", flush=True)
    policy, hist, majority = train_bc(env, X, Y, epochs=args.epochs, lr=args.lr, seed=cfg.seed)

    policy.save(str(run / "Policy" / "final.pth"))
    env.save_rms(str(run / "Policy" / "final_rms.npz"))
    pd.DataFrame(hist).to_csv(run / "bc_history.csv", index=False)
    (run / "meta.json").write_text(json.dumps(dict(
        git_commit=git_commit(), kind="behavior_cloning_of_svo_rules", samples=int(len(X)),
        rules_keyframe_rate=float(Y[:, 0].mean()), majority_baseline=float(majority),
        final_val_acc_keyframe=hist[-1]["val_acc_keyframe"], minutes=(time.time() - t0) / 60), indent=2))
    print(f"[bc] saved -> {run}", flush=True)


if __name__ == "__main__":
    main()
