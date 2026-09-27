"""Evaluate trained runs (and SVO's heuristics) on held-out sequences, several repeats each.

    .venv/bin/python scripts/evaluate.py --runs runs/baseline_s0 runs/failure_penalty_s0 --heuristic --repeats 3
    .venv/bin/python scripts/evaluate.py --runs runs/*_s0 --dataset euroc --repeats 3

Writes results/eval/<dataset>/<out-tag>/<run>.csv (one row per trajectory x repeat).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import rlvo  # noqa: E402
from policies.attention_policy import CustomActorCriticPolicy  # noqa: E402
from rlvo.env import RLVOEnv  # noqa: E402
from rlvo.evaluate import evaluate  # noqa: E402
from rlvo.train import env_config, load_config, make_policy_kwargs  # noqa: E402

DATASETS = {
    "tartanair": dict(dir="data/TartanAir", params="tartan_train.yaml", calib="data/calibration/tartan_pinhole.yaml"),
    "euroc": dict(dir="data/EuRoC", params="euroc.yaml", calib="data/calibration/euroc_mono.yaml"),
}

_glog = [True]


def make_val_env(cfg, dataset):
    d = DATASETS[dataset]
    data = str(ROOT / d["dir"])
    if dataset == "tartanair":
        from dataloader.tartan_loader import test_split
        n = sum(1 for t in Path(data).glob("*/*/P*") if any(s in str(t) for s in test_split))
    else:
        from dataloader.euroc_loader import test_split
        n = sum(1 for s in test_split if (Path(data) / s).is_dir())
    env = RLVOEnv(str(rlvo.SVO_PARAMS / d["params"]), str(ROOT / d["calib"]), data, n, 'val', env_config(cfg),
                  initialize_glog=_glog[0], dataset=dataset)
    _glog[0] = False
    return env


def load_policy(run, env, checkpoint):
    pth = run / "Policy" / f"{checkpoint}.pth"
    env.load_rms(str(run / "Policy" / f"{checkpoint}_rms.npz"))
    policy = CustomActorCriticPolicy.load(str(pth), device="cpu")
    policy.set_training_mode(False)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--heuristic", action="store_true", help="also evaluate SVO's own heuristics")
    ap.add_argument("--dataset", default="tartanair", choices=list(DATASETS))
    ap.add_argument("--checkpoint", default="final")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-tag", default="default", help="results go to results/eval/<dataset>/<out-tag>/")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "eval" / args.dataset / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("heuristic", None)] if args.heuristic else []
    jobs += [(Path(r).name, Path(r)) for r in args.runs]

    for name, run in jobs:
        cfg = load_config(ROOT / "configs/base.yaml") if run is None else OmegaConf.load(run / "config.yaml")
        env = make_val_env(cfg, args.dataset)
        policy = None if run is None else load_policy(run, env, args.checkpoint)
        rows = []
        for rep in range(args.repeats):
            env.seed(rep)
            for r in evaluate(env, policy):
                rows.append({"run": name, "repeat": rep, **r})
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ates = [r["ate"] for r in rows]
        print(f"{name:28s} mean ATE {sum(a for a in ates if a == a) / max(1, sum(a == a for a in ates)):.3f} "
              f"| tracked {sum(r['tracked_frac'] for r in rows) / len(rows):.3f} "
              f"| failures {sum(r['failures'] for r in rows) / args.repeats:.1f}/run")
        del env


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
