"""Train an RL-VO agent.

    .venv/bin/python scripts/train.py --name baseline
    .venv/bin/python scripts/train.py --variant failure_penalty            # configs/variants/<name>.yaml on top of base
    .venv/bin/python scripts/train.py --variant gamma_0.9 --set total_timesteps=200000 seed=1

Outputs go to runs/<name>/: config.yaml, meta.json, train.csv, eval.csv, Policy/*.pth + *_rms.npz
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvo.train import load_config, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, help="name of configs/variants/<variant>.yaml")
    ap.add_argument("--name", default=None, help="run name (default: variant name or 'baseline')")
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. agent.gamma=0.9")
    args = ap.parse_args()

    paths = [ROOT / "configs/base.yaml"]
    if args.variant:
        paths.append(ROOT / "configs/variants" / f"{args.variant}.yaml")
    cfg = load_config(*paths, overrides=args.set)
    name = args.name or args.variant or "baseline"
    seed_tag = f"_s{cfg.seed}"
    train(cfg, ROOT / "runs" / (name + seed_tag))


if __name__ == "__main__":
    main()
