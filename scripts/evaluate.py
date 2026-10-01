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
from rlvo.train import env_config, load_config, make_policy_kwargs, set_svo_threads  # noqa: E402

DATASETS = {
    "tartanair": dict(dir="data/TartanAir", params="tartan_train.yaml", calib="data/calibration/tartan_pinhole.yaml"),
    "euroc": dict(dir="data/EuRoC", params="euroc.yaml", calib="data/calibration/euroc_mono.yaml"),
    # STANDARD for TUM: tartan_test.yaml + quality_min_fts 25 (our small grid search, scripts/svo_param_search.py;
    # the paper's SVO baseline was grid-searched too). Used for SVO rules AND RL agents.
    "tum": dict(dir="data/TUM-RGBD", params="tum_tuned.yaml", calib="data/calibration/tum.yaml"),
    # authors' untuned file (their config_eval.yaml uses tartan_test.yaml for TUM-RGBD)
    "tum_default": dict(dir="data/TUM-RGBD", params="tartan_test.yaml", calib="data/calibration/tum.yaml"),
}

_glog = [True]
# TartanAir evaluation sets: "core" = the fixed 6 held-out Easy trajectories of our first 3 scenes (all results so far);
# "all" = every held-out trajectory present on disk (more scenes + Hard). Default: the run's config (core).
CORE_VAL = ["japanesealley/Easy", "carwelding/Easy", "westerndesert/Easy"]
TARTAN_VAL = {"core": CORE_VAL, "all": None}
_tartan_val = ["config"]


def make_val_env(cfg, dataset, data_dir=None):
    set_svo_threads(cfg)
    d = DATASETS[dataset]
    data = str(ROOT / (data_dir or d["dir"]))
    extra_val = []
    if dataset == "tartanair":
        from rlvo.data import split_trajectories
        extra_val = list(cfg.data.get("extra_val_trajs", []))
        if _tartan_val[0] == "config":   # runs from before val_include existed -> the fixed core set
            val_include = cfg.data.get("val_include", CORE_VAL)
            val_include = None if val_include is None else list(val_include)
        else:
            val_include = TARTAN_VAL[_tartan_val[0]]
        n = len(split_trajectories(data, extra_val, val_include)[1])
    else:
        mod = __import__("dataloader.euroc_loader" if dataset == "euroc" else "dataloader.tum_loader",  # tum, tum_default
                         fromlist=["test_split"])
        n = sum(1 for s in mod.test_split if (Path(data) / s).is_dir())
        if n == 0:
            raise SystemExit(f"no {dataset} sequences found in {data}")
    env = RLVOEnv(str(rlvo.SVO_PARAMS / d["params"]), str(ROOT / d["calib"]), data, n, 'val', env_config(cfg),
                  initialize_glog=_glog[0], dataset=dataset.split("_")[0], extra_val=extra_val,
                  val_include=val_include if dataset == "tartanair" else None)
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
    ap.add_argument("--tartan-val", default="config", choices=["config", "core", "all"],
                    help="TartanAir evaluation set (core = fixed 6 trajectories; all = every held-out one on disk)")
    ap.add_argument("--data-dir", default=None, help="override the dataset folder (e.g. a folder of complete sequences)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax (writes to <out-tag>_stoch)")
    args = ap.parse_args()

    _tartan_val[0] = args.tartan_val
    tag = args.out_tag + ("_stoch" if args.stochastic else "") + ("_valall" if args.tartan_val == "all" else "")
    out_dir = ROOT / "results" / "eval" / args.dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("heuristic", None)] if args.heuristic else []
    jobs += [(Path(r).name, Path(r)) for r in args.runs]

    for name, run in jobs:
        cfg = load_config(ROOT / "configs/base.yaml") if run is None else OmegaConf.load(run / "config.yaml")
        env = make_val_env(cfg, args.dataset, args.data_dir)
        policy = None if run is None else load_policy(run, env, args.checkpoint)
        rows = []
        for rep in range(args.repeats):
            env.seed(rep)
            for r in evaluate(env, policy, deterministic=not args.stochastic):
                rows.append({"run": name, "repeat": rep, **r})
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ates = [r["ate"] for r in rows]
        alls = [r["ate_all"] for r in rows if r["ate_all"] == r["ate_all"]]
        print(f"{name:28s} ATE {sum(a for a in ates if a == a) / max(1, sum(a == a for a in ates)):.3f} "
              f"| ATE_all {sum(alls) / max(1, len(alls)):.3f} "
              f"| tracked {sum(r['tracked_frac'] for r in rows) / len(rows):.3f} "
              f"| failures {sum(r['failures'] for r in rows) / args.repeats:.1f}/run")
        del env


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
