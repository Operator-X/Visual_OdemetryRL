"""Small grid search over SVO's heuristic parameters on a dataset (SVO rules only, no RL).

The paper's SVO baseline used per-dataset grid-searched parameters (not released). This searches the keyframe-selection
and tracking-quality thresholds, starting from the file the authors use for that dataset.

    .venv/bin/python scripts/svo_param_search.py --dataset tum_default --repeats 1 [--top 3 --top-repeats 3]

Uses fix_reset_gt_indexing=True (reference re-init bug fixed). Ranking (like the paper's tables): 1) sequences finished without any tracking failure, 2) mean ATE on the sequences
the DEFAULT parameters finish (a fixed set, comparable across configs), 3) total failures.
Writes results/svo_param_search/<dataset>/{configs/*.yaml, results.csv, summary.md}.
"""
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import rlvo  # noqa: E402
from evaluate import DATASETS  # noqa: E402
from rlvo.env import EnvConfig, RLVOEnv  # noqa: E402
from rlvo.evaluate import evaluate  # noqa: E402

GRID = {
    "kfselect_min_disparity": [30, 60],
    "kfselect_min_angle": [8, 15],
    "kfselect_min_dist_metric": [0.1, 0.2],
    "quality_min_fts": [25, 40],
}
_glog = [True]


def run(dataset, params_path, repeats):
    d = DATASETS[dataset]
    data = ROOT / d["dir"]
    mod = __import__("dataloader.euroc_loader" if dataset == "euroc" else "dataloader.tum_loader",
                     fromlist=["test_split"])
    ds = dataset.split("_")[0]
    n = sum(1 for s in mod.test_split if (data / s).is_dir())
    env = RLVOEnv(str(params_path), str(ROOT / d["calib"]), str(data), n, 'val', EnvConfig(fix_reset_gt_indexing=True),
                  initialize_glog=_glog[0], dataset=ds)
    _glog[0] = False
    rows = []
    for rep in range(repeats):
        env.seed(rep)
        rows += [{"repeat": rep, **r} for r in evaluate(env, None)]
    del env
    df = pd.DataFrame(rows)
    df["seq"] = df["trajectory"].str.replace("rgbd_dataset_freiburg1_", "")
    return df


def summarize(df, fixed_set):
    g = df.groupby("seq").agg(ate=("ate", "mean"), fails=("failures", "mean"), kf=("keyframe_rate", "mean"),
                              ms=("ms_per_step", "mean"))
    return dict(finished=int((g["fails"] == 0).sum()), fails=float(g["fails"].sum()),
                ate_fixed=float(g.loc[[s for s in fixed_set if s in g.index], "ate"].mean()),
                kf_rate=float(g["kf"].mean()), ms_step=float(g["ms"].mean())), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tum_default", choices=["tum_default", "euroc"],
                    help="searches from the authors' untuned file (tum_default = tartan_test.yaml)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--top-repeats", type=int, default=3)
    args = ap.parse_args()

    base_path = rlvo.SVO_PARAMS / DATASETS[args.dataset]["params"]
    base = yaml.safe_load(base_path.read_text())
    out = ROOT / "results" / "svo_param_search" / args.dataset.split("_")[0]
    (out / "configs").mkdir(parents=True, exist_ok=True)

    # default first: defines the fixed sequence set for the ATE comparison
    default_df = run(args.dataset, base_path, args.repeats)
    s0, g0 = summarize(default_df, [])
    fixed = list(g0.index[g0["fails"] == 0])
    s0, _ = summarize(default_df, fixed)
    print(f"default: finished {s0['finished']}/9  fails {s0['fails']:.1f}  ATE(fixed {len(fixed)}) {s0['ate_fixed']:.3f}")

    records = [dict(config="default", **{k: base.get(k) for k in GRID}, **s0)]
    for i, vals in enumerate(itertools.product(*GRID.values())):
        params = dict(base, **dict(zip(GRID, vals)))
        name = f"c{i:02d}"
        path = out / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(params, sort_keys=False))
        s, _ = summarize(run(args.dataset, path, args.repeats), fixed)
        records.append(dict(config=name, **dict(zip(GRID, vals)), **s))
        print(f"{name} {dict(zip(GRID, vals))}: finished {s['finished']}/9 fails {s['fails']:.1f} "
              f"ATE {s['ate_fixed']:.3f} kf {s['kf_rate']:.2f}")

    res = pd.DataFrame(records).sort_values(["finished", "ate_fixed", "fails"], ascending=[False, True, True])
    res.to_csv(out / "results.csv", index=False)

    # re-evaluate the top configs (+ default) with more repeats
    top = [c for c in res["config"] if c != "default"][:args.top]
    lines = []
    for c in ["default"] + top:
        path = base_path if c == "default" else out / "configs" / f"{c}.yaml"
        s, g = summarize(run(args.dataset, path, args.top_repeats), fixed)
        lines.append(f"### {c} ({args.top_repeats} repeats): finished {s['finished']}/9, failures {s['fails']:.1f}, "
                     f"ATE on default-finished seqs {s['ate_fixed']:.3f}, keyframe rate {s['kf_rate']:.2f}, "
                     f"{s['ms_step']:.2f} ms/step\n\n" + g.round(3).to_markdown() + "\n")
        print(lines[-1])
    (out / "summary.md").write_text(
        f"# SVO parameter search — {args.dataset}\n\nBase file: `{base_path.name}`. Grid: `{GRID}`.\n"
        f"Ranking: sequences finished without failure, then mean ATE on the {len(fixed)} sequences the default "
        f"finishes ({', '.join(fixed)}), then failures.\n\n## All configs ({args.repeats} repeat)\n\n"
        + res.round(3).to_markdown(index=False) + "\n\n## Top configs re-evaluated\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
