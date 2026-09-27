"""Build the baseline-vs-variants comparison table from evaluation CSVs and training logs.

    .venv/bin/python scripts/compare.py --tag pilot [--dataset tartanair]

Writes results/tables/<tag>_<dataset>.md and .csv. Per run: mean ATE over trajectories x repeats (NaN-ignoring),
tracked fraction, failures per trajectory, keyframe rate, final training stats, wall time, and deltas vs baseline.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="tartanair")
    args = ap.parse_args()

    eval_dir = ROOT / "results" / "eval" / args.dataset / args.tag
    rows = []
    for f in sorted(eval_dir.glob("*.csv")):
        df = pd.read_csv(f)
        run = f.stem
        variant = "heuristic (SVO rules)" if run == "heuristic" else run.removeprefix(args.tag + "_").removesuffix("_s0")
        row = dict(variant=variant, ate_m=df["ate"].mean(skipna=True), ate_median_m=df["ate"].median(skipna=True),
                   ate_coverage=df["first_sub_frac"].mean(),   # share of the sequence the ATE segment covers
                   tracked=df["tracked_frac"].mean(), failures_per_traj=df["failures"].mean(),
                   keyframe_rate=df["keyframe_rate"].mean())
        run_dir = ROOT / "runs" / run
        if (run_dir / "train.csv").exists():
            tr = pd.read_csv(run_dir / "train.csv")
            last = tr.tail(max(1, len(tr) // 4))   # last quarter of training
            meta = json.loads((run_dir / "meta.json").read_text())
            row.update(train_reward=last["reward_per_step"].mean(), train_valid=last["valid_ratio"].mean(),
                       train_fail_per_1k=last["failures_per_1k"].mean(), steps=int(meta.get("timesteps", 0)),
                       minutes=60 * meta.get("train_hours", np.nan))
        rows.append(row)
    if not rows:
        sys.exit(f"no evaluation CSVs in {eval_dir}")

    t = pd.DataFrame(rows)
    order = ["heuristic (SVO rules)", "baseline"]
    t["_o"] = t["variant"].map(lambda v: order.index(v) if v in order else 2)
    t = t.sort_values(["_o", "variant"]).drop(columns="_o").reset_index(drop=True)
    if "baseline" in set(t["variant"]):
        b = t[t["variant"] == "baseline"].iloc[0]
        t["d_ate_%"] = 100 * (t["ate_m"] - b["ate_m"]) / b["ate_m"]
        t["d_tracked_pts"] = 100 * (t["tracked"] - b["tracked"])

    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"{args.tag}_{args.dataset}.csv", index=False)
    md = t.to_markdown(index=False, floatfmt=".3f")
    (out / f"{args.tag}_{args.dataset}.md").write_text(
        f"# {args.tag} — {args.dataset}\n\nATE: first sub-trajectory before a tracking failure (authors' metric), "
        f"mean over held-out trajectories x repeats. `d_*` columns are relative to the baseline.\n\n"
        f"**Read ATE together with `ate_coverage`:** a policy that fails early gets a short first segment and a small, "
        f"flattering ATE. Only compare ATE between runs with similar coverage.\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
