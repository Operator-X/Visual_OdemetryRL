"""Main results tables for the write-up, generated from the evaluation CSVs (never typed by hand).

    .venv/bin/python scripts/main_tables.py      -> results/tables/main_results.md

TUM: ATE [m] (first segment, Umeyama) averaged over the 5 sequences every method finishes, finished sequences,
failures per run, keyframe rate, SVO time per frame. Multi-seed methods: mean +- std over seeds (3 repeats each).
TartanAir: finished trajectories and failures per trajectory on the fixed core set (6) and, where available, on all
13 held-out trajectories (incl. Hard).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "eval"
COMMON = ["desk", "desk2", "plant", "teddy", "xyz"]
PAPER = {"Paper: SVO": {"desk": 0.681, "desk2": 0.898, "plant": 0.320, "teddy": 0.769, "xyz": 0.057},
         "Paper: RL-SVO": {"desk": 0.556, "desk2": 0.755, "plant": 0.260, "teddy": 0.697, "xyz": 0.062}}

# (label, TUM csvs per seed, TartanAir core csvs, TartanAir all-13 csvs, training data)
METHODS = [
    ("SVO rules (tuned)", ["tum/svo_rules_std/heuristic.csv"], ["tartanair/bc/heuristic.csv"],
     ["tartanair/v2_valall/heuristic.csv"], "-"),
    ("PPO, authors' setup", [f"tum/reward1/reward1_baseline_s{s}.csv" for s in range(3)],
     [f"tartanair/reward1/reward1_baseline_s{s}.csv" for s in range(3)], [], "3 scenes"),
    ("PPO + 20-frame reward window", [f"tum/reward1/reward1_long_window_s{s}.csv" for s in range(3)],
     [f"tartanair/reward1/reward1_long_window_s{s}.csv" for s in range(3)], [], "3 scenes"),
    ("Behavior cloning of the rules", ["tum/bc/bc_rules_s0.csv"], ["tartanair/bc/bc_rules_s0.csv"], [], "3 scenes"),
    ("BC -> PPO", [f"tum/bc/bc_ppo_s{s}.csv" for s in range(3)], [f"tartanair/bc/bc_ppo_s{s}.csv" for s in range(3)],
     [], "3 scenes"),
    ("Residual RL over the rules", [f"tum/residual/residual_s{s}.csv" for s in range(3)],
     [f"tartanair/residual/residual_s{s}.csv" for s in range(3)], [], "3 scenes"),
    ("Constrained PPO (kf rate 0.30)", [f"tum/constrained/constrained_lw_s{s}.csv" for s in range(3)],
     [f"tartanair/constrained/constrained_lw_s{s}.csv" for s in range(3)], [], "3 scenes"),
    ("PPO, authors' setup", ["tum/v2/ppo_v2_s0.csv"], ["tartanair/v2/ppo_v2_s0.csv"],
     ["tartanair/v2_valall/ppo_v2_s0.csv"], "7 scenes"),
    ("BC -> PPO", ["tum/v2/bc_ppo_v2_s0.csv"], ["tartanair/v2/bc_ppo_v2_s0.csv"],
     ["tartanair/v2_valall/bc_ppo_v2_s0.csv"], "7 scenes"),
    ("Constrained PPO (kf rate 0.30)", ["tum/v2/constrained_v2_s0.csv"], ["tartanair/v2/constrained_v2_s0.csv"],
     ["tartanair/v2_valall/constrained_v2_s0.csv"], "7 scenes"),
    ("Shadow-relative reward (sampled)", ["tum/v2_stoch/shadow_rel_v2_s0.csv"], [], [], "7 scenes"),
    ("Shadow-relative reward", ["tum/v2/shadow_rel_v2_s0.csv"], ["tartanair/v2/shadow_rel_v2_s0.csv"],
     ["tartanair/v2_valall/shadow_rel_v2_s0.csv"], "7 scenes"),
]


def per_seq(path, strip="rgbd_dataset_freiburg1_"):
    d = pd.read_csv(E / path)
    d["seq"] = d["trajectory"].str.replace(strip, "")
    return d.groupby("seq").agg(ate=("ate", "mean"), fin=("failures", lambda x: (x == 0).all()),
                                fails=("failures", "mean"), kf=("keyframe_rate", "mean"), ms=("ms_per_step", "mean"))


def fmt(vals, digits=3):
    vals = [v for v in vals if v == v]
    if not vals:
        return "-"
    m = np.mean(vals)
    return f"{m:.{digits}f}" + (f" ± {np.std(vals, ddof=1):.{digits}f}" if len(vals) > 1 else "")


def tartan(paths):
    if not paths:
        return "-", "-"
    gs = [per_seq(p, strip="") for p in paths if (E / p).exists()]
    if not gs:
        return "-", "-"
    n = len(gs[0])
    return fmt([g.fin.sum() for g in gs], 1) + f" / {n}", fmt([g.fails.mean() for g in gs], 2)


# Checkpoint-averaged table (protocol of 2026-10-02): the final checkpoint alone is a +-7% random draw, so each seed is
# scored by the mean over its last 3 policy snapshots (scripts/eval_snapshots.sh, evaluation repeat 0), then mean +- std
# over seeds. (label, runs, final-eval folder per dataset, checkpoints)
LAST3 = ["iter_00090", "iter_00120", "final"]
SNAP_METHODS = [
    ("PPO, authors' setup (last 3 of 1.5M)", ["ppo15_s0"], {"tum": "ppo15_1p5M", "tartanair": "snapshots/final"},
     ["iter_00330", "iter_00360", "final"], "3 scenes"),
    ("PPO, authors' setup (all 11 from 270k-1.5M)", ["ppo15_s0"], {"tum": "ppo15_1p5M", "tartanair": "snapshots/final"},
     [f"iter_{i:05d}" for i in range(90, 361, 30)] + ["final"], "3 scenes"),
    ("PPO, authors' setup (re-run with snapshots)", [f"ppo_snap_s{s}" for s in range(3)],
     {"tum": "ppo_snap", "tartanair": "ppo_snap"}, LAST3, "3 scenes"),
    ("BC -> PPO", [f"bc_ppo_s{s}" for s in range(3)], {"tum": "bc", "tartanair": "bc"}, LAST3, "3 scenes"),
    ("Residual RL over the rules", [f"residual_s{s}" for s in range(3)], {"tum": "residual", "tartanair": "residual"},
     LAST3, "3 scenes"),
    ("Constrained PPO (kf rate 0.30)", [f"constrained_lw_s{s}" for s in range(3)],
     {"tum": "constrained", "tartanair": "constrained"}, LAST3, "3 scenes"),
    ("BC -> constrained PPO", ["bc_constrained_lw_s0"], {"tum": "constrained", "tartanair": "constrained"}, LAST3,
     "3 scenes"),
    ("PPO, authors' setup", ["ppo_v2_s0"], {"tum": "v2", "tartanair": "v2"}, LAST3, "7 scenes"),
    ("BC -> PPO", ["bc_ppo_v2_s0"], {"tum": "v2", "tartanair": "v2"}, LAST3, "7 scenes"),
    ("Constrained PPO (kf rate 0.30)", ["constrained_v2_s0"], {"tum": "v2", "tartanair": "v2"}, LAST3, "7 scenes"),
    ("Shadow-relative reward", ["shadow_rel_v2_s0"], {"tum": "v2", "tartanair": "v2"}, LAST3, "7 scenes"),
]


def rep0(path, strip):
    d = pd.read_csv(path)
    d = d[d["repeat"] == 0].copy()
    d["seq"] = d["trajectory"].str.replace(strip, "")
    return d.set_index("seq")


def snap_path(ds, run, ckpt, finals):
    return E / ds / (finals[ds] if ckpt == "final" else f"snapshots/{ckpt}") / f"{run}.csv"


def seed_scores(run, finals, ckpts, rules):
    """One seed: per-sequence ATE averaged over the checkpoints, then summary numbers."""
    tum = [rep0(snap_path("tum", run, c, finals), "rgbd_dataset_freiburg1_") for c in ckpts]
    tar = [rep0(p, "") for c in ckpts if (p := snap_path("tartanair", run, c, finals)).exists()]
    # a sequence's ATE counts only from checkpoints that finished it (otherwise it is a short, flattering first segment)
    seq_ate = pd.concat([t["ate"].where(t["failures"] == 0) for t in tum], axis=1).mean(axis=1)
    return dict(
        ate=np.mean([seq_ate[q] for q in COMMON]),
        geo=np.exp(np.mean([np.log(seq_ate[q] / rules.loc[q, "ate"]) for q in COMMON])),
        common_failed=np.mean([(t.loc[COMMON, "failures"] > 0).sum() for t in tum]),
        fin=np.mean([(t["failures"] == 0).sum() for t in tum]), fails=np.mean([t["failures"].sum() for t in tum]),
        kf=np.mean([t["keyframe_rate"].mean() for t in tum]),
        tfin=np.mean([(t["failures"] == 0).sum() for t in tar]) if tar else np.nan,
        tfails=np.mean([t["failures"].mean() for t in tar]) if tar else np.nan, n_ckpt=len(tum), n_tar=len(tar))


def snapshot_table():
    rules = rep0(E / "tum/svo_rules_std/heuristic.csv", "rgbd_dataset_freiburg1_")
    rules_tar = rep0(E / "tartanair/bc/heuristic.csv", "")
    ref = np.mean([rules.loc[q, "ate"] for q in COMMON])
    lines = ["\n## Checkpoint-averaged results (protocol of 2026-10-02)\n",
             "The final checkpoint alone is a random draw (one run's TUM ATE varies +-7% between consecutive snapshots "
             "with no trend; see results/tables/snapshot_trends.md). Here each seed is scored by the MEAN over its "
             "last 3 policy snapshots (~270k, ~360k, 400k steps), evaluation repeat 0 only (the rules row too), then "
             "mean ± std over seeds. 'geo vs rules' = geometric mean over the 5 sequences of the per-sequence ATE ratio "
             "to the rules (not dominated by desk/desk2). 'common failed' = how many of the 5 common sequences had a "
             "failure; such (checkpoint, sequence) pairs are left out of the ATE (first-segment ATE would flatter them). Runs without snapshots (reward1 runs, BC clone) are "
             "not in this table.\n",
             "| Method | Train data | Seeds x ckpts | TUM ATE [m] | vs rules | geo vs rules | common failed /5 | "
             "TUM finished /9 | TUM failures/run | Keyframe rate | TartanAir core finished /6 | core failures/traj |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|",
             f"| SVO rules (tuned), repeat 0 | - | 1 | {ref:.3f} | +0.0% | 1.000 | "
             f"{(rules.loc[COMMON, 'failures'] > 0).sum()} | {(rules['failures'] == 0).sum()} | "
             f"{rules['failures'].sum():.1f} | {rules['keyframe_rate'].mean():.2f} | "
             f"{(rules_tar['failures'] == 0).sum()} | {rules_tar['failures'].mean():.2f} |"]
    for name, runs, finals, ckpts, data in SNAP_METHODS:
        ss = [seed_scores(r, finals, ckpts, rules) for r in runs if (ROOT / "runs" / r).exists()]
        if not ss:
            continue
        col = lambda k, d=3: fmt([s[k] for s in ss], d)  # noqa: E731
        lines.append(f"| {name} | {data} | {len(ss)} x {ss[0]['n_ckpt']} | {col('ate')} | "
                     f"{100 * (np.mean([s['ate'] for s in ss]) / ref - 1):+.1f}% | {col('geo')} | "
                     f"{col('common_failed', 1)} | {col('fin', 1)} | {col('fails', 1)} | {col('kf', 2)} | "
                     f"{col('tfin', 1)} | {col('tfails', 2)} |")
    return lines


def main():
    rules = per_seq(METHODS[0][1][0])
    ref = np.mean([rules.loc[q, "ate"] for q in COMMON])
    lines = ["# Main results (generated by scripts/main_tables.py — do not edit by hand)\n",
             f"TUM-RGBD: ATE [m] averaged over {', '.join(COMMON)} (the sequences every method finishes); same tuned "
             "SVO settings for all methods; 3 evaluation repeats per policy; multi-seed rows are mean ± std over "
             "seeds. 'vs rules' is relative to SVO's tuned rules. TartanAir: held-out trajectories; core = the fixed "
             "6 used throughout, all = 13 incl. Hard.\n",
             "| Method | Train data | Seeds | TUM ATE [m] | vs rules | TUM finished /9 | TUM failures/run | "
             "Keyframe rate | ms/frame | TartanAir core finished | core failures/traj | TartanAir all finished | "
             "all failures/traj |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, vals in PAPER.items():
        a = np.mean([vals[q] for q in COMMON])
        lines.append(f"| {name} | full TartanAir | ? | {a:.3f} | {100 * (a / ref - 1):+.0f}%* | 8 (paper) | - | - | - | - | - | - | - |")
    for name, tum, core, allv, data in METHODS:
        gs = [per_seq(p) for p in tum if (E / p).exists()]
        if not gs:
            continue
        ates = [np.mean([g.loc[q, "ate"] for q in COMMON]) for g in gs]
        rel = 100 * (np.mean(ates) / ref - 1)
        cf, cfl = tartan(core)
        af, afl = tartan(allv)
        lines.append(
            f"| {name} | {data} | {len(gs)} | {fmt(ates)} | {rel:+.1f}% | {fmt([g.fin.sum() for g in gs], 1)} | "
            f"{fmt([g.fails.sum() for g in gs], 1)} | {fmt([g.kf.mean() for g in gs], 2)} | "
            f"{fmt([g.ms.mean() for g in gs], 2)} | {cf} | {cfl} | {af} | {afl} |")
    lines.append("\n*Paper rows are relative to OUR tuned SVO rules (the paper's own SVO baseline is 0.545 on these "
                 "sequences; their RL-SVO is -14.5% vs their SVO).")
    lines += snapshot_table()
    out = ROOT / "results" / "tables" / "main_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
