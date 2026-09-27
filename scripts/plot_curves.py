"""Learning curves: one figure per metric, one small panel per variant (variant in blue vs baseline in dashed gray).

    .venv/bin/python scripts/plot_curves.py --tag pilot

Reads runs/<tag>_<variant>_s0/train.csv, writes results/figures/<tag>_curves_<metric>.png.
Small multiples instead of 12 colored lines in one plot (too many series to tell apart).
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): series slot 1 + chart chrome
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"

METRICS = {
    "reward_per_step": "Reward per step",
    "valid_ratio": "Share of steps usable for RL (tracking)",
    "keyframe_rate": "Keyframe rate (training, sampled actions)",
    "failures_per_1k": "Tracking failures per 1k steps",
}


def load(tag):
    runs = {}
    for d in sorted((ROOT / "runs").glob(f"{tag}_*_s0")):
        f = d / "train.csv"
        if f.exists():
            runs[d.name.removeprefix(tag + "_").removesuffix("_s0")] = pd.read_csv(f)
    return runs


def smooth(y, w):
    return y.rolling(w, min_periods=1, center=True).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smooth", type=int, default=3, help="rolling window in rollouts")
    args = ap.parse_args()

    runs = load(args.tag)
    if "baseline" not in runs:
        raise SystemExit(f"no runs/{args.tag}_baseline_s0/train.csv")
    base = runs.pop("baseline")
    names = sorted(runs)
    cols = 4
    rows = math.ceil(len(names) / cols)
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK})
    for key, label in METRICS.items():
        fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.3 * rows), sharex=True, sharey=True,
                                 facecolor=SURFACE)
        axes = axes.ravel()
        for ax, name in zip(axes, names):
            df = runs[name]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.plot(base["timesteps"] / 1e3, smooth(base[key], args.smooth), color=MUTED, linewidth=1.5,
                    linestyle=(0, (4, 2)), label="baseline")
            ax.plot(df["timesteps"] / 1e3, smooth(df[key], args.smooth), color=SERIES, linewidth=2, label=name)
            ax.set_title(name, fontsize=9, loc="left", color=INK)
        for j, ax in enumerate(axes):
            if j >= len(names):
                ax.axis("off")
            elif j + cols >= len(names):     # last panel in its column -> show x tick labels
                ax.tick_params(labelbottom=True)
        # direct label on the first panel so the gray line is never identified by color alone
        a0 = axes[0]
        a0.annotate("baseline", xy=(base["timesteps"].iloc[-1] / 1e3, smooth(base[key], args.smooth).iloc[-1]),
                    xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8, color=INK2)
        fig.supxlabel("Training steps (thousands)", color=INK2, fontsize=9)
        fig.suptitle(f"{label} — {args.tag} (blue: variant, dashed gray: baseline; rolling mean of "
                     f"{args.smooth} rollouts)", color=INK, fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        path = out / f"{args.tag}_curves_{key}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
