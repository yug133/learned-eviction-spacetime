"""
paper_figures.py - the three summary figures, sized for ONE column.

IEEE single column = 3.5 in wide; all text 8 pt; vector PDF. Same colours, markers and
label placement as plot.py (one fixed colour + marker per policy; Belady is a dashed grey
reference line, not a competitor). Reads only the CSVs written by run_all.py.

  per_day.pdf  per-day miss ratio, days 1-15 (partial day 16 left out), rows = cache
               sizes, columns = seed 0 / seed 1; LeCaR = mean of its 5 run seeds
  weights.pdf  LeCaR's w_lru over time, one panel per cache size, seed 0 solid and seed 1
               dashed; mean over 6-hour windows and LeCaR's 5 run seeds
  pareto.pdf   miss ratio (excl. day 1) vs time per request relative to LRU, rows =
               cache sizes, columns = seeds; Belady marked as the offline bound

Run from the repo root:
  python3 -m experiments.paper_figures --seed0 results/wiki_0.01 \\
      --seed1 results/wiki_0.01_seed1 --out figures
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from experiments.plot import (AXIS, INK, LINE_WIDTH, MUTED, POLICY_ORDER, STYLE,
                              place_point_labels, read_csv, read_partial_days)

WIDTH = 3.5    # inches: IEEE single column
FRACTIONS = ["0.001", "0.01", "0.1"]
SIZE_NAME = {"0.001": "0.1%", "0.01": "1%", "0.1": "10%"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)    # for a quick look
    plt.close(fig)


def per_day_series(folder, policy, fraction):
    """(days, mean miss ratio) for complete days only."""
    partial = read_partial_days(folder)
    by_day = defaultdict(list)
    for r in read_csv(os.path.join(folder, "per_day.csv")):
        day = int(r["day"])
        if r["policy"] == policy and r["cache_fraction"] == fraction and day not in partial:
            by_day[day].append(float(r["miss_ratio"]))
    days = sorted(by_day)
    return days, [sum(by_day[d]) / len(by_day[d]) for d in days]


def plot_per_day(folders, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 4.4), sharex=True, sharey="row")
    for col, (label, folder) in enumerate(folders):
        for row, fraction in enumerate(FRACTIONS):
            ax = axes[row][col]
            for policy in POLICY_ORDER:
                colour, marker, style = STYLE[policy]
                days, values = per_day_series(folder, policy, fraction)
                ax.plot(days, values, color=colour, marker=marker, linestyle=style,
                        linewidth=1.0, markersize=2.5, zorder=10 - POLICY_ORDER.index(policy),
                        label=policy)
            ax.set_xticks([1, 5, 10, 15])
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{SIZE_NAME[fraction]} cache\nmiss ratio")
            if row == 2:
                ax.set_xlabel("Day")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.07), columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, out_dir, "per_day")


def weight_series(folder, fraction, window_days=0.25):
    """Window centres and mean w_lru (over 6-hour windows and LeCaR's run seeds)."""
    partial = read_partial_days(folder)
    windows = defaultdict(list)
    for r in read_csv(os.path.join(folder, "weights.csv")):
        if r["policy"] != "LeCaR" or r["cache_fraction"] != fraction:
            continue
        t = float(r["time_days"])
        if int(t) + 1 in partial:
            continue
        windows[int(t // window_days)].append(float(r["w_lru"]))
    keys = sorted(windows)
    return ([(w + 0.5) * window_days for w in keys],
            [sum(windows[w]) / len(windows[w]) for w in keys])


def plot_weights(folders, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.9), sharex=True, sharey=True)
    colour = STYLE["LeCaR"][0]
    for ax, fraction in zip(axes, FRACTIONS):
        ax.axhline(0.5, color=AXIS, linewidth=0.8)
        for (label, folder), style in zip(folders, ("-", "--")):
            x, y = weight_series(folder, fraction)
            ax.plot(x, y, color=colour, linestyle=style, linewidth=LINE_WIDTH * 0.8, label=label)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"{SIZE_NAME[fraction]}\n$w_{{LRU}}$")
    axes[-1].set_xlabel("Time (days)")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(h_pad=1.0)
    save(fig, out_dir, "weights")


def plot_pareto(folders, out_dir):
    # y-axes are NOT shared across seeds: their absolute miss ratios differ (one very
    # popular object is in seed 0 only), so each panel gets its own range.
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 4.8))
    to_label = []
    for col, (label, folder) in enumerate(folders):
        miss = {(r["policy"], r["cache_fraction"]): float(r["miss_ratio_excl_day1_mean"])
                for r in read_csv(os.path.join(folder, "miss_ratio.csv"))}
        ratio = {(r["policy"], r["cache_fraction"]): float(r["ratio_to_lru"])
                 for r in read_csv(os.path.join(folder, "timing.csv"))}
        for row, fraction in enumerate(FRACTIONS):
            ax = axes[row][col]
            ax.grid(axis="x", visible=True)
            points = [(ratio[(p, fraction)], miss[(p, fraction)], p)
                      for p in POLICY_ORDER if p != "Belady"]
            for x, y, policy in points:
                ax.plot(x, y, marker=STYLE[policy][1], markersize=4.5, linestyle="none",
                        color=INK, markerfacecolor=INK)
            belady = miss[("Belady", fraction)]
            ax.axhline(belady, color=STYLE["Belady"][0], linestyle="--", linewidth=0.9)
            ax.set_xlim(0, 10.5)
            low = min(belady, min(p[1] for p in points))
            high = max(p[1] for p in points)
            ax.set_ylim(low - 0.06 * (high - low), high + 0.18 * (high - low))
            belady_text = ax.text(10.3, belady, "Belady (offline)", ha="right", va="bottom",
                                  fontsize=7, color=MUTED)
            to_label.append((ax, points, belady_text))
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{SIZE_NAME[fraction]} cache\nmiss ratio")
            if row == 2:
                ax.set_xlabel("Time per request / LRU")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    for ax, points, belady_text in to_label:
        place_point_labels(fig, ax, points, belady_text)
    save(fig, out_dir, "pareto")


def main():
    parser = argparse.ArgumentParser(description="Single-column figures for the paper.")
    parser.add_argument("--seed0", required=True)
    parser.add_argument("--seed1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    folders = [("Seed 0", args.seed0), ("Seed 1", args.seed1)]
    plot_per_day(folders, args.out)
    plot_weights(folders, args.out)
    plot_pareto(folders, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
