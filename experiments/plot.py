"""
plot.py - draw every figure from the CSV files written by run_all.py.

Run from the repo root:
  python3 -m experiments.plot --results results/wiki_0.001/ [--shift-day 7]

Writes PNG (for viewing) and PDF (vector, for print) into <results>/plots/.

Style: each policy has ONE fixed colour and marker everywhere (colour follows the policy,
never its rank); markers keep the figures readable in grayscale print. Colours are the
first five categorical slots of the dataviz reference palette, checked with its validator
(light mode, adjacent pairs: worst CVD Delta E 9.1, normal-vision 19.6). Belady is not a
competitor but the offline bound, so it is a dashed gray reference line. Bar charts and the
Pareto scatter use one neutral ink, because the axis or a text label already names each
policy. Never two y-axes on one chart.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")    # no display needed
import matplotlib.pyplot as plt  # noqa: E402

POLICY_ORDER = ["LeCaR", "LeCaR-LCS", "ARC", "LRU", "LFU", "Belady"]
STYLE = {    # colour, marker, line style
    "LeCaR": ("#2a78d6", "o", "-"),
    "LeCaR-LCS": ("#eb6834", "s", "-"),
    "ARC": ("#1baf7a", "^", "-"),
    "LRU": ("#eda100", "D", "-"),
    "LFU": ("#e87ba4", "v", "-"),
    "Belady": ("#52514e", "x", "--"),
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAR = "#52514e"
WIDE = 7.16      # IEEE double-column width, inches
LINE_WIDTH = 1.5

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def policies_in(rows):
    present = {r["policy"] for r in rows}
    return [p for p in POLICY_ORDER if p in present]


def size_label(fraction):
    return f"{float(fraction) * 100:g}% of uniques"


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def figure_legend(fig, axes, ncol=6):
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.06))


def line(ax, x, y, policy, yerr=None, **extra):
    colour, marker, style = STYLE[policy]
    # White ring on filled markers keeps overlapping series visible ('x' has no fill, so it
    # keeps its own colour). Earlier policies in POLICY_ORDER are drawn on top (LeCaR first).
    ring = {} if marker == "x" else {"markeredgecolor": "white", "markeredgewidth": 0.7}
    ax.errorbar(x, y, yerr=yerr, color=colour, marker=marker, linestyle=style,
                linewidth=LINE_WIDTH, markersize=4.5, capsize=2 if yerr else 0,
                zorder=10 - POLICY_ORDER.index(policy), label=policy, **ring, **extra)


def mark_shift(ax, shift_day, x_is_day_number):
    if shift_day is None:
        return
    # day numbers start at 1, so the boundary before day shift_day + 1 is at shift_day + 0.5
    x = shift_day + 0.5 if x_is_day_number else shift_day
    ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=":")


def add_notes(fig, notes):
    """Small grey notes under the figure, one per line."""
    for i, note in enumerate(notes):
        fig.text(0.5, -0.10 - 0.05 * i, note, ha="center", fontsize=6.5, color=MUTED)


def shift_notes(shift_day):
    return [] if shift_day is None else ["Dotted line: popularity shift built into the synthetic trace"]


def read_partial_days(results):
    """1-based numbers of the days run_all marked as partial (empty for older results)."""
    rows = read_csv(os.path.join(results, "per_day.csv"))
    return {int(r["day"]): float(r["day_coverage"]) for r in rows if r.get("partial") == "True"}


def partial_note(partial, excluded):
    if not partial:
        return []
    days = ", ".join(f"day {d} ({c:.0%} of a day)" for d, c in sorted(partial.items()))
    return [f"Partial {'days excluded' if excluded else 'days shown'}: {days}"]


# --- figures ---

def plot_miss_ratio(results, out_dir):
    rows = read_csv(os.path.join(results, "miss_ratio.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.6), sharey=True)
    for ax, column, title in [(axes[0], "miss_ratio", "All days"),
                              (axes[1], "miss_ratio_excl_day1", "Excluding day 1 (cold start)")]:
        for policy in policies_in(rows):
            points = sorted((float(r["cache_fraction"]) * 100, float(r[f"{column}_mean"]),
                             float(r[f"{column}_std"])) for r in rows if r["policy"] == policy)
            x, y, err = zip(*points)
            line(ax, x, y, policy, yerr=err if policy.startswith("LeCaR") else None)
        ax.set_xscale("log")
        ax.set_xticks([x for x in sorted({float(r["cache_fraction"]) * 100 for r in rows})])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_xlabel("Cache size (% of unique objects)")
        ax.set_title(title)
    axes[0].set_ylabel("Miss ratio")
    figure_legend(fig, axes[0])
    save(fig, out_dir, "miss_ratio_vs_size")


def plot_per_day(results, out_dir, shift_day, include_partial):
    rows = read_csv(os.path.join(results, "per_day.csv"))
    partial = read_partial_days(results)
    if not include_partial:
        rows = [r for r in rows if int(r["day"]) not in partial]
    fractions = sorted({r["cache_fraction"] for r in rows}, key=float)
    fig, axes = plt.subplots(1, len(fractions), figsize=(WIDE, 2.4), sharey=True, squeeze=False)
    for ax, fraction in zip(axes[0], fractions):
        for policy in policies_in(rows):
            by_day = defaultdict(list)    # mean over seeds for LeCaR
            for r in rows:
                if r["policy"] == policy and r["cache_fraction"] == fraction and r["miss_ratio"]:
                    by_day[int(r["day"])].append(float(r["miss_ratio"]))
            days = sorted(by_day)
            line(ax, days, [sum(by_day[d]) / len(by_day[d]) for d in days], policy)
        mark_shift(ax, shift_day, x_is_day_number=True)
        ax.set_title(size_label(fraction))
        ax.set_xlabel("Day")
        ax.set_xticks(range(1, max(days) + 1, 2))
    axes[0][0].set_ylabel("Miss ratio (per day)")
    figure_legend(fig, axes[0][0])
    add_notes(fig, shift_notes(shift_day) + partial_note(partial, not include_partial))
    save(fig, out_dir, "miss_ratio_per_day")


def plot_weights(results, out_dir, shift_day, include_partial, window_days=0.25):
    """w_lru moves on every regret, so raw samples are unreadable. Plot the mean over
    `window_days` windows (6 hours) and over seeds; the band spans the per-seed means.
    Points inside partial days are dropped unless include_partial (same days as the
    per-day plot; time_days in [d - 1, d) belongs to day number d)."""
    rows = read_csv(os.path.join(results, "weights.csv"))
    if not rows:
        return
    partial = read_partial_days(results)
    if not include_partial:
        rows = [r for r in rows if int(float(r["time_days"])) + 1 not in partial]
    fractions = sorted({r["cache_fraction"] for r in rows}, key=float)
    fig, axes = plt.subplots(1, len(fractions), figsize=(WIDE, 2.4), sharey=True, squeeze=False)
    for ax, fraction in zip(axes[0], fractions):
        ax.axhline(0.5, color=AXIS, linewidth=0.8)
        for policy in policies_in(rows):
            colour = STYLE[policy][0]
            per_seed = defaultdict(lambda: defaultdict(list))    # seed -> window -> values
            for r in rows:
                if r["policy"] == policy and r["cache_fraction"] == fraction:
                    window = int(float(r["time_days"]) // window_days)
                    per_seed[r["seed"]][window].append(float(r["w_lru"]))
            windows = sorted(set().union(*[set(w) for w in per_seed.values()]))
            centres, means, lows, highs = [], [], [], []
            for w in windows:
                seed_means = [sum(v[w]) / len(v[w]) for v in per_seed.values() if w in v]
                centres.append((w + 0.5) * window_days)
                means.append(sum(seed_means) / len(seed_means))
                lows.append(min(seed_means))
                highs.append(max(seed_means))
            ax.fill_between(centres, lows, highs, color=colour, alpha=0.2, linewidth=0)
            ax.plot(centres, means, color=colour, linewidth=LINE_WIDTH, label=policy)
        mark_shift(ax, shift_day, x_is_day_number=False)
        ax.set_ylim(0, 1)
        ax.set_title(size_label(fraction))
        ax.set_xlabel("Time (days)")
    axes[0][0].set_ylabel("w_lru (weight of LRU)")
    figure_legend(fig, axes[0][0], ncol=2)
    hours = window_days * 24
    add_notes(fig, [f"Mean of w_lru over {hours:g}-hour windows and seeds; band = range of the "
                    f"per-seed means"] + shift_notes(shift_day)
              + partial_note(partial, not include_partial))
    save(fig, out_dir, "lecar_weights")


def bar_panels(rows, value_column, fractions, xlabel, out_dir, name, fmt):
    fig, axes = plt.subplots(1, len(fractions), figsize=(WIDE, 2.2), sharey=True, squeeze=False)
    for ax, fraction in zip(axes[0], fractions):
        policies = [p for p in policies_in(rows)
                    if any(r["policy"] == p and r["cache_fraction"] == fraction for r in rows)]
        values = [float(next(r[value_column] for r in rows
                             if r["policy"] == p and r["cache_fraction"] == fraction))
                  for p in policies]
        y = list(range(len(policies)))
        ax.set_axisbelow(True)
        ax.barh(y, values, color=BAR, height=0.6)
        for yi, v in zip(y, values):
            ax.text(v, yi, " " + fmt(v), va="center", fontsize=6.5, color=INK_SECONDARY)
        ax.set_yticks(y, policies)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(size_label(fraction))
    fig.supxlabel(xlabel, fontsize=8, color=INK_SECONDARY, y=-0.02)
    save(fig, out_dir, name)


def plot_timing(results, out_dir):
    """Main figure: time per request relative to LRU in the same run (older results without
    the ratio column fall back to absolute ns/request)."""
    rows = read_csv(os.path.join(results, "timing.csv"))
    fractions = sorted({r["cache_fraction"] for r in rows}, key=float)
    if all(r.get("ratio_to_lru") for r in rows):
        bar_panels(rows, "ratio_to_lru", fractions,
                   "Time per request relative to LRU (median of runs; LRU = 1)",
                   out_dir, "timing", lambda v: f"{v:.2f}x")
    else:
        bar_panels(rows, "ns_per_request_median", fractions, "ns per request (median of runs)",
                   out_dir, "timing", lambda v: f"{v:,.0f}")


def plot_metadata(results, out_dir):
    rows = read_csv(os.path.join(results, "metadata.csv"))
    fractions = sorted({r["cache_fraction"] for r in rows}, key=float)
    # Primary space figure: measured bytes (absent if run_all used --no-python-bytes).
    if all(r["python_bytes_per_slot"] for r in rows):
        bar_panels(rows, "python_bytes_per_slot", fractions,
                   "Measured CPython bytes per cache slot (tracemalloc peak / cache size)",
                   out_dir, "memory_bytes_per_slot", lambda v: f"{v:,.0f}")
    # Secondary: structural count, independent of the implementation language.
    bar_panels(rows, "entries_per_slot", fractions, "Peak metadata entries per cache slot",
               out_dir, "metadata_entries_per_slot", lambda v: f"{v:.2f}")


def pareto_front(points):
    """Points not dominated by another point with smaller-or-equal x AND y."""
    front = set()
    for i, (xi, yi, _) in enumerate(points):
        if not any((xj <= xi and yj <= yi) and (xj < xi or yj < yi)
                   for j, (xj, yj, _) in enumerate(points) if j != i):
            front.add(i)
    return front


def plot_pareto(results, out_dir):
    miss = read_csv(os.path.join(results, "miss_ratio.csv"))
    timing_rows = read_csv(os.path.join(results, "timing.csv"))
    use_ratio = all(r.get("ratio_to_lru") for r in timing_rows)
    timing_column = "ratio_to_lru" if use_ratio else "ns_per_request_median"
    timing = {(r["policy"], r["cache_fraction"]): float(r[timing_column]) for r in timing_rows}
    meta = {(r["policy"], r["cache_fraction"]): r
            for r in read_csv(os.path.join(results, "metadata.csv"))}
    fractions = sorted({r["cache_fraction"] for r in miss}, key=float)
    use_bytes = all(r["python_bytes_per_slot"] for r in meta.values())
    overheads = [("Time per request relative to LRU" if use_ratio else "ns per request",
                  lambda k: timing[k])]
    overheads.append(("CPython bytes per cache slot", lambda k: float(meta[k]["python_bytes_per_slot"]))
                     if use_bytes else
                     ("Metadata entries per cache slot", lambda k: float(meta[k]["entries_per_slot"])))

    fig, axes = plt.subplots(2, len(fractions), figsize=(WIDE, 4.6), squeeze=False)
    to_label = []    # (axes, points, Belady text): labels are placed once the layout is final
    for row, (xlabel, overhead) in enumerate(overheads):
        for col, fraction in enumerate(fractions):
            ax = axes[row][col]
            ax.grid(axis="x", visible=True)
            points = []
            belady = None
            for r in miss:
                if r["cache_fraction"] != fraction:
                    continue
                y = float(r["miss_ratio_excl_day1_mean"])
                if r["policy"] == "Belady":
                    belady = y
                    continue
                points.append((overhead((r["policy"], fraction)), y, r["policy"]))
            front = pareto_front(points)
            for i, (x, y, policy) in enumerate(points):
                on_front = i in front
                ax.plot(x, y, marker=STYLE[policy][1], markersize=6, linestyle="none",
                        color=INK, markerfacecolor=INK if on_front else "white")
            belady_text = None
            if belady is not None:
                ax.axhline(belady, color=STYLE["Belady"][0], linestyle="--", linewidth=1)
                belady_text = ax.text(1.0, belady, "Belady (offline) ",
                                      transform=ax.get_yaxis_transform(), ha="right",
                                      va="bottom", fontsize=6.5, color=INK_SECONDARY)
            to_label.append((ax, points, belady_text))
            ax.set_xlabel(xlabel)
            if row == 0:
                ax.set_title(size_label(fraction))
            if col == 0:
                ax.set_ylabel("Miss ratio (excl. day 1)")
    fig.text(0.5, 1.0, "Filled = Pareto-optimal (no policy is both faster/smaller and lower "
             "miss ratio); hollow = dominated", ha="center", fontsize=7, color=INK_SECONDARY)
    fig.tight_layout()
    for ax, points, belady_text in to_label:
        place_point_labels(fig, ax, points, belady_text)
    save(fig, out_dir, "pareto")


# Label positions tried around a point, in order: (x offset, y offset in points, alignment).
LABEL_POSITIONS = [(4, 3, "left", "bottom"), (4, -3, "left", "top"),
                   (-4, 3, "right", "bottom"), (-4, -3, "right", "top")]


def place_point_labels(fig, ax, points, belady_text):
    """Name each point with a text label that overlaps no other label, no marker and the
    Belady label, and stays inside the axes. Tries LABEL_POSITIONS in order and keeps the
    first that fits (the first position if none does). Uses the real text sizes, so call
    it after the layout is final."""
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    taken = []    # boxes in display (pixel) coordinates that labels must avoid
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        taken.append(matplotlib.transforms.Bbox.from_extents(px - 5, py - 5, px + 5, py + 5))
    if belady_text is not None:
        taken.append(belady_text.get_window_extent(renderer))

    for x, y, policy in points:
        chosen = None
        for dx, dy, ha, va in LABEL_POSITIONS:
            label = ax.annotate(policy, (x, y), xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=6.5, color=INK_SECONDARY)
            box = label.get_window_extent(renderer)
            inside = (box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1
                      and box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1)
            if inside and not any(box.overlaps(other) for other in taken):
                chosen = box
                break
            label.remove()
        if chosen is None:    # nothing fits: fall back to the first position
            dx, dy, ha, va = LABEL_POSITIONS[0]
            label = ax.annotate(policy, (x, y), xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=6.5, color=INK_SECONDARY)
            chosen = label.get_window_extent(renderer)
        taken.append(chosen)


def plot_lecar_vs_lcs(results, out_dir):
    rows = read_csv(os.path.join(results, "lecar_vs_lcs.csv"))
    if not rows:
        return
    fractions = [r["cache_fraction"] for r in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.3))
    panels = [("miss_ratio_excl_day1_mean", "Miss ratio (excl. day 1)"),
              ("same_victim_fraction_mean", "Evictions where LRU and LFU\nnominate the same key")]
    for ax, (column, ylabel) in zip(axes, panels):
        for offset, (policy, tag) in zip((-width / 2, width / 2), (("LeCaR", "sy"), ("LeCaR-LCS", "lcs"))):
            values = [float(r[f"{tag}_{column}"]) for r in rows]
            # 2px surface gap between adjacent bars: draw slightly narrower than the slot
            ax.set_axisbelow(True)    # gridlines behind the bars
            ax.bar([xi + offset for xi in x], values, width=width - 0.04, color=STYLE[policy][0],
                   label=policy)
            for xi, v in zip(x, values):
                ax.text(xi + offset, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5,
                        color=INK_SECONDARY)
        ax.set_xticks(x, [size_label(f) for f in fractions])
        ax.set_ylabel(ylabel)
    figure_legend(fig, axes[0], ncol=2)
    save(fig, out_dir, "lecar_vs_lcs")


def main():
    parser = argparse.ArgumentParser(description="Plot the CSVs written by run_all.py.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--shift-day", type=int, default=None,
                        help="mark a known popularity shift (synthetic traces only), as a "
                             "0-based day index: 7 = start of the 8th day")
    parser.add_argument("--include-partial-days", action="store_true",
                        help="show days run_all marked as partial (e.g. a last day the trace "
                             "covers only part of); by default they are left out of the "
                             "per-day and weight plots, with a note saying so")
    args = parser.parse_args()
    out_dir = os.path.join(args.results, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_miss_ratio(args.results, out_dir)
    plot_per_day(args.results, out_dir, args.shift_day, args.include_partial_days)
    plot_weights(args.results, out_dir, args.shift_day, args.include_partial_days)
    plot_timing(args.results, out_dir)
    plot_metadata(args.results, out_dir)
    plot_pareto(args.results, out_dir)
    plot_lecar_vs_lcs(args.results, out_dir)
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
