"""
compare_seeds.py - do the conclusions survive a second, independent sample?

Compares two results folders written by run_all.py (e.g. the seed-0 and seed-1 1% samples):
  rankings.csv       online policies ranked by miss ratio (excluding day 1) at each size
  lecar_arc_gap.csv  ARC minus LeCaR miss ratio: in percentage points and relative to ARC
                     (positive = LeCaR better)
  day_wins.csv       complete days each policy was best (from each folder's day_wins.csv)
  timing_ratios.csv  time per request relative to LRU, both samples, and how much it moved
The LRU drift check of each run (run_info.json) is printed too. Belady is offline, so it
is not ranked.

Run from the repo root:
  python3 -m experiments.compare_seeds --a results/wiki_0.01 --b results/wiki_0.01_seed1 \\
      --out results/seed_comparison
"""

import argparse
import csv
import json
import os

ONLINE = ["LRU", "LFU", "ARC", "LeCaR", "LeCaR-LCS"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def miss_ratios(folder):
    """(policy, fraction) -> mean miss ratio excluding day 1."""
    return {(r["policy"], r["cache_fraction"]): float(r["miss_ratio_excl_day1_mean"])
            for r in read_csv(os.path.join(folder, "miss_ratio.csv"))}


def ranking(ratios, fraction):
    """Online policies at one size, best (lowest miss ratio) first."""
    present = [p for p in ONLINE if (p, fraction) in ratios]
    return sorted(present, key=lambda p: ratios[(p, fraction)])


def compare(a, b, label_a, label_b):
    ratios = {label_a: miss_ratios(a), label_b: miss_ratios(b)}
    fractions = sorted({f for (_, f) in ratios[label_a]}, key=float)

    rankings = []
    for fraction in fractions:
        order_a = ranking(ratios[label_a], fraction)
        order_b = ranking(ratios[label_b], fraction)
        rankings.append({"cache_fraction": fraction,
                         f"ranking_{label_a}": " < ".join(order_a),
                         f"ranking_{label_b}": " < ".join(order_b),
                         "same_ranking": order_a == order_b})

    gaps = []
    for fraction in fractions:
        row = {"cache_fraction": fraction}
        for label in (label_a, label_b):
            arc = ratios[label][("ARC", fraction)]
            lecar = ratios[label][("LeCaR", fraction)]
            row[f"gap_pp_{label}"] = 100 * (arc - lecar)
            row[f"gap_relative_{label}"] = (arc - lecar) / arc
        gaps.append(row)

    wins = []
    wins_a = {(r["cache_fraction"], r["policy"]): r for r in read_csv(os.path.join(a, "day_wins.csv"))}
    wins_b = {(r["cache_fraction"], r["policy"]): r for r in read_csv(os.path.join(b, "day_wins.csv"))}
    for key in sorted(wins_a, key=lambda k: (float(k[0]), ONLINE.index(k[1]))):
        wins.append({"cache_fraction": key[0], "policy": key[1],
                     f"days_best_{label_a}": wins_a[key]["days_best"],
                     f"days_best_{label_b}": wins_b[key]["days_best"] if key in wins_b else "",
                     f"complete_days_{label_a}": wins_a[key]["complete_days"],
                     f"complete_days_{label_b}": wins_b[key]["complete_days"] if key in wins_b else ""})

    timing = []
    ratio_a = {(r["policy"], r["cache_fraction"]): float(r["ratio_to_lru"])
               for r in read_csv(os.path.join(a, "timing.csv"))}
    ratio_b = {(r["policy"], r["cache_fraction"]): float(r["ratio_to_lru"])
               for r in read_csv(os.path.join(b, "timing.csv"))}
    for (policy, fraction) in sorted(ratio_a, key=lambda k: (float(k[1]), k[0])):
        if policy == "LRU" or (policy, fraction) not in ratio_b:
            continue
        ra, rb = ratio_a[(policy, fraction)], ratio_b[(policy, fraction)]
        timing.append({"cache_fraction": fraction, "policy": policy,
                       f"ratio_to_lru_{label_a}": ra, f"ratio_to_lru_{label_b}": rb,
                       "difference": rb - ra, "relative_difference": (rb - ra) / ra})
    return rankings, gaps, wins, timing


def main():
    parser = argparse.ArgumentParser(description="Compare run_all results of two samples.")
    parser.add_argument("--a", required=True, help="first results folder (e.g. seed 0)")
    parser.add_argument("--b", required=True, help="second results folder (e.g. seed 1)")
    parser.add_argument("--labels", nargs=2, default=["seed0", "seed1"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    label_a, label_b = args.labels
    os.makedirs(args.out, exist_ok=True)

    rankings, gaps, wins, timing = compare(args.a, args.b, label_a, label_b)
    write_csv(os.path.join(args.out, "rankings.csv"), rankings, list(rankings[0]))
    write_csv(os.path.join(args.out, "lecar_arc_gap.csv"), gaps, list(gaps[0]))
    write_csv(os.path.join(args.out, "day_wins.csv"), wins, list(wins[0]))
    write_csv(os.path.join(args.out, "timing_ratios.csv"), timing, list(timing[0]))

    print("Rankings (best first, miss ratio excluding day 1):")
    for r in rankings:
        print(f"  {r['cache_fraction']:<6} {label_a}: {r[f'ranking_{label_a}']}")
        print(f"  {'':<6} {label_b}: {r[f'ranking_{label_b}']}   same: {r['same_ranking']}")
    print("\nLeCaR vs ARC (ARC - LeCaR; positive = LeCaR better):")
    for g in gaps:
        print(f"  {g['cache_fraction']:<6} " + "   ".join(
            f"{label}: {g[f'gap_pp_{label}']:+.2f} pp ({g[f'gap_relative_{label}']:+.1%})"
            for label in (label_a, label_b)))
    print("\nDays best (complete days):")
    for w in wins:
        print(f"  {w['cache_fraction']:<6} {w['policy']:<10} {label_a} {w[f'days_best_{label_a}']:>2}/"
              f"{w[f'complete_days_{label_a}']}   {label_b} {w[f'days_best_{label_b}']:>2}/"
              f"{w[f'complete_days_{label_b}']}")
    print("\nTime per request relative to LRU:")
    for t in timing:
        print(f"  {t['cache_fraction']:<6} {t['policy']:<10} {label_a} {t[f'ratio_to_lru_{label_a}']:.2f}x   "
              f"{label_b} {t[f'ratio_to_lru_{label_b}']:.2f}x   change {t['relative_difference']:+.1%}")
    for label, folder in ((label_a, args.a), (label_b, args.b)):
        with open(os.path.join(folder, "run_info.json")) as f:
            drift = json.load(f).get("lru_reference_ns_per_request", {}).get("end_over_start")
        if drift:
            print(f"LRU drift within the {label} run (end / start): "
                  + ", ".join(f"{k}: {v:.3f}" for k, v in drift.items()))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
