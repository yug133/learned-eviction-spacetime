"""
day_winners.py - which online policy wins each day (from per_day.csv).

For every COMPLETE day (partial days, e.g. a last day the trace covers only part of, are
left out) and every cache size: each online policy's miss ratio (mean over seeds for
LeCaR) and its rank that day, 1 = lowest miss ratio. Belady is offline, so it is not
ranked. Ties (equal miss ratio) share the better rank; it does not happen in practice
with millions of requests per day, but the rule is fixed so counts are reproducible.

Writes, next to per_day.csv:
  day_ranks.csv  cache_fraction, day, policy, miss_ratio, rank
  day_wins.csv   cache_fraction, policy, days_best, days_best_excl_day1, complete_days,
                 mean_rank  (days_best counts the days the policy had rank 1; day 1 is a
                 cold start, so it is also counted without it)

run_all.py calls write_day_winners(); to redo it for existing results:
  python3 -m experiments.day_winners --results results/wiki_0.01/
"""

import argparse
import csv
import os
from collections import defaultdict

ONLINE = ["LRU", "LFU", "ARC", "LeCaR", "LeCaR-LCS"]


def day_ranks(per_day_rows):
    """Rows of per_day.csv -> list of dicts (cache_fraction, day, policy, miss_ratio, rank)."""
    ratios = defaultdict(list)    # (fraction, day, policy) -> miss ratios (one per seed)
    for r in per_day_rows:
        if r["policy"] not in ONLINE or r["partial"] == "True" or r["miss_ratio"] == "":
            continue
        ratios[(r["cache_fraction"], int(r["day"]), r["policy"])].append(float(r["miss_ratio"]))

    by_day = defaultdict(dict)    # (fraction, day) -> policy -> mean miss ratio
    for (fraction, day, policy), values in ratios.items():
        by_day[(fraction, day)][policy] = sum(values) / len(values)

    rows = []
    for (fraction, day) in sorted(by_day, key=lambda k: (float(k[0]), k[1])):
        policy_ratio = by_day[(fraction, day)]
        for policy, ratio in policy_ratio.items():
            rank = 1 + sum(1 for other in policy_ratio.values() if other < ratio)
            rows.append({"cache_fraction": fraction, "day": day, "policy": policy,
                         "miss_ratio": ratio, "rank": rank})
    return rows


def day_wins(rank_rows):
    """Count, per cache size and policy, the complete days it was best (rank 1)."""
    stats = defaultdict(lambda: {"days_best": 0, "days_best_excl_day1": 0,
                                 "complete_days": 0, "rank_sum": 0})
    for r in rank_rows:
        s = stats[(r["cache_fraction"], r["policy"])]
        s["complete_days"] += 1
        s["rank_sum"] += r["rank"]
        if r["rank"] == 1:
            s["days_best"] += 1
            if r["day"] != 1:
                s["days_best_excl_day1"] += 1
    rows = []
    for (fraction, policy), s in sorted(stats.items(),
                                        key=lambda kv: (float(kv[0][0]), ONLINE.index(kv[0][1]))):
        rows.append({"cache_fraction": fraction, "policy": policy,
                     "days_best": s["days_best"], "days_best_excl_day1": s["days_best_excl_day1"],
                     "complete_days": s["complete_days"],
                     "mean_rank": s["rank_sum"] / s["complete_days"]})
    return rows


def write_day_winners(results_dir):
    with open(os.path.join(results_dir, "per_day.csv")) as f:
        per_day_rows = list(csv.DictReader(f))
    ranks = day_ranks(per_day_rows)
    wins = day_wins(ranks)
    for name, rows, columns in [
        ("day_ranks.csv", ranks, ["cache_fraction", "day", "policy", "miss_ratio", "rank"]),
        ("day_wins.csv", wins, ["cache_fraction", "policy", "days_best", "days_best_excl_day1",
                                "complete_days", "mean_rank"]),
    ]:
        with open(os.path.join(results_dir, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return wins


def main():
    parser = argparse.ArgumentParser(description="Per-day winner tables from per_day.csv.")
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    for row in write_day_winners(args.results):
        print(f"{row['cache_fraction']:<6} {row['policy']:<10} best on {row['days_best']:>2} of "
              f"{row['complete_days']} complete days ({row['days_best_excl_day1']} excl. day 1), "
              f"mean rank {row['mean_rank']:.2f}")


if __name__ == "__main__":
    main()
