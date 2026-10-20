"""
make_synthetic.py - a synthetic trace in sample.py's output format ("time id size"),
for testing sim.py / run_all.py / plot.py before the real trace is available.

Model (all numbers are command-line options):
  - `objects` objects; request popularity follows a Zipf law over popularity ranks:
    P(rank r) proportional to 1 / r**alpha, r = 1..objects.
  - Request times are uniform over `days` days, sorted.
  - Drift: from `shift_day` on, the ranks are reassigned to objects by a NEW random
    permutation, so the popular set changes abruptly. Before: rank r -> perm_before[r];
    after: rank r -> perm_after[r].
  - Ids are random 9-digit integers (like real object ids, not 0..U-1).
  - Size is random and ignored by the simulator (unit-size objects).

This is test data. Nothing measured on it says anything about the Wikipedia trace.

Run from the repo root:
  python3 -m experiments.make_synthetic --out data/synthetic.tr
"""

import argparse

import numpy as np

T0 = 1_500_000_000     # arbitrary start timestamp (seconds)
SECONDS_PER_DAY = 86400


def make_trace(requests, objects, alpha, days, shift_day, seed):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, days * SECONDS_PER_DAY, size=requests)) + T0

    weights = 1.0 / np.arange(1, objects + 1) ** alpha
    ranks = rng.choice(objects, size=requests, p=weights / weights.sum())

    ids = rng.choice(np.arange(100_000_000, 1_000_000_000), size=objects, replace=False)
    perm_before = rng.permutation(objects)
    perm_after = rng.permutation(objects)
    after = times >= T0 + shift_day * SECONDS_PER_DAY
    obj = np.where(after, perm_after[ranks], perm_before[ranks])

    sizes = rng.integers(100, 100_000, size=objects)
    return times, ids[obj], sizes[obj]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="data/synthetic.tr")
    parser.add_argument("--requests", type=int, default=1_000_000)
    parser.add_argument("--objects", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--shift-day", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    times, ids, sizes = make_trace(args.requests, args.objects, args.alpha,
                                   args.days, args.shift_day, args.seed)
    np.savetxt(args.out, np.column_stack([times, ids, sizes]), fmt="%d", delimiter=" ")
    print(f"wrote {args.out}: {len(times):,} requests, {len(np.unique(ids)):,} unique objects, "
          f"{args.days} days, popularity reshuffled at day {args.shift_day}")


if __name__ == "__main__":
    main()
