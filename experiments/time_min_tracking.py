"""
time_min_tracking.py - LeCaR's LFU min-count lookup: lazy scan vs min-heap, timed.

Both methods give identical hits (tests/test_lecar.py); only the cost differs:
  scan: when the min count is unknown, min() over the distinct counts (O(F) per lookup)
  heap: a push on every bucket creation, lazy pops on lookup (amortised O(log F))

For each cache size: --runs rounds; in each round scan and heap run back to back, and
the order alternates between rounds (scan first, then heap first, ...) so slow drift in
machine load hits both. ns per request = median over the rounds, measured exactly as in
run_all.py (src/sim.py). Each pair of runs must give identical per-day misses and final
weights, or the script stops.

Work counters per request:
  scan: buckets looked at by min() scans
  heap: heap pushes + stale pops + entries written by rebuilds (each push/pop costs
        O(log F) comparisons, so these are not the same unit as bucket visits)

Run from the repo root (refuses to start while sample.py is running, so timings are clean):
  python3 -m experiments.time_min_tracking --trace data/wiki2018_0.001.tr --t0 0 \\
      --out results/wiki_0.001/min_tracking.csv
"""

import argparse
import csv
import os
import statistics
import sys

from src.policies.lecar import LeCaR
from src.sim import prepare, simulate
from src.trace import load_trace

MODES = ["scan", "heap"]


def other_process_running(name):
    """True if another process's command line contains `name` (Linux /proc)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if name.encode() in f.read():
                    return True
        except OSError:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Time LeCaR's LFU min lookup: scan vs heap.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True, help="CSV file for the results")
    parser.add_argument("--t0", type=int, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-concurrent", action="store_true",
                        help="time even if sample.py is running (timings will be noisy)")
    args = parser.parse_args()
    if other_process_running("sample.py") and not args.allow_concurrent:
        sys.exit("sample.py is still running; timings would be distorted. Try again later.")

    trace = load_trace(args.trace, t0=args.t0)
    prep = prepare(trace, t0=args.t0 if args.t0 is not None else int(trace.times[0]))
    del trace
    n = prep.num_requests
    print(f"{args.trace}: {n:,} requests, {prep.num_unique:,} unique objects", flush=True)

    rows = []
    for fraction in args.fractions:
        capacity = max(1, round(fraction * prep.num_unique))    # same rule as run_all.py
        for run in range(1, args.runs + 1):
            order = MODES if run % 2 == 1 else MODES[::-1]
            results = {}
            for mode in order:
                policy = LeCaR(capacity, seed=args.seed, lfu_min_tracking=mode)
                results[mode] = simulate(policy, prep)
                s = results[mode].stats
                rows.append({
                    "cache_fraction": fraction, "capacity": capacity, "mode": mode, "run": run,
                    "ns_per_request": results[mode].loop_ns / n,
                    "misses": results[mode].misses,
                    "min_lookups": s["min_scans"],
                    "buckets_scanned_per_request": s["buckets_scanned"] / n,
                    "heap_ops_per_request":
                        (s["heap_pushes"] + s["heap_pops"] + s["heap_rebuild_items"]) / n,
                })
                print(f"  size {fraction:<6g} ({capacity:>6,} slots) run {run} {mode}: "
                      f"{results[mode].loop_ns / n:7.0f} ns/request", flush=True)
            scan, heap = results["scan"], results["heap"]
            if (scan.misses_per_day != heap.misses_per_day
                    or scan.stats["final_w_lru"] != heap.stats["final_w_lru"]):
                sys.exit(f"scan and heap disagree at size {fraction}, run {run}: this is a bug")

    columns = ["cache_fraction", "capacity", "mode", "run", "ns_per_request", "misses",
               "min_lookups", "buckets_scanned_per_request", "heap_ops_per_request"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'size':<7} {'mode':<5} {'median ns/req':>13} {'runs':>22} "
          f"{'min lookups':>11} {'buckets/req':>11} {'heap ops/req':>12}")
    for fraction in args.fractions:
        for mode in MODES:
            group = [r for r in rows if r["cache_fraction"] == fraction and r["mode"] == mode]
            times = [r["ns_per_request"] for r in group]
            first = group[0]
            print(f"{fraction:<7g} {mode:<5} {statistics.median(times):13.0f} "
                  f"{' '.join(f'{t:6.0f}' for t in times):>22} {first['min_lookups']:>11,} "
                  f"{first['buckets_scanned_per_request']:>11.2f} "
                  f"{first['heap_ops_per_request']:>12.3f}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
