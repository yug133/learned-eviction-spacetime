"""
run_all.py - every experiment of CLAUDE.md section 6 on one sampled trace.

Run from the repo root:
  python3 -m experiments.run_all --trace data/wiki2018_0.001.tr --out results/wiki_0.001/

Policies: LRU, LFU, ARC, Belady, LeCaR (authors' config), LeCaR-LCS (libCacheSim config).
Cache sizes: --fractions of the unique objects in the trace (default 0.1%, 1%, 10%).
Runs: randomised policies once per seed (default 1-5); deterministic ones --timing-runs
times (default 3), which must give identical misses.

Outputs in --out (CSV, so plots and the paper's tables are made from the same files):
  runs.csv          one row per policy run
  miss_ratio.csv    mean ± std miss ratio per policy and size, with and without day 1
  per_day.csv       miss ratio per day, the day's most requested object and its share of the
                    day's requests (flash crowds), each day's coverage and a "partial" flag (a day
                    the trace covers < 99% of, e.g. the last day if the trace stops mid-day).
                    Overall miss ratios (miss_ratio.csv) still count every request.
  day_ranks.csv     each online policy's miss ratio and rank on each complete day
  day_wins.csv      per policy and size: complete days it was best (experiments/day_winners.py)
  weights.csv       LeCaR w_lru over time
  timing.csv        ratio_to_lru = a policy's median ns/request / LRU's median ns/request in
                    the same run and size (the paper's main time figure: comparable across
                    runs and machines); absolute ns/request (each timed run, median) after it
  metadata.csv      measured CPython bytes per cache slot (primary space metric), plus the
                    peak count of metadata entries (structural, implementation-independent)
  lecar_stats.csv   LeCaR counters: same-victim fraction, regrets, O(F) min scans
  lecar_vs_lcs.csv  LeCaR vs LeCaR-LCS per cache size
  competitive.csv   observed ratio to OPT (misses / Belady's misses) next to each policy's
                    worst-case (competitive-ratio) bound, in separate columns
  run_info.json     configuration, trace size, machine, peak memory, durations
  sanity.txt        automatic checks; exit status 1 if a hard check fails

ns per request = median over the first --timing-runs runs (LeCaR: seeds 1-3), measured
around the request loop only (see src/sim.py). Drift check: LRU is also timed once per
cache size before all other runs and once after them; both, and their ratio, go into
run_info.json (lru_reference), and sanity.txt warns if they differ by more than 5%. Measured CPython bytes, the primary space
metric, come from one extra run per policy and size under tracemalloc (see src/sim.py for
what it counts). --no-python-bytes skips those runs and leaves only the entry counts.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from src.policies.arc import ARC
from src.policies.belady import Belady, compute_next_use
from src.policies.lecar import LeCaR, lecar_lcs
from src.policies.lfu import LFU
from src.policies.lru import LRU
from experiments.day_winners import write_day_winners
from src.sim import prepare, simulate
from src.trace import load_trace

POLICIES = ["LRU", "LFU", "ARC", "Belady", "LeCaR", "LeCaR-LCS"]
RANDOMISED = {"LeCaR", "LeCaR-LCS"}

# Worst-case bound (competitive ratio, k = cache size) and where it comes from. This is a
# THEORETICAL guarantee, kept apart from the OBSERVED ratio to OPT on our trace.
WORST_CASE_BOUND = {
    "LRU": ("k", "k-competitive, Sleator and Tarjan 1985"),
    "LFU": ("needs citation", "no tight bound stated"),
    "ARC": ("needs citation", "no tight bound stated"),
    "Belady": ("1 (is OPT)", "1 (is OPT), Belady 1966"),
    "LeCaR": ("no proven bound", "no proven bound (learned policy without a fallback)"),
    "LeCaR-LCS": ("no proven bound", "no proven bound (learned policy without a fallback)"),
}


def make_policy(name, capacity, seed, prep, next_use):
    if name == "LRU":
        return LRU(capacity)
    if name == "LFU":
        return LFU(capacity)
    if name == "ARC":
        return ARC(capacity)
    if name == "Belady":
        return Belady(capacity, prep.keys, next_use)
    if name == "LeCaR":
        return LeCaR(capacity, seed=seed)
    if name == "LeCaR-LCS":
        return lecar_lcs(capacity, seed=seed)
    raise ValueError(f"unknown policy {name}")


def lru_reference(prep, capacities):
    """ns/request of one LRU run per cache size: a fixed workload to detect machine drift."""
    return {str(f): simulate(LRU(c), prep).loop_ns / prep.num_requests
            for f, c in capacities.items()}


def python_peak_bytes(make, keys):
    """Peak bytes allocated by CPython while building a policy and running the whole trace.
    The key list (and Belady's next_use array) already exist, so they are not counted."""
    tracemalloc.start()
    policy = make()
    for key in keys:
        policy.request(key)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del policy
    return peak


def memory_mb():
    """Current (VmRSS) and peak (VmHWM) resident memory of this process, in MB (Linux)."""
    values = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, amount, _unit = line.split()
                    values[name.rstrip(":")] = int(amount) / 1024
    except OSError:
        pass
    return values


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def mean_std(values):
    if not values:
        return "", ""
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def main():
    parser = argparse.ArgumentParser(description="Run every experiment on one trace.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    parser.add_argument("--t0", type=int, default=None,
                        help="start of day 0 (default: first timestamp of the trace); pass "
                             "the FULL trace's first timestamp so samples share day boundaries")
    parser.add_argument("--no-python-bytes", action="store_true",
                        help="skip the extra tracemalloc run per policy and size (this drops "
                             "the primary space metric; only entry counts remain)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    phases = {}
    memory = {"start": memory_mb()}

    # --- load and prepare the trace ---
    t = time.time()
    trace = load_trace(args.trace, t0=args.t0)
    t0 = args.t0 if args.t0 is not None else int(trace.times[0])
    phases["load_s"] = time.time() - t
    memory["after_load"] = memory_mb()

    t = time.time()
    prep = prepare(trace, t0=t0)
    del trace    # the numpy arrays are no longer needed
    phases["prepare_s"] = time.time() - t
    memory["after_prepare"] = memory_mb()

    next_use = None
    if "Belady" in args.policies:
        t = time.time()
        next_use = compute_next_use(prep.keys)
        phases["belady_next_use_s"] = time.time() - t
        memory["after_next_use"] = memory_mb()

    n = prep.num_requests
    day0 = prep.requests_per_day[0]
    print(f"{args.trace}: {n:,} requests, {prep.num_unique:,} unique objects "
          f"({prep.one_hit_wonder_fraction:.1%} requested once), {prep.num_days} days", flush=True)

    for d in prep.partial_days:
        print(f"  NOTE: day {d + 1} is partial ({prep.day_coverage[d]:.0%} of a day covered); "
              f"marked in per_day.csv")
    capacities = {f: max(1, round(f * prep.num_unique)) for f in args.fractions}
    for f, c in capacities.items():
        if c < 100:
            print(f"  WARNING: cache size {f:g} of uniques = {c} slots (< 100), flag in paper")

    # --- run everything ---
    reference = {"start": lru_reference(prep, capacities)}
    print("  LRU reference at start (ns/request): "
          + ", ".join(f"{f}: {v:.0f}" for f, v in reference["start"].items()), flush=True)
    runs = []            # one dict per run, with the RunResult attached under "result"
    python_bytes = {}    # (policy, fraction) -> peak CPython bytes
    t = time.time()
    for fraction in args.fractions:
        capacity = capacities[fraction]
        for name in args.policies:
            run_ids = args.seeds if name in RANDOMISED else range(1, args.timing_runs + 1)
            for run_id in run_ids:
                seed = run_id if name in RANDOMISED else None
                result = simulate(make_policy(name, capacity, seed, prep, next_use), prep)
                misses_excl = result.misses - result.misses_per_day[0]
                runs.append({
                    "policy": name, "cache_fraction": fraction, "capacity": capacity,
                    "run": run_id, "seed": seed if seed is not None else "",
                    "requests": n, "misses": result.misses, "miss_ratio": result.misses / n,
                    "misses_excl_day1": misses_excl,
                    "miss_ratio_excl_day1": misses_excl / (n - day0) if n > day0 else "",
                    "ns_per_request": result.loop_ns / n,
                    "peak_metadata_entries": result.peak_metadata_entries,
                    "result": result,
                })
                print(f"  {name:<10} size {fraction:<6g} ({capacity:>7,} slots) run {run_id}: "
                      f"miss ratio {result.misses / n:.4f}, {result.loop_ns / n:7.0f} ns/request",
                      flush=True)
    phases["runs_s"] = time.time() - t
    reference["end"] = lru_reference(prep, capacities)
    reference["end_over_start"] = {f: reference["end"][f] / reference["start"][f]
                                   for f in reference["start"]}
    print("  LRU reference at end (ns/request): "
          + ", ".join(f"{f}: {v:.0f} ({reference['end_over_start'][f]:.3f}x start)"
                      for f, v in reference["end"].items()), flush=True)
    # Checkpoint BEFORE the tracemalloc runs: tracemalloc's own bookkeeping uses a lot of
    # memory, so "after_runs" is the peak a run with --no-python-bytes would have.
    memory["after_runs"] = memory_mb()

    if not args.no_python_bytes:
        t = time.time()
        for fraction in args.fractions:
            capacity = capacities[fraction]
            for name in args.policies:
                python_bytes[(name, fraction)] = python_peak_bytes(
                    lambda: make_policy(name, capacity, 1, prep, next_use), prep.keys)
        phases["python_bytes_s"] = time.time() - t
    memory["end"] = memory_mb()

    write_outputs(args, prep, capacities, runs, python_bytes, next_use, t0)
    write_day_winners(args.out)
    failed = write_sanity(args, prep, capacities, runs, reference)

    info = {
        "trace": args.trace, "t0": t0, "requests": n, "unique_objects": prep.num_unique,
        "one_hit_wonders": prep.one_hit_wonders,
        "one_hit_wonder_fraction": prep.one_hit_wonder_fraction,
        "days": prep.num_days, "requests_per_day": prep.requests_per_day,
        "day_coverage": prep.day_coverage,
        "partial_days": [d + 1 for d in prep.partial_days],    # 1-based, as in per_day.csv
        "day_top_object": prep.day_top_object,
        "day_top_share": prep.day_top_share,
        "fractions": args.fractions, "capacities": {str(f): c for f, c in capacities.items()},
        "seeds": args.seeds, "timing_runs": args.timing_runs, "policies": args.policies,
        "space_metric": ("entry counts only (--no-python-bytes)" if args.no_python_bytes
                         else "python_bytes_per_slot = tracemalloc peak / capacity"),
        "weight_samples_per_run": len(prep.boundaries),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "cpu": cpu_model(),
        "lru_reference_ns_per_request": reference,
        "memory_mb": memory, "phases_s": phases, "total_s": time.time() - started,
    }
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    peak = memory["after_runs"].get("VmHWM")
    print(f"done in {info['total_s']:.0f} s; peak memory without tracemalloc runs "
          f"{peak:.0f} MB" if peak else "done")
    sys.exit(1 if failed else 0)


def write_outputs(args, prep, capacities, runs, python_bytes, next_use, t0):
    out = args.out
    n = prep.num_requests

    write_csv(os.path.join(out, "runs.csv"), runs,
              ["policy", "cache_fraction", "capacity", "run", "seed", "requests", "misses",
               "miss_ratio", "misses_excl_day1", "miss_ratio_excl_day1", "ns_per_request",
               "peak_metadata_entries"])

    groups = {}    # (policy, fraction) -> list of runs
    for r in runs:
        groups.setdefault((r["policy"], r["cache_fraction"]), []).append(r)

    # miss ratio summary
    rows = []
    for (name, fraction), group in groups.items():
        mr_mean, mr_std = mean_std([r["miss_ratio"] for r in group])
        ex_mean, ex_std = mean_std([r["miss_ratio_excl_day1"] for r in group])
        rows.append({"policy": name, "cache_fraction": fraction, "capacity": capacities[fraction],
                     "runs": len(group), "miss_ratio_mean": mr_mean, "miss_ratio_std": mr_std,
                     "miss_ratio_excl_day1_mean": ex_mean, "miss_ratio_excl_day1_std": ex_std})
    write_csv(os.path.join(out, "miss_ratio.csv"), rows,
              ["policy", "cache_fraction", "capacity", "runs", "miss_ratio_mean", "miss_ratio_std",
               "miss_ratio_excl_day1_mean", "miss_ratio_excl_day1_std"])

    # per-day miss ratio (deterministic policies: run 1 only, all runs are identical)
    rows = []
    for r in runs:
        if r["policy"] not in RANDOMISED and r["run"] != 1:
            continue
        for day, (misses, requests) in enumerate(zip(r["result"].misses_per_day,
                                                     prep.requests_per_day)):
            rows.append({"policy": r["policy"], "cache_fraction": r["cache_fraction"],
                         "seed": r["seed"], "day": day + 1, "requests": requests,
                         "misses": misses, "miss_ratio": misses / requests if requests else "",
                         "top_object_id": prep.day_top_object[day],
                         "top_object_share": prep.day_top_share[day],
                         "day_coverage": prep.day_coverage[day],
                         "partial": day in prep.partial_days})
    write_csv(os.path.join(out, "per_day.csv"), rows,
              ["policy", "cache_fraction", "seed", "day", "requests", "misses", "miss_ratio",
               "top_object_id", "top_object_share", "day_coverage", "partial"])

    # LeCaR weights over time
    rows = []
    for r in runs:
        for request_index, days, w_lru in r["result"].weight_log:
            rows.append({"policy": r["policy"], "cache_fraction": r["cache_fraction"],
                         "seed": r["seed"], "request_index": request_index,
                         "time_days": days, "w_lru": w_lru})
    write_csv(os.path.join(out, "weights.csv"), rows,
              ["policy", "cache_fraction", "seed", "request_index", "time_days", "w_lru"])

    # timing: median of the first --timing-runs runs
    rows = []
    timing = {}
    for (name, fraction), group in groups.items():
        times = [r["ns_per_request"] for r in group[:args.timing_runs]]
        timing[(name, fraction)] = statistics.median(times)
        row = {"policy": name, "cache_fraction": fraction, "capacity": capacities[fraction],
               "ns_per_request_median": timing[(name, fraction)]}
        for i, value in enumerate(times, start=1):
            row[f"ns_per_request_run{i}"] = value
        rows.append(row)
    for row in rows:    # ratio to LRU's median at the same size, in this same run
        lru = timing.get(("LRU", row["cache_fraction"]))
        row["ratio_to_lru"] = row["ns_per_request_median"] / lru if lru else ""
    write_csv(os.path.join(out, "timing.csv"), rows,
              ["policy", "cache_fraction", "capacity", "ratio_to_lru", "ns_per_request_median"]
              + [f"ns_per_request_run{i}" for i in range(1, args.timing_runs + 1)])

    # metadata / memory
    rows = []
    for (name, fraction), group in groups.items():
        capacity = capacities[fraction]
        entries = max(r["peak_metadata_entries"] for r in group)
        measured = python_bytes.get((name, fraction), "")
        rows.append({
            "policy": name, "cache_fraction": fraction, "capacity": capacity,
            "python_bytes_per_slot": measured / capacity if measured != "" else "",
            "python_peak_bytes": measured,
            "peak_metadata_entries": entries, "entries_per_slot": entries / capacity,
            "preprocessing_bytes": next_use.nbytes if name == "Belady" else 0,
        })
    write_csv(os.path.join(out, "metadata.csv"), rows,
              ["policy", "cache_fraction", "capacity", "python_bytes_per_slot", "python_peak_bytes",
               "peak_metadata_entries", "entries_per_slot", "preprocessing_bytes"])

    # LeCaR counters
    rows = []
    stat_columns = ["evictions", "same_victim", "same_victim_fraction", "evict_lru", "evict_lfu",
                    "regret_lru", "regret_lfu", "history_dropped", "lfu_min_tracking",
                    "min_scans", "buckets_scanned", "heap_pushes", "heap_pops",
                    "heap_rebuild_items", "final_w_lru"]
    for r in runs:
        if r["policy"] in RANDOMISED:
            rows.append({"policy": r["policy"], "cache_fraction": r["cache_fraction"],
                         "seed": r["seed"], **r["result"].stats})
    write_csv(os.path.join(out, "lecar_stats.csv"), rows,
              ["policy", "cache_fraction", "seed"] + stat_columns)

    # LeCaR vs LeCaR-LCS
    rows = []
    if {"LeCaR", "LeCaR-LCS"} <= set(args.policies):
        for fraction in args.fractions:
            row = {"cache_fraction": fraction, "capacity": capacities[fraction]}
            for name, tag in (("LeCaR", "sy"), ("LeCaR-LCS", "lcs")):
                group = groups[(name, fraction)]
                row[f"{tag}_miss_ratio_mean"], row[f"{tag}_miss_ratio_std"] = mean_std(
                    [r["miss_ratio"] for r in group])
                row[f"{tag}_miss_ratio_excl_day1_mean"], _ = mean_std(
                    [r["miss_ratio_excl_day1"] for r in group])
                row[f"{tag}_same_victim_fraction_mean"], _ = mean_std(
                    [r["result"].stats["same_victim_fraction"] for r in group])
                row[f"{tag}_final_w_lru_mean"], _ = mean_std(
                    [r["result"].stats["final_w_lru"] for r in group])
                row[f"{tag}_ns_per_request_median"] = timing[(name, fraction)]
            row["lcs_minus_sy_miss_ratio"] = row["lcs_miss_ratio_mean"] - row["sy_miss_ratio_mean"]
            rows.append(row)
    columns = ["cache_fraction", "capacity"]
    for tag in ("sy", "lcs"):
        columns += [f"{tag}_miss_ratio_mean", f"{tag}_miss_ratio_std",
                    f"{tag}_miss_ratio_excl_day1_mean", f"{tag}_same_victim_fraction_mean",
                    f"{tag}_final_w_lru_mean", f"{tag}_ns_per_request_median"]
    write_csv(os.path.join(out, "lecar_vs_lcs.csv"), rows, columns + ["lcs_minus_sy_miss_ratio"])

    # observed ratio to OPT (misses / Belady misses) next to the worst-case bound
    rows = []
    if "Belady" in args.policies:
        for (name, fraction), group in groups.items():
            belady = groups[("Belady", fraction)][0]
            ratio, _ = mean_std([r["misses"] / belady["misses"] for r in group])
            ratio_excl, _ = mean_std([r["misses_excl_day1"] / belady["misses_excl_day1"]
                                      for r in group])
            bound, source = WORST_CASE_BOUND[name]
            rows.append({"policy": name, "cache_fraction": fraction,
                         "capacity": capacities[fraction],
                         "observed_ratio_to_opt": ratio,
                         "observed_ratio_to_opt_excl_day1": ratio_excl,
                         "worst_case_bound": bound, "bound_source": source})
    write_csv(os.path.join(out, "competitive.csv"), rows,
              ["policy", "cache_fraction", "capacity", "observed_ratio_to_opt",
               "observed_ratio_to_opt_excl_day1", "worst_case_bound", "bound_source"])


def write_sanity(args, prep, capacities, runs, reference):
    """Automatic checks. Returns True if a hard check failed."""
    lines = []
    failed = False

    def check(ok, text, hard=True):
        nonlocal failed
        lines.append(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {text}")
        if hard and not ok:
            failed = True

    check(sum(prep.requests_per_day) == prep.num_requests, "requests per day add up to the total")
    for fraction, ratio in reference["end_over_start"].items():
        check(abs(ratio - 1) <= 0.05,
              f"LRU reference timing at size {fraction} drifted {ratio - 1:+.1%} from start to end "
              f"(timings within this run are comparable if within 5%)", hard=False)
    for fraction, capacity in capacities.items():
        check(capacity >= 100, f"cache size {fraction:g}: {capacity} slots >= 100", hard=False)

    by_key = {}
    for r in runs:
        by_key.setdefault((r["policy"], r["cache_fraction"]), []).append(r)

    for (name, fraction), group in by_key.items():
        if name not in RANDOMISED:
            check(len({r["misses"] for r in group}) == 1,
                  f"{name} size {fraction:g}: repeated runs give identical misses")
        if ("Belady", fraction) in by_key and name != "Belady":
            belady = by_key[("Belady", fraction)][0]["misses"]
            worst = min(r["misses"] for r in group)
            check(belady <= worst, f"Belady ({belady}) <= {name} ({worst}) at size {fraction:g}")
        for r in group:
            if name in RANDOMISED:
                s = r["result"].stats
                check(s["evict_lru"] + s["evict_lfu"] + s["same_victim"] == s["evictions"],
                      f"{name} size {fraction:g} seed {r['seed']}: eviction counters add up")

    # LRU and Belady are stack algorithms: a bigger cache can never miss more.
    fractions = sorted(capacities)
    for name in ("LRU", "Belady"):
        if all((name, f) in by_key for f in fractions):
            misses = [by_key[(name, f)][0]["misses"] for f in fractions]
            check(misses == sorted(misses, reverse=True),
                  f"{name} misses do not increase with cache size: {misses}")

    text = "\n".join(lines)
    with open(os.path.join(args.out, "sanity.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return failed


if __name__ == "__main__":
    main()
