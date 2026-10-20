"""
diagnose_lecar.py - why does LeCaR stay at LRU's miss count on the LFU-favouring trace?

Trace (tests/util.py): 3 hot keys warmed up to count 2, then interleaved with one-time
keys; capacity 4. LRU misses 6001, LFU 3003 (checked against brute force in the tests).

For each variant and seeds 1-5 it reports misses, final w_lru and LeCaR's counters, plus:
  hot evictions   evictions of a hot key (0, 1, 2), split by route: LRU / LFU / both agreed
  lost regrets    misses on a key that had been in a history but was dropped because the
                  history was full (a regret the history was too small to see)
For the LCS preset, seed 1, it also prints the event log (regrets, hot-key evictions)
and w_lru over time.

Run from the repo root:  python3 -m experiments.diagnose_lecar
"""

import statistics

from src.policies.lecar import LeCaR, lecar_lcs
from tests.util import lfu_favouring_trace

CAPACITY = 4
HOT_KEYS = {0, 1, 2}
SEEDS = [1, 2, 3, 4, 5]
# name -> function(seed) building the policy. "SY" = the default (authors' code),
# "LCS" = the libCacheSim preset; the others change one setting at a time.
VARIANTS = {
    "SY (default)": lambda seed: LeCaR(CAPACITY, seed=seed),
    "SY, reset freq": lambda seed: LeCaR(CAPACITY, seed=seed, restore_freq=False),
    "SY, history C/2": lambda seed: LeCaR(CAPACITY, seed=seed, history_size=CAPACITY // 2),
    "SY, no clamp": lambda seed: LeCaR(CAPACITY, seed=seed, weight_clamp=0.0),
    "SY, Alg.1 same-victim": lambda seed: LeCaR(CAPACITY, seed=seed, same_victim_history=True),
    "LCS preset": lambda seed: lecar_lcs(CAPACITY, seed=seed),
    "LCS, restore freq": lambda seed: lecar_lcs(CAPACITY, seed=seed, restore_freq=True),
    "LCS, history C": lambda seed: lecar_lcs(CAPACITY, seed=seed, history_size=CAPACITY),
    "LCS, no floor": lambda seed: lecar_lcs(CAPACITY, seed=seed, weight_floor=0.0),
    "LCS, Alg.1 same-victim": lambda seed: lecar_lcs(CAPACITY, seed=seed, same_victim_history=True),
    "LCS, discount d = 1": lambda seed: lecar_lcs(CAPACITY, seed=seed, discount=1.0),
}
DETAIL_VARIANT = "LCS preset"    # the configuration whose failure is being explained


def run_instrumented(lecar, keys):
    """Run keys through lecar one by one, recording events by comparing state before and
    after each request (no instrumentation inside the policy itself)."""
    misses = 0
    events = []
    weights = []
    hot_evictions = {"LRU": 0, "LFU": 0, "both": 0}
    dropped = set()       # keys pushed out of a full history, not yet requested again
    lost_regrets = 0

    for key in keys:
        resident_before = set(lecar.lru.order)
        counts_before = dict(lecar.lfu.freq_of)
        history_before = set(lecar.h_lru) | set(lecar.h_lfu)
        regret = None
        if key in lecar.h_lru:
            regret = ("LRU", lecar.now + 1 - lecar.h_lru[key][0])
        elif key in lecar.h_lfu:
            regret = ("LFU", lecar.now + 1 - lecar.h_lfu[key][0])
        counters_before = (lecar.n_evict_lru, lecar.n_evict_lfu, lecar.n_same_victim)

        hit = lecar.request(key)
        now = lecar.now

        if not hit:
            misses += 1
            if key in dropped:
                lost_regrets += 1
                dropped.discard(key)
        if regret is not None:
            blamed, t = regret
            events.append((now, f"regret: {blamed} evicted {key} {t} requests ago, "
                                f"reward d^t = {lecar.discount ** t:.4f}, w_lru -> {lecar.w_lru:.4f}"))

        evicted = resident_before - set(lecar.lru.order)
        if evicted:
            (victim,) = evicted
            if lecar.n_evict_lru > counters_before[0]:
                route = "LRU"
            elif lecar.n_evict_lfu > counters_before[1]:
                route = "LFU"
            else:
                route = "both"
            if victim in HOT_KEYS:
                hot_evictions[route] += 1
                events.append((now, f"hot key {victim} (count {counts_before[victim]}) "
                                    f"evicted by {route}"))

        history_after = set(lecar.h_lru) | set(lecar.h_lfu)
        dropped |= history_before - history_after - {key}

        if now % 500 == 0:
            weights.append((now, lecar.w_lru))

    return misses, events, weights, hot_evictions, lost_regrets


def mean_std(values):
    return f"{statistics.mean(values):8.1f} ± {statistics.stdev(values):6.1f}"


def main():
    keys = lfu_favouring_trace()
    print(f"Trace: {len(keys)} requests, capacity {CAPACITY}. LRU 6001 misses, LFU 3003.\n")

    header = (f"{'variant':<24} {'misses (mean ± std)':>20} {'final w_lru':>17} "
              f"{'regret LRU':>10} {'regret LFU':>10} {'evict LRU':>9} {'evict LFU':>9} "
              f"{'same vic':>8} {'hist drop':>9} {'lost regr':>9} {'min scans':>9} {'hot ev L/F/both':>16}")
    print(header)
    print("-" * len(header))
    detail = None
    for name, make_policy in VARIANTS.items():
        rows = []
        for seed in SEEDS:
            lecar = make_policy(seed)
            result = run_instrumented(lecar, keys)
            rows.append((lecar, result))
            if name == DETAIL_VARIANT and seed == 1:
                detail = result
        misses = [r[1][0] for r in rows]
        w_lru = [r[0].w_lru for r in rows]

        def avg(attr):
            return statistics.mean(getattr(r[0], attr) for r in rows)

        hot = [sum(r[1][3][k] for r in rows) / len(rows) for k in ("LRU", "LFU", "both")]
        lost = statistics.mean(r[1][4] for r in rows)
        print(f"{name:<24} {mean_std(misses):>20} {statistics.mean(w_lru):8.3f} ± {statistics.stdev(w_lru):.3f} "
              f"{avg('n_hit_lru_history'):>10.1f} {avg('n_hit_lfu_history'):>10.1f} "
              f"{avg('n_evict_lru'):>9.1f} {avg('n_evict_lfu'):>9.1f} {avg('n_same_victim'):>8.1f} "
              f"{avg('n_history_dropped'):>9.1f} {lost:>9.1f} "
              f"{statistics.mean(r[0].lfu.n_min_scans for r in rows):>9.1f} "
              f"{hot[0]:>5.1f}/{hot[1]:.1f}/{hot[2]:.1f}")

    misses, events, weights, hot_evictions, lost_regrets = detail
    print(f"\n{DETAIL_VARIANT}, seed 1: first 25 events")
    for now, text in events[:25]:
        print(f"  t={now:>5}  {text}")
    print(f"  ... {len(events)} events in total")
    print(f"\n{DETAIL_VARIANT}, seed 1: w_lru over time")
    print("  " + "  ".join(f"{now}:{w:.3f}" for now, w in weights))


if __name__ == "__main__":
    main()
