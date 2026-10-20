"""
sim.py - run one policy over a trace and collect the metrics of CLAUDE.md section 6.

Two steps:
  prepare(trace)          once per trace: turn the ids into compact Python ints and cut the
                          trace into chunks (every day boundary + every `sample_every`
                          requests).
  simulate(policy, prep)  once per policy run: feed every request to policy.request().

Timing (G1): time.perf_counter_ns() is started and stopped around each chunk's loop
    for key in chunk: if request(key): hits += 1
and the chunk times are summed. Between chunks the clock is stopped while per-day misses,
LeCaR's w_lru and the metadata size are recorded, so that bookkeeping is not timed. The
hit counter and the loop itself are timed; they cost the same for every policy, so
policies must be compared RELATIVE to each other (also: this is CPython, so the absolute
numbers include interpreter overhead). The garbage collector is paused during a run
(as timeit does); the policies create no reference cycles, so nothing leaks.

Memory: ids are remapped to 0..U-1 and every request refers to ONE shared int object per
object id, so the key list costs 8 bytes per request plus ~32 bytes per unique object
(instead of ~40 bytes per request for a plain ids.tolist()).

Space metric (G1). The primary figure is MEASURED: the peak bytes CPython allocates
(tracemalloc) while a policy is built and runs the whole trace, divided by the cache size
(run_all.py, python_peak_bytes). It counts every structure the policy keeps (dict and
OrderedDict tables, list nodes, tuples, int and float objects it creates) plus transient
allocations, and excludes the trace itself (and Belady's next_use array, which is reported
separately as preprocessing). It is CPython-specific: object headers and hash-table slack
make it larger than a C implementation would be, so compare policies RELATIVE to each other.
metadata_entries() (entries across all of a policy's structures) is reported next to it as
an implementation-independent structural count. No fixed bytes-per-entry model is used:
the size of an entry depends on the implementation (field layout, pointer width, allocator,
hash-table load factor), so any single constant would be a guess.
"""

import gc
import time
from dataclasses import dataclass, field

import numpy as np

SECONDS_PER_DAY = 86400

# A day is "partial" if the trace covers less than this fraction of its 86,400 s (the
# trace starts after the day begins or ends before it ends). 0.99 allows ~14 min of slack.
PARTIAL_DAY_COVERAGE = 0.99


@dataclass
class PreparedTrace:
    keys: list                 # one small Python int per request, ids remapped to 0..U-1
    num_unique: int
    one_hit_wonders: int       # objects requested exactly once
    boundaries: list           # chunk start indices, then len(keys) as the final end
    chunk_day: list            # day index of each chunk (a chunk never spans two days)
    chunk_end_days: list       # time at the end of each chunk, in days since t0 (float)
    requests_per_day: list     # requests in day 0, 1, ..., num_days - 1
    day_coverage: list         # fraction of each day's 86,400 s inside [first, last] timestamp
    day_top_object: list       # original id of each day's most requested object (None if empty)
    day_top_share: list        # its share of that day's requests (0.0 if the day is empty)

    @property
    def num_requests(self):
        return len(self.keys)

    @property
    def num_days(self):
        return len(self.requests_per_day)

    @property
    def partial_days(self):
        """0-based indices of days the trace covers only partly (e.g. the last day, if the
        trace stops in the middle of it)."""
        return [d for d, c in enumerate(self.day_coverage) if c < PARTIAL_DAY_COVERAGE]

    @property
    def one_hit_wonder_fraction(self):
        """Objects requested exactly once / unique objects."""
        return self.one_hit_wonders / self.num_unique if self.num_unique else 0.0


@dataclass
class RunResult:
    policy: str
    capacity: int
    misses: int
    misses_per_day: list
    loop_ns: int                         # total time inside the request loops
    peak_metadata_entries: int           # largest metadata_entries() seen at a chunk end
    weight_log: list = field(default_factory=list)    # (request index, time in days, w_lru)
    stats: dict = field(default_factory=dict)          # policy-specific counters


def dense_keys(ids, chunk_size=1_000_000):
    """Map ids to 0..U-1. Returns (list of shared Python ints, U, one-hit wonders).

    Building the list chunk by chunk avoids one huge temporary list of new int objects.
    """
    _, inverse = np.unique(ids, return_inverse=True)
    num_unique = int(inverse.max()) + 1 if len(inverse) else 0
    one_hit_wonders = int(np.count_nonzero(np.bincount(inverse, minlength=num_unique) == 1))
    pool = list(range(num_unique))    # one int object per unique id, shared by all requests
    keys = []
    for start in range(0, len(inverse), chunk_size):
        keys.extend(map(pool.__getitem__, inverse[start:start + chunk_size].tolist()))
    return keys, num_unique, one_hit_wonders


def prepare(trace, sample_every=None, t0=None):
    """Build the PreparedTrace for a src.trace.Trace. t0 must match the one used for trace.days."""
    n = len(trace)
    keys, num_unique, one_hit_wonders = dense_keys(trace.ids)
    if sample_every is None:
        sample_every = max(1, n // 500)    # about 500 points for the weight-over-time plot
    if t0 is None:
        t0 = int(trace.times[0]) if n else 0

    day_starts = (np.flatnonzero(np.diff(trace.days)) + 1).tolist()
    boundaries = sorted(set([0] + day_starts + list(range(0, n, sample_every)))) + [n]
    chunk_day = [int(trace.days[b]) for b in boundaries[:-1]]
    chunk_end_days = [(int(trace.times[end - 1]) - t0) / SECONDS_PER_DAY for end in boundaries[1:]]
    requests_per_day = np.bincount(trace.days, minlength=trace.num_days()).tolist()
    day_coverage = coverage_per_day(int(trace.times[0]), int(trace.times[-1]), t0,
                                    len(requests_per_day))
    day_top_object, day_top_share = top_object_per_day(trace.ids, requests_per_day)
    return PreparedTrace(keys, num_unique, one_hit_wonders, boundaries, chunk_day,
                         chunk_end_days, requests_per_day, day_coverage,
                         day_top_object, day_top_share)


def top_object_per_day(ids, requests_per_day):
    """Each day's most requested object and its share of that day's requests.
    The trace is in time order, so day d is the contiguous slice after days 0..d-1."""
    top_objects, shares = [], []
    start = 0
    for requests in requests_per_day:
        if requests == 0:
            top_objects.append(None)
            shares.append(0.0)
            continue
        day_ids, counts = np.unique(ids[start:start + requests], return_counts=True)
        best = int(np.argmax(counts))
        top_objects.append(int(day_ids[best]))
        shares.append(int(counts[best]) / requests)
        start += requests
    return top_objects, shares


def coverage_per_day(first_time, last_time, t0, num_days):
    """Fraction of each day [t0 + d*86400, t0 + (d+1)*86400) that lies inside the trace's
    time span [first_time, last_time]."""
    coverage = []
    for d in range(num_days):
        start = t0 + d * SECONDS_PER_DAY
        end = start + SECONDS_PER_DAY
        covered = min(last_time, end) - max(first_time, start)
        coverage.append(max(0, covered) / SECONDS_PER_DAY)
    return coverage


def simulate(policy, prep):
    """Run every request of prep through policy. Returns a RunResult."""
    misses_per_day = [0] * prep.num_days
    request = policy.request    # local name: avoids an attribute lookup per request
    loop_ns = 0
    peak_entries = 0
    weight_log = []
    has_weights = hasattr(policy, "w_lru")
    if has_weights:
        weight_log.append((0, 0.0, policy.w_lru))

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for c in range(len(prep.boundaries) - 1):
            chunk = prep.keys[prep.boundaries[c]:prep.boundaries[c + 1]]    # copied before timing
            hits = 0
            start = time.perf_counter_ns()
            for key in chunk:
                if request(key):
                    hits += 1
            loop_ns += time.perf_counter_ns() - start

            misses_per_day[prep.chunk_day[c]] += len(chunk) - hits
            peak_entries = max(peak_entries, policy.metadata_entries())
            if has_weights:
                weight_log.append((prep.boundaries[c + 1], prep.chunk_end_days[c], policy.w_lru))
    finally:
        if gc_was_enabled:
            gc.enable()

    misses = sum(misses_per_day)
    return RunResult(policy=policy.name, capacity=policy.capacity, misses=misses,
                     misses_per_day=misses_per_day, loop_ns=loop_ns,
                     peak_metadata_entries=peak_entries, weight_log=weight_log,
                     stats=policy_stats(policy, misses))


def policy_stats(policy, misses):
    """Counters worth reporting, per policy type (empty dict for policies without any)."""
    if hasattr(policy, "n_same_victim"):    # LeCaR
        evictions = misses - len(policy)
        return {
            "evictions": evictions,
            "same_victim": policy.n_same_victim,
            "same_victim_fraction": policy.n_same_victim / evictions if evictions else 0.0,
            "evict_lru": policy.n_evict_lru,
            "evict_lfu": policy.n_evict_lfu,
            "regret_lru": policy.n_hit_lru_history,
            "regret_lfu": policy.n_hit_lfu_history,
            "history_dropped": policy.n_history_dropped,
            "lfu_min_tracking": "heap" if policy.lfu.use_heap else "scan",
            "min_scans": policy.lfu.n_min_scans,
            "buckets_scanned": policy.lfu.n_buckets_scanned,
            "heap_pushes": policy.lfu.n_heap_pushes,
            "heap_pops": policy.lfu.n_heap_pops,
            "heap_rebuild_items": policy.lfu.n_heap_rebuild_items,
            "final_w_lru": policy.w_lru,
        }
    if hasattr(policy, "p"):    # ARC
        return {"final_p": policy.p}
    return {}
