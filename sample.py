#!/usr/bin/env python3
"""
sample.py - spatial (SHARDS-style) sampling of the Wikipedia CDN trace.

Reads the trace from stdin in ONE streaming pass and writes one sampled trace per rate.
The full trace is never held in memory.

Usage:
    tar -xzOf wiki2018.tr.tar.gz | python3 sample.py --rates 0.001 0.01 --out data/
    pigz -dc wiki2018.tr.tar.gz | tar -xO | python3 sample.py --rates 0.001 0.01 --out data/

A second, independent 1% sample (one rate, another hash seed):
    pigz -dc wiki2018.tr.tar.gz | tar -xO | python3 sample.py --rates 0.01 --seed 1 --out data/
    -> data/wiki2018_0.01_seed1.tr and results/trace_stats_seed1.txt
Existing output files are never overwritten unless --overwrite is given.

Input lines:  "time id size [extra columns...]"  (space separated, per the LRB README)
Output lines: "time id size"  (extra columns dropped; size kept for a possible LRB run later)

Sampling rule (spatial sampling): keep EVERY request of an object iff h(id) < rate * 2**32.
The same hash is used for every rate, so a smaller-rate sample is a subset of a larger one.

Hash: h(id) = top 32 bits of splitmix64(id + HASH_SEED), Vigna's 64-bit mixer
(prng.di.unimi.it/splitmix64.c). Every input bit affects every output bit, so the selected
ids look random even though the Wikipedia ids are small consecutive integers (0, 1, 2, ...
in order of first appearance). The earlier multiplicative hash, (id * 2654435761) mod 2**32,
was biased: h(0) = 0, so object 0 (one of the hottest) was in EVERY sample; in the first 1M
lines the 0.1% sample got 13,840 requests instead of about 1,000.
Non-integer ids are first turned into a 64-bit integer with blake2b (stdlib hashlib).

Seed: --seed 0 (the default) is the original sampling, h(id) = splitmix64(id) >> 32.
For seed s != 0, h_s(id) = splitmix64(splitmix64(id) + s) >> 32: the id is hashed FIRST and
the seed is mixed into the hashed value. Why not splitmix64(id + s) (used until 20 Oct
2026)? Then object k is in the seed-s sample exactly when object k + s is in the seed-0
sample, so the seed-s sample is the seed-0 sample shifted by s ids. Wikipedia ids are
assigned in order of first appearance, so neighbouring ids are objects first seen at the
same moment: the shifted sample had the same unique-object count (376,550), paired
objects had identical first-request times and request counts with rank correlation 0.52.
Not an independent sample (that version is kept in data/discarded/). With the id hashed
first, seed-s membership of object k depends on splitmix64(k) + s, a pseudo-random 64-bit
value, so no simple relation between ids links the two samples; a test checks that for
offsets d = 1..5 the share of seed-1 ids k with k + d in the seed-0 sample is about the rate. Samples of different rates made with the
SAME seed are nested; samples with different seeds share about rate x rate of objects.

Statistics printed at the end (and written to --stats):
  - full trace: total requests, first/last timestamp, number of days
  - each sample: total requests, unique objects, one-hit wonders, the share of requests
    that go to its single most popular object, first/last timestamp, number of days.
    With heavy-tailed popularity, one hot object landing in (or out of) a small sample
    moves its request count a lot; the top-object share makes that visible.
  - full-trace unique objects is ESTIMATED as (unique objects in largest sample) / rate,
    the standard SHARDS estimator. Counting it exactly would need a set of every id in
    the trace, which does not fit in this machine's free memory.
  - one-hit-wonder fraction = objects requested exactly once / unique objects. Spatial
    sampling keeps every request of a sampled object, so each sampled object's count is
    its full-trace count; the largest sample's fraction is therefore an estimate of the
    full trace's fraction (sampling error only, no bias from the sampling itself).
Days assume the timestamp is in seconds; check that the printed span is about 14 days.
"""

import argparse
import hashlib
import os
import sys

HASH_SPACE = 2 ** 32
MASK64 = 2 ** 64 - 1
HASH_SEED = 0                          # fixed; changing it gives a different (equally valid)
                                       # sample, so never change it between runs
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15    # splitmix64's increment (2**64 / golden ratio)
PROGRESS_EVERY = 50_000_000
SECONDS_PER_DAY = 86400


def splitmix64(x: int) -> int:
    """splitmix64 with state x: add the increment, then the 64-bit finalizer
    (xor-shift, multiply, xor-shift, multiply, xor-shift). Returns a 64-bit integer."""
    z = (x + SPLITMIX_GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def id_hash(key: bytes, seed: int = HASH_SEED) -> int:
    """Deterministic 32-bit hash of an object id given as raw bytes: top 32 bits of splitmix64.
    Seed 0: splitmix64(id). Other seeds: splitmix64(splitmix64(id) + seed) (see docstring)."""
    if key.isdigit():
        x = int(key) & MASK64
    else:
        x = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    if seed == HASH_SEED:
        return splitmix64(x) >> 32
    return splitmix64((splitmix64(x) + seed) & MASK64) >> 32


def sample_path(out_dir, rate, seed):
    """data/wiki2018_0.01.tr for seed 0; data/wiki2018_0.01_seed1.tr for seed 1."""
    suffix = "" if seed == HASH_SEED else f"_seed{seed}"
    return os.path.join(out_dir, f"wiki2018_{rate:g}{suffix}.tr")


def parse_time(t: bytes) -> int:
    """Timestamps should be integers; tolerate a float like b'1.0' just in case."""
    try:
        return int(t)
    except ValueError:
        return int(float(t))


class Sample:
    """One output sample: its file plus running statistics."""

    def __init__(self, rate, path):
        self.rate = rate
        self.threshold = int(rate * HASH_SPACE)    # keep id iff id_hash(id) < threshold
        self.path = path
        self.file = open(self.path, "wb", buffering=1 << 20)
        self.requests = 0
        self.counts = {}           # id -> requests; small: ~rate * (unique objects) entries
        self.first_time = None
        self.last_time = None
        self.out_of_order = 0      # requests whose time is earlier than the previous one

    def add(self, t: bytes, key: bytes, size: bytes):
        time_value = parse_time(t)
        if self.first_time is None:
            self.first_time = time_value
        elif time_value < self.last_time:
            self.out_of_order += 1
        self.last_time = time_value
        self.requests += 1
        self.counts[key] = self.counts.get(key, 0) + 1
        if size:
            self.file.write(t + b" " + key + b" " + size + b"\n")
        else:
            self.file.write(t + b" " + key + b"\n")

    def top_object(self):
        """(id, requests) of the most requested object in this sample."""
        return max(self.counts.items(), key=lambda item: item[1])

    def one_hit_wonders(self):
        return sum(1 for c in self.counts.values() if c == 1)

    def one_hit_wonder_fraction(self):
        return self.one_hit_wonders() / len(self.counts) if self.counts else 0.0


def day_count(first, last):
    if first is None:
        return 0
    return (last - first) // SECONDS_PER_DAY + 1


def format_stats(lines_read, bad_lines, first_t, last_t, samples, seed):
    largest = samples[0]
    estimated_unique = round(len(largest.counts) / largest.rate)
    out = []
    out.append("=== Full trace ===")
    out.append(f"hash seed             : {seed}")
    out.append(f"lines read            : {lines_read:,}")
    out.append(f"malformed lines       : {bad_lines:,}")
    out.append(f"total requests        : {lines_read - bad_lines:,}")
    out.append(f"unique objects (est.) : {estimated_unique:,}  "
               f"(= {len(largest.counts):,} / {largest.rate:g}, SHARDS estimate)")
    out.append(f"one-hit wonders (est.): {largest.one_hit_wonder_fraction():.4f} of unique objects "
               f"(from the {largest.rate:g} sample)")
    out.append(f"first timestamp       : {first_t}")
    out.append(f"last timestamp        : {last_t}")
    if first_t is not None:
        span = last_t - first_t
        out.append(f"span                  : {span:,} time units = {span / SECONDS_PER_DAY:.2f} days if seconds")
        out.append(f"days (if seconds)     : {day_count(first_t, last_t)}")
    for s in samples:
        out.append("")
        out.append(f"=== Sample rate {s.rate:g} -> {s.path} ===")
        out.append(f"total requests        : {s.requests:,}")
        out.append(f"unique objects        : {len(s.counts):,}")
        out.append(f"one-hit wonders       : {s.one_hit_wonders():,} "
                   f"({s.one_hit_wonder_fraction():.4f} of unique objects)")
        if s.counts:
            top_id, top_count = s.top_object()
            out.append(f"top object share      : {top_count / s.requests:.4f} of requests "
                       f"(id {top_id.decode(errors='replace')}, {top_count:,} requests)")
        out.append(f"first timestamp       : {s.first_time}")
        out.append(f"last timestamp        : {s.last_time}")
        out.append(f"days (if seconds)     : {day_count(s.first_time, s.last_time)}")
        out.append(f"out-of-order requests : {s.out_of_order:,}")
        for fraction in (0.001, 0.01, 0.1):
            slots = int(fraction * len(s.counts))
            note = "  <-- fewer than ~100 slots, flag in paper" if slots < 100 else ""
            out.append(f"cache {fraction:>5.1%} of uniques : {slots:,} slots{note}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Spatial sampling of a trace read from stdin.")
    parser.add_argument("--rates", type=float, nargs="+", required=True,
                        help="sampling rates, e.g. 0.001 0.01")
    parser.add_argument("--out", default="data/", help="output directory for sampled traces")
    parser.add_argument("--seed", type=int, default=HASH_SEED,
                        help=f"hash seed (default {HASH_SEED} = the original samples); another "
                             "seed gives a different sample sharing about rate x rate of objects")
    parser.add_argument("--stats", default=None,
                        help="file the final statistics are also written to (default "
                             "results/trace_stats.txt, or results/trace_stats_seed<N>.txt)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing existing sample or statistics files")
    args = parser.parse_args()

    for rate in args.rates:
        if not 0 < rate <= 1:
            parser.error(f"rate must be in (0, 1], got {rate}")
    if not 0 <= args.seed <= MASK64:
        parser.error("seed must be between 0 and 2**64 - 1")
    if args.stats is None:
        suffix = "" if args.seed == HASH_SEED else f"_seed{args.seed}"
        args.stats = f"results/trace_stats{suffix}.txt"

    rates = sorted(set(args.rates), reverse=True)
    paths = [sample_path(args.out, rate, args.seed) for rate in rates]
    existing = [p for p in paths + [args.stats] if os.path.exists(p)]
    if existing and not args.overwrite:
        parser.error(f"would overwrite {', '.join(existing)}; use --overwrite or another --seed")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)

    # Largest rate first: if an id fails the largest threshold it fails every smaller one,
    # so the inner loop can stop early (99%+ of lines need only one comparison).
    samples = [Sample(rate, path) for rate, path in zip(rates, paths)]
    seed = args.seed

    lines_read = 0
    bad_lines = 0
    first_t = None
    last_t = None

    for line in sys.stdin.buffer:
        lines_read += 1
        if lines_read % PROGRESS_EVERY == 0:
            kept = ", ".join(f"{s.rate:g}: {s.requests:,}" for s in samples)
            print(f"[progress] {lines_read:,} lines read; kept {kept}", file=sys.stderr, flush=True)

        parts = line.split(None, 3)    # time, id, size, rest-of-line (ignored)
        if len(parts) < 2:
            bad_lines += 1
            continue
        t, key = parts[0], parts[1]
        size = parts[2] if len(parts) > 2 else b""

        # Full-trace timestamps: kept as raw bytes and parsed once at the end (cheaper).
        # This assumes the trace is in time order; the samples' out-of-order count checks it.
        if first_t is None:
            first_t = t
        last_t = t

        h = id_hash(key, seed)
        for s in samples:
            if h >= s.threshold:
                break
            s.add(t, key, size)

    for s in samples:
        s.file.close()

    first_value = parse_time(first_t) if first_t is not None else None
    last_value = parse_time(last_t) if last_t is not None else None
    report = format_stats(lines_read, bad_lines, first_value, last_value, samples, seed)
    print(report)
    with open(args.stats, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
