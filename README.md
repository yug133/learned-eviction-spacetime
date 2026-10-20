# learned-eviction-spacetime

How much does a learned cache eviction policy actually buy you once you count what each
decision costs?

This project puts the learned policy **LeCaR** next to the classic heuristics **LRU**, **LFU**
and **ARC**, plus **Belady's offline optimum** as a lower bound. All five run in one Python
simulator on a real workload: the 2.8-billion-request Wikipedia CDN trace. For each policy it
measures three things side by side:

- miss ratio,
- time per request,
- memory per cache slot.

## Results at a glance

Two independent 1% samples of the trace (different hash seeds); cache sizes of 0.1%, 1% and
10% of unique objects; day 1 excluded as warm-up.

| Cache size | Best online policy | LeCaR's misses vs ARC | LeCaR's time vs ARC | LeCaR's memory vs ARC |
|---|---|---|---|---|
| 0.1% | ARC | 1.9–2.0% more | 3.2–3.5× | 2.4× |
| 1% | LeCaR | 4.5% fewer | 4.5× | 2.7–2.8× |
| 10% | LeCaR | 8.0–8.1% fewer | 5.8–6.3× | 2.3× |

Ranges cover the two samples. Memory is CPython bytes per cache slot.

- **Learning pays only at larger caches.** At 0.1%, ARC beats LeCaR on both miss ratio and
  speed.
- **The winner does not drift.** On both samples the best policy is the same on all 15 days
  at 0.1% and 1%, and on 14 of 15 at 10% (ARC wins day 1).
- **Configuration matters as much as the algorithm.** With the settings libCacheSim uses,
  LeCaR misses 5.9–8.7 percentage points more than with the authors' settings at the two
  smaller sizes, ending up close to LRU.
- **Big-O isn't the whole story.** For LeCaR's least-frequently-used lookup, an O(F) linear
  scan beats an O(log F) heap until the number of distinct frequency counts gets large. In
  CPython the scan runs in C, while each heap step is an interpreted call.

Time is reported relative to LRU in the same run, so the ratios carry across machines better
than absolute nanoseconds do (see [Reproducibility](#reproducibility)).

## What's inside

- **Policies** (`src/policies/`): LRU, O(1) frequency-bucket LFU, ARC, LeCaR (with a
  libCacheSim-compatible preset), and Belady via precomputed next-use plus a lazy max-heap.
  Each file's docstring states its time and space complexity.
- **Sampler** (`sample.py`): streams the 16.8 GB compressed trace in one pass, never
  unpacking it. It does SHARDS-style spatial sampling with a splitmix64 hash, so each sampled
  object keeps every one of its requests.
- **Simulator** (`src/sim.py`): miss ratio overall and per day, timing, and bytes per slot
  measured with `tracemalloc`.
- **Tests** (`tests/`): every policy is checked against a brute-force reference, including
  exhaustive search for Belady on all 16,384 traces of length 7 over 4 keys. There are also
  invariant tests, such as Belady never missing more than any online policy.
- **Experiments** (`experiments/`): one command per sample produces every CSV and figure.

## Requirements

Python 3.10+, NumPy, Matplotlib and pytest; nothing else. Reference setup: Python 3.10.14,
NumPy 1.26.4, Matplotlib 3.10.9 on an Intel Core i7-12700H with 14.8 GiB RAM, Ubuntu 25.10.
`pigz` is optional and speeds up decompression.

## Get the trace

The trace is the Wikipedia CDN trace released with LRB: **https://github.com/sunnyszy/lrb**.
Download `wiki2018.tr.tar.gz` (16.8 GB compressed) from the link in that repository's README
and put it in `data/`. Neither the trace nor the samples are committed (`data/` is ignored).

## Reproduce

Run everything from the repository root.

```bash
# 1. Sample (one pass over 2.8B lines, about 1 hour each).
#    Seed 0: 0.1% (development) and 1% samples; seed 1: an independent 1% sample.
pigz -dc data/wiki2018.tr.tar.gz | tar -xO | python3 sample.py --rates 0.001 0.01 --out data/
pigz -dc data/wiki2018.tr.tar.gz | tar -xO | python3 sample.py --rates 0.01 --seed 1 --out data/
#    (without pigz: tar -xzOf data/wiki2018.tr.tar.gz | python3 sample.py ...)
#    Statistics: results/trace_stats.txt, results/trace_stats_seed1.txt

# 2. Tests (under a minute)
python3 -m pytest

# 3. All experiments on both 1% samples (about 45 min each, peak memory about 1.8 GB).
#    --t0 0 uses the full trace's day boundaries. Run nothing else alongside (timings).
python3 -m experiments.run_all --trace data/wiki2018_0.01.tr --out results/wiki_0.01 --t0 0
python3 -m experiments.run_all --trace data/wiki2018_0.01_seed1.tr --out results/wiki_0.01_seed1 --t0 0
python3 -m experiments.compare_seeds --a results/wiki_0.01 --b results/wiki_0.01_seed1 \
    --out results/seed_comparison

# 4. Figures
python3 -m experiments.plot --results results/wiki_0.01
python3 -m experiments.plot --results results/wiki_0.01_seed1
python3 -m experiments.paper_figures --seed0 results/wiki_0.01 --seed1 results/wiki_0.01_seed1 \
    --out figures
```

Two optional scripts:

- `experiments.time_min_tracking` times the scan against the heap for LeCaR's LFU minimum.
- `experiments.diagnose_lecar` shows why the libCacheSim configuration cannot learn on a trace
  that favours LFU.

Each script's docstring explains its inputs and outputs.

## Reproducibility

- **Deterministic:** sampling (fixed hash seeds), and every miss ratio and counter. LeCaR is
  randomised and runs with seeds 1–5; the standard deviation is at most 0.0002.
- **Not deterministic:** absolute time per request. It depends on the machine and drifted up
  to 13% between sessions here. Results therefore report time as a ratio to LRU in the same
  run (`ratio_to_lru` in `timing.csv`, accurate to about ±10%). Each run times LRU at its start
  and end to record drift (`run_info.json`).

## Folder layout

```
sample.py                   stream the trace; spatial (SHARDS-style) hash sampling
src/
  trace.py                  load a sampled trace into numpy arrays, day index
  sim.py                    run one policy over a trace; timing, per-day misses, counters
  policies/                 lru.py lfu.py arc.py lecar.py belady.py
tests/                      pytest; reference.py = brute-force implementations
experiments/
  run_all.py                every experiment for one sample -> CSVs, run_info.json, sanity.txt
  compare_seeds.py          rankings, LeCaR-ARC gap, per-day wins, timing ratios: seed 0 vs 1
  day_winners.py            per-day ranks and win counts
  plot.py, paper_figures.py figures from the CSVs only
  time_min_tracking.py      LeCaR's LFU minimum: scan vs heap
  diagnose_lecar.py         LeCaR configuration ablation on a crafted trace
  make_synthetic.py         Zipf trace with a popularity shift (pipeline tests only)
results/
  wiki_0.01/  wiki_0.01_seed1/  seed_comparison/   main results
  wiki_0.001/                                      0.1% development sample
  archive/                  superseded run (LeCaR with the scan default), kept for the record
  trace_stats*.txt, *.log   sampling statistics and run logs
figures/                    summary figures (Pareto view, per-day miss ratio, LeCaR weights)
```

## How LeCaR's settings are sourced

The LeCaR paper (Vietri et al., HotStorage 2018) leaves several details open, and public
implementations disagree on them. Every choice is traced to a pinned commit and line numbers
in the docstring of [`src/policies/lecar.py`](src/policies/lecar.py):

| Detail | Default here (authors' code) | `lecar_lcs()` preset (libCacheSim) |
|---|---|---|
| History size per expert | C | C/2 |
| Frequency of a key returning from history | stored count + 1 | reset to 1 |
| Weight bound | clamp to [0.01, 0.99] | 1e-10 floor |
| Victim both experts agree on | into no history | into no history |

**Sources**

- The default follows the authors' repository [sylab/LeCaR](https://github.com/sylab/LeCaR)
  (commit e4044de).
- Also consulted: [sylab/cacheus](https://github.com/sylab/cacheus) (1eec63c).
- The `lecar_lcs()` preset follows [libCacheSim](https://github.com/1a1a11a/libCacheSim)
  `LeCaR.c` (5b4dd39), the version used in the S3-FIFO study.

All three use learning rate 0.45 and discount 0.005^(1/C).

The same docstring lists the remaining deviations: FIFO tie-break in the LFU expert, float64
weights, Python's random generator, and a heap for the LFU minimum. LeCaR is checked against a
brute-force reimplementation for every combination of options.

## Limitations

- Capacity is counted in objects, not bytes.
- Timings come from one Python implementation; a compiled one would change absolute costs and
  possibly the scan-versus-heap crossover.
- One trace from one CDN.
- Belady is the optimum only for caches that admit every miss.

Next steps: LRB on the same samples, byte-sized objects, and a compiled implementation.

## License

MIT, see [LICENSE](LICENSE). The Wikipedia trace is not included; see the LRB repository for
its terms.
