import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.policies.belady import Belady, compute_next_use
from src.policies.lecar import LeCaR
from src.policies.lru import LRU
from src.sim import dense_keys, prepare, simulate
from src.trace import Trace, compute_days
from tests.util import random_traces, run

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400


def make_trace(times, ids):
    times = np.array(times, dtype=np.int64)
    return Trace(times=times, ids=np.array(ids, dtype=np.int64), days=compute_days(times))


def test_dense_keys_share_one_int_per_object():
    keys, unique, one_hit = dense_keys(np.array([900, 5, 900, 77, 5], dtype=np.int64))
    assert unique == 3
    assert keys == [2, 0, 2, 1, 0]    # ids sorted: 5 -> 0, 77 -> 1, 900 -> 2
    assert keys[0] is keys[2]         # same object, not a copy
    assert one_hit == 1               # only 77 is requested exactly once


def test_one_hit_wonder_fraction_matches_counting():
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 2000, size=3000)
    trace = make_trace(np.sort(rng.integers(0, 3 * DAY, size=3000)), ids)
    prep = prepare(trace)
    counts = {}
    for i in ids.tolist():
        counts[i] = counts.get(i, 0) + 1
    expected = sum(1 for c in counts.values() if c == 1)
    assert prep.one_hit_wonders == expected
    assert prep.one_hit_wonder_fraction == expected / len(counts)


def test_chunks_never_span_a_day():
    times = [0, 10, DAY - 1, DAY, DAY + 5, 3 * DAY]    # no requests on day 2
    trace = make_trace(times, [1, 2, 3, 1, 2, 3])
    prep = prepare(trace, sample_every=4)
    assert prep.requests_per_day == [3, 2, 0, 1]
    assert 3 in prep.boundaries and 5 in prep.boundaries    # day starts
    assert 4 in prep.boundaries                               # sample point
    for c, start in enumerate(prep.boundaries[:-1]):
        end = prep.boundaries[c + 1]
        assert len(set(trace.days[start:end].tolist())) == 1
        assert prep.chunk_day[c] == trace.days[start]


def test_simulate_matches_plain_loop_per_day():
    rng = np.random.default_rng(0)
    for keys, capacity in random_traces(count=10, seed=12):
        times = np.sort(rng.integers(0, 5 * DAY, size=len(keys)))
        trace = make_trace(times, keys)
        prep = prepare(trace, sample_every=37)
        result = simulate(LRU(capacity), prep)

        hits = run(LRU(capacity), prep.keys)
        expected_per_day = [0] * prep.num_days
        for hit, day in zip(hits, trace.days.tolist()):
            if not hit:
                expected_per_day[day] += 1
        assert result.misses_per_day == expected_per_day
        assert result.misses == hits.count(False)
        assert result.peak_metadata_entries <= capacity


def test_lecar_weight_log_and_stats():
    keys, capacity = next(random_traces(seed=13))
    trace = make_trace(list(range(len(keys))), keys)
    prep = prepare(trace, sample_every=50)
    result = simulate(LeCaR(capacity, seed=1), prep)
    assert len(result.weight_log) == len(prep.boundaries)    # start + one per chunk
    assert result.weight_log[0] == (0, 0.0, 0.5)
    assert result.weight_log[-1][0] == len(keys)
    stats = result.stats
    assert stats["evict_lru"] + stats["evict_lfu"] + stats["same_victim"] == stats["evictions"]


def test_belady_shared_next_use_gives_same_result():
    keys, capacity = next(random_traces(seed=14))
    shared = compute_next_use(keys)
    assert run(Belady(capacity, keys, shared), keys) == run(Belady(capacity, keys), keys)


def test_run_all_end_to_end(tmp_path):
    trace_path = tmp_path / "synthetic.tr"
    subprocess.run([sys.executable, "-m", "experiments.make_synthetic", "--out", str(trace_path),
                    "--requests", "20000", "--objects", "5000"], cwd=ROOT, check=True,
                   capture_output=True)
    out = tmp_path / "results"
    subprocess.run([sys.executable, "-m", "experiments.run_all", "--trace", str(trace_path),
                    "--out", str(out), "--fractions", "0.02", "0.2", "--seeds", "1", "2",
                    "--timing-runs", "2"], cwd=ROOT, check=True, capture_output=True)

    for name in ["runs", "miss_ratio", "per_day", "weights", "timing", "metadata",
                 "lecar_stats", "lecar_vs_lcs", "competitive"]:
        with open(out / f"{name}.csv") as f:
            assert len(list(csv.DictReader(f))) > 0, name
    info = json.loads((out / "run_info.json").read_text())
    assert info["requests"] == 20000 and info["days"] == 14
    assert 0 < info["one_hit_wonder_fraction"] < 1
    assert info["one_hit_wonders"] == round(info["one_hit_wonder_fraction"] * info["unique_objects"])
    with open(out / "competitive.csv") as f:
        rows = list(csv.DictReader(f))
    assert {"observed_ratio_to_opt", "worst_case_bound", "bound_source"} <= set(rows[0])
    bounds = {r["policy"]: r["worst_case_bound"] for r in rows}
    assert bounds["LFU"] == bounds["ARC"] == "needs citation"
    assert all(float(r["observed_ratio_to_opt"]) >= 1 for r in rows)
    with open(out / "metadata.csv") as f:
        rows = list(csv.DictReader(f))
    assert "model_bytes" not in rows[0]
    assert all(float(r["python_bytes_per_slot"]) > 0 for r in rows)
    assert "FAIL" not in (out / "sanity.txt").read_text()
    with open(out / "per_day.csv") as f:
        days = {int(row["day"]) for row in csv.DictReader(f)}
    assert days == set(range(1, 15))

    subprocess.run([sys.executable, "-m", "experiments.plot", "--results", str(out),
                    "--shift-day", "7"], cwd=ROOT, check=True, capture_output=True)
    assert len(list((out / "plots").glob("*.png"))) >= 6


def test_coverage_per_day_and_partial_last_day():
    # Trace from t0 to 2.25 days: days 1 and 2 are full, day 3 has a quarter of a day.
    times = list(range(0, int(2.25 * DAY), 600))
    trace = make_trace(times, [i % 7 for i in range(len(times))])
    prep = prepare(trace, t0=0)
    assert prep.num_days == 3
    assert prep.day_coverage[0] == 1.0 and prep.day_coverage[1] == 1.0
    assert abs(prep.day_coverage[2] - (times[-1] - 2 * DAY) / DAY) < 1e-12
    assert prep.partial_days == [2]    # 0-based


def test_late_start_makes_first_day_partial():
    # t0 = 0 but the trace starts at noon of day 1
    times = list(range(DAY // 2, 3 * DAY, 600))
    prep = prepare(make_trace(times, [1] * len(times)), t0=0)
    assert prep.partial_days == [0]


def test_partial_day_is_marked_and_can_be_left_out_of_plots(tmp_path):
    times = list(range(0, int(2.3 * DAY), 20))    # day 3 is partial (30%)
    rng = np.random.default_rng(0)
    ids = rng.zipf(1.3, size=len(times)) % 3000
    trace_path = tmp_path / "partial.tr"
    trace_path.write_text("".join(f"{t} {i} 1\n" for t, i in zip(times, ids)))
    out = tmp_path / "results"
    subprocess.run([sys.executable, "-m", "experiments.run_all", "--trace", str(trace_path),
                    "--out", str(out), "--fractions", "0.05", "--seeds", "1", "2",
                    "--timing-runs", "1", "--no-python-bytes", "--t0", "0"],
                   cwd=ROOT, check=True, capture_output=True)
    info = json.loads((out / "run_info.json").read_text())
    assert info["partial_days"] == [3]
    with open(out / "per_day.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["day"] for r in rows if r["partial"] == "True"} == {"3"}
    assert all(0 < float(r["top_object_share"]) <= 1 for r in rows)
    lru_rows = sorted((r for r in rows if r["policy"] == "LRU"), key=lambda r: int(r["day"]))
    assert [float(r["top_object_share"]) for r in lru_rows] == info["day_top_share"]
    assert all(r["partial"] == "False" for r in rows if r["day"] in ("1", "2"))

    from experiments.plot import read_partial_days
    days = read_partial_days(str(out))
    assert list(days) == [3] and abs(days[3] - 0.3) < 0.01
    for extra in ([], ["--include-partial-days"]):
        subprocess.run([sys.executable, "-m", "experiments.plot", "--results", str(out), *extra],
                       cwd=ROOT, check=True, capture_output=True)



def test_top_object_share_per_day():
    # day 1: ids 5 5 5 6 (5 has 3/4); day 2: 6 7 7 (7 has 2/3); day 3 empty; day 4: 9
    times = [0, 1, 2, 3, DAY, DAY + 1, DAY + 2, 3 * DAY]
    ids = [5, 5, 5, 6, 6, 7, 7, 9]
    prep = prepare(make_trace(times, ids), t0=0)
    assert prep.requests_per_day == [4, 3, 0, 1]
    assert prep.day_top_object == [5, 7, None, 9]
    assert prep.day_top_share == [0.75, 2 / 3, 0.0, 1.0]


def test_day_winners_ranks_and_counts(tmp_path):
    from experiments.day_winners import write_day_winners
    rows = [
        # day 1: LeCaR best; day 2: ARC best (LeCaR is the mean of its two seeds); day 3 partial
        ("ARC", "", 1, 0.30, False), ("LeCaR", "1", 1, 0.20, False), ("LeCaR", "2", 1, 0.22, False),
        ("LRU", "", 1, 0.50, False), ("Belady", "", 1, 0.10, False),
        ("ARC", "", 2, 0.25, False), ("LeCaR", "1", 2, 0.26, False), ("LeCaR", "2", 2, 0.30, False),
        ("LRU", "", 2, 0.40, False), ("Belady", "", 2, 0.05, False),
        ("ARC", "", 3, 0.90, True), ("LeCaR", "1", 3, 0.10, True), ("LRU", "", 3, 0.95, True),
    ]
    with open(tmp_path / "per_day.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "cache_fraction", "seed", "day", "miss_ratio", "partial"])
        for policy, seed, day, ratio, partial in rows:
            writer.writerow([policy, "0.01", seed, day, ratio, partial])
    wins = {r["policy"]: r for r in write_day_winners(str(tmp_path))}
    assert set(wins) == {"ARC", "LeCaR", "LRU"}            # Belady is not ranked
    assert wins["LeCaR"]["days_best"] == 1 and wins["ARC"]["days_best"] == 1
    assert wins["LeCaR"]["days_best_excl_day1"] == 0 and wins["ARC"]["days_best_excl_day1"] == 1
    assert all(r["complete_days"] == 2 for r in wins.values())    # partial day 3 left out
    with open(tmp_path / "day_ranks.csv") as f:
        ranks = {(r["day"], r["policy"]): r for r in csv.DictReader(f)}
    assert ranks[("1", "LeCaR")]["rank"] == "1" and abs(float(ranks[("1", "LeCaR")]["miss_ratio"]) - 0.21) < 1e-12
    assert ranks[("2", "LeCaR")]["rank"] == "2" and ranks[("2", "LRU")]["rank"] == "3"


def test_ratio_to_lru_drift_check_and_seed_comparison(tmp_path):
    folders = []
    for seed in (0, 1):
        trace_path = tmp_path / f"synthetic{seed}.tr"
        subprocess.run([sys.executable, "-m", "experiments.make_synthetic", "--out", str(trace_path),
                        "--requests", "20000", "--objects", "5000", "--seed", str(seed)],
                       cwd=ROOT, check=True, capture_output=True)
        out = tmp_path / f"results{seed}"
        subprocess.run([sys.executable, "-m", "experiments.run_all", "--trace", str(trace_path),
                        "--out", str(out), "--fractions", "0.02", "0.2", "--seeds", "1", "2",
                        "--timing-runs", "2", "--no-python-bytes"],
                       cwd=ROOT, check=True, capture_output=True)
        folders.append(out)

        with open(out / "timing.csv") as f:
            timing = {(r["policy"], r["cache_fraction"]): r for r in csv.DictReader(f)}
        for (policy, fraction), r in timing.items():
            lru = float(timing[("LRU", fraction)]["ns_per_request_median"])
            assert abs(float(r["ratio_to_lru"]) - float(r["ns_per_request_median"]) / lru) < 1e-9
        assert all(float(timing[("LRU", f)]["ratio_to_lru"]) == 1.0 for f in ("0.02", "0.2"))
        reference = json.loads((out / "run_info.json").read_text())["lru_reference_ns_per_request"]
        assert set(reference) == {"start", "end", "end_over_start"}
        assert set(reference["start"]) == {"0.02", "0.2"}

    result = subprocess.run([sys.executable, "-m", "experiments.compare_seeds", "--a", str(folders[0]),
                             "--b", str(folders[1]), "--out", str(tmp_path / "comparison")],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    assert "Rankings" in result.stdout and "LRU drift" in result.stdout
    for name in ("rankings", "lecar_arc_gap", "day_wins", "timing_ratios"):
        with open(tmp_path / "comparison" / f"{name}.csv") as f:
            assert len(list(csv.DictReader(f))) > 0, name
