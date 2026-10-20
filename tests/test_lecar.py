import math

import pytest

from src.policies.belady import Belady
from src.policies.lecar import LeCaR, lecar_lcs
from src.policies.lru import LRU
from tests.reference import NaiveLeCaR, brute_force_lfu, brute_force_lru
from tests.util import lfu_favouring_trace, lru_favouring_trace, random_traces, run

# Options that make the weights a pure switch: no restore, clamp or floor.
PURE = {"restore_freq": False, "weight_clamp": 0.0, "weight_floor": 0.0}
# The libCacheSim configuration as keyword arguments (history size is set per capacity).
LCS = {"restore_freq": False, "weight_clamp": 0.0, "weight_floor": 1e-10}


def naive_lcs(capacity, seed):
    return NaiveLeCaR(capacity, seed=seed, history_size=capacity // 2, **LCS)


# --- configuration ---

def test_default_is_the_authors_configuration():
    lecar = LeCaR(200)
    assert lecar.learning_rate == 0.45
    assert lecar.discount == pytest.approx(0.005 ** (1 / 200))
    assert (lecar.w_lru, lecar.w_lfu) == (0.5, 0.5)
    assert lecar.history_size == 200
    assert lecar.restore_freq is True
    assert lecar.weight_clamp == 0.01
    assert lecar.weight_floor == 0.0
    assert lecar.same_victim_history is False
    assert lecar.lfu.use_heap is True    # heap default since 21 Oct 2026 (timed on 1% sample)
    assert lecar.name == "LeCaR"


def test_lcs_preset():
    lecar = lecar_lcs(200, seed=3)
    assert lecar.history_size == 100
    assert lecar.restore_freq is False
    assert lecar.weight_clamp == 0.0
    assert lecar.weight_floor == 1e-10
    assert lecar.name == "LeCaR-LCS"
    assert lecar.lfu.use_heap is False   # LCS never looks up the min: scan costs nothing
    assert lecar_lcs(200, history_size=7).history_size == 7    # overrides work


# --- frozen weights ---

def test_frozen_at_lru_behaves_exactly_like_lru():
    for keys, capacity in random_traces(seed=5):
        lecar = LeCaR(capacity, initial_weights=(1.0, 0.0), **PURE)
        assert run(lecar, keys) == run(LRU(capacity), keys)
        assert (lecar.w_lru, lecar.w_lfu) == (1.0, 0.0)


def test_frozen_at_lfu_behaves_exactly_like_lfu():
    for keys, capacity in random_traces(seed=6):
        lecar = LeCaR(capacity, initial_weights=(0.0, 1.0), **PURE)
        assert run(lecar, keys) == brute_force_lfu(capacity, keys)
        assert (lecar.w_lru, lecar.w_lfu) == (0.0, 1.0)


# --- the update rule ---

def test_same_victim_goes_to_no_history():
    # capacity 2: A, B both count 1; on C, LRU victim = A and LFU victim = A (oldest count-1)
    lecar = LeCaR(2)
    run(lecar, ["A", "B", "C"])
    assert "A" not in lecar
    assert not lecar.h_lru and not lecar.h_lfu
    assert lecar.n_same_victim == 1


def test_regret_update_uses_time_in_history():
    # capacity 2: A A B C -> LRU victim A (least recent), LFU victim B (count 1 < 2).
    # They differ, so a random draw picks the expert; check the update for either outcome.
    for seed in range(1, 11):
        lecar = LeCaR(2, seed=seed)
        run(lecar, ["A", "A", "B", "C"])    # eviction at time 4
        if "A" in lecar.h_lru:
            regretted, evicted_key = "lru", "A"
        else:
            assert "B" in lecar.h_lfu
            regretted, evicted_key = "lfu", "B"
        run(lecar, ["C"] * 3)               # times 5, 6, 7 (hits)
        lecar.request(evicted_key)          # time 8: miss on a history key, t = 8 - 4 = 4

        reward = math.exp(0.45 * lecar.discount ** 4)    # paper Alg. 2 form
        if regretted == "lru":    # LRU's eviction regretted -> LFU rewarded
            expected_w_lru = 0.5 / (0.5 + 0.5 * reward)
        else:
            expected_w_lru = 0.5 * reward / (0.5 * reward + 0.5)
        assert lecar.w_lru == pytest.approx(expected_w_lru)
        assert lecar.w_lru + lecar.w_lfu == pytest.approx(1.0)
        assert evicted_key not in lecar.h_lru and evicted_key not in lecar.h_lfu


def test_restore_freq_gives_stored_count_plus_one():
    lecar = LeCaR(2, initial_weights=(1.0, 0.0))    # default restore_freq=True; always LRU
    run(lecar, ["A", "A", "A", "B", "C"])    # A (count 3) evicted by LRU into H_lru
    assert lecar.h_lru["A"][1] == 3
    lecar.request("A")
    assert lecar.lfu.frequency("A") == 4


def test_without_restore_returning_key_starts_at_one():
    lecar = lecar_lcs(2, initial_weights=(1.0, 0.0))
    run(lecar, ["A", "A", "A", "B", "C", "A"])    # A evicted by LRU with count 3, returns
    assert lecar.lfu.frequency("A") == 1


def test_restore_ignores_same_victim_keys():
    # A key evicted when both experts agree is in no history, so it restarts at 1.
    lecar = LeCaR(2)
    run(lecar, ["A", "B", "C", "A"])
    assert lecar.n_same_victim >= 1
    assert lecar.lfu.frequency("A") == 1


def test_clamp_sets_exactly_99_01():
    # Start at pure LRU. The first regret against LRU leaves w_lru = 1 before the clamp
    # (0 + anything stays 0 for LFU), so the clamp must set exactly (0.99, 0.01).
    lecar = LeCaR(4, initial_weights=(1.0, 0.0))
    for key in lfu_favouring_trace():
        lecar.request(key)
        if lecar.n_hit_lru_history == 1:
            break
    assert (lecar.w_lru, lecar.w_lfu) == (0.99, 0.01)


def test_clamp_keeps_weights_in_range_after_updates():
    for keys, capacity in random_traces(seed=11):
        lecar = LeCaR(capacity, seed=2)
        for key in keys:
            lecar.request(key)
            if lecar.n_hit_lru_history + lecar.n_hit_lfu_history > 0:
                assert 0.01 <= lecar.w_lru <= 0.99
                assert lecar.w_lru + lecar.w_lfu == pytest.approx(1.0)


def test_weight_floor_keeps_both_weights_positive():
    # Frozen at pure LRU on a trace where LRU's evictions are regretted.
    keys = lfu_favouring_trace()
    plain = LeCaR(4, initial_weights=(1.0, 0.0), **PURE)
    floored = LeCaR(4, initial_weights=(1.0, 0.0), **{**PURE, "weight_floor": 1e-10})
    run(plain, keys)
    run(floored, keys)
    assert plain.n_hit_lru_history > 0
    assert plain.w_lfu == 0.0          # no floor: a zero weight stays zero forever
    assert floored.w_lfu > 0.0         # floor: the frozen expert can come back
    assert floored.w_lru + floored.w_lfu == pytest.approx(1.0)


def test_same_victim_history_follows_algorithm_1():
    lecar = LeCaR(2, same_victim_history=True)
    run(lecar, ["A", "B", "C"])        # both experts nominate A
    assert lecar.n_same_victim == 1
    assert "A" in lecar.h_lru or "A" in lecar.h_lfu


def test_lcs_history_size_zero_for_capacity_one():
    lecar = lecar_lcs(1)               # C // 2 = 0, as in libCacheSim
    run(lecar, [1, 2, 1, 2, 3, 1])
    assert not lecar.h_lru and not lecar.h_lfu


# --- learning behaviour on the two crafted traces ---

def test_weight_moves_towards_lru_when_recency_wins():
    # The working set changes every 300 requests, so old frequent keys go cold.
    keys = lru_favouring_trace()
    lru_misses = brute_force_lru(6, keys).count(False)
    lfu_misses = brute_force_lfu(6, keys).count(False)
    assert (lru_misses, lfu_misses) == (180, 7257)
    for seed in range(1, 6):
        for lecar, naive in [(LeCaR(6, seed=seed), NaiveLeCaR(6, seed=seed)),
                             (lecar_lcs(6, seed=seed), naive_lcs(6, seed))]:
            hits = run(lecar, keys)
            assert hits == run(naive, keys)
            assert hits.count(False) < lfu_misses / 10
            assert lecar.w_lru > 0.8


def test_default_learns_lfu_when_frequency_wins():
    # A hot key evicted by LRU comes back with its count (restore_freq), so the experts
    # keep disagreeing, LRU keeps being regretted, and LeCaR converges to LFU.
    keys = lfu_favouring_trace()
    lru_misses = brute_force_lru(4, keys).count(False)
    lfu_misses = brute_force_lfu(4, keys).count(False)
    assert (lru_misses, lfu_misses) == (6001, 3003)
    for seed in range(1, 6):
        lecar = LeCaR(4, seed=seed)
        hits = run(lecar, keys)
        assert hits == run(NaiveLeCaR(4, seed=seed), keys)
        assert hits.count(False) < 3200
        assert lecar.w_lru == 0.01    # pinned at the clamp


def test_lcs_stays_at_lru_when_frequency_wins():
    # Frequency reset: after LRU evicts each hot key once, all keys have count 1, both
    # experts nominate the same victim, nothing enters a history, nothing is learned.
    # See experiments/diagnose_lecar.py.
    keys = lfu_favouring_trace()
    for seed in range(1, 6):
        lecar = lecar_lcs(4, seed=seed)
        hits = run(lecar, keys)
        assert hits == run(naive_lcs(4, seed), keys)
        assert hits.count(False) > 5900
        assert lecar.n_same_victim > 0.99 * (hits.count(False) - len(lecar))


def test_lcs_no_learning_when_every_eviction_has_count_one():
    keys = []
    for i in range(3000):
        keys += [i % 4, 1000 + i]
    lecar = lecar_lcs(4, seed=1)
    run(lecar, keys)
    assert (lecar.w_lru, lecar.w_lfu) == (0.5, 0.5)


# --- structure and invariants ---

def test_histories_stay_within_cap_and_disjoint():
    for keys, capacity in random_traces(seed=7):
        for lecar in (LeCaR(capacity, seed=3), lecar_lcs(capacity, seed=3)):
            for key in keys:
                lecar.request(key)
                assert len(lecar.h_lru) <= lecar.history_size
                assert len(lecar.h_lfu) <= lecar.history_size
                assert not (set(lecar.h_lru) & set(lecar.h_lfu))
                assert set(lecar.lru.order) == set(lecar.lfu.freq_of)    # same contents
                assert not (set(lecar.h_lru) & set(lecar.lru.order))    # ghosts not resident


def test_same_seed_is_reproducible():
    keys, capacity = next(random_traces(seed=8))
    assert run(LeCaR(capacity, seed=4), keys) == run(LeCaR(capacity, seed=4), keys)


def test_weight_log():
    lecar = LeCaR(3, log_every=10)
    run(lecar, list(range(50)))
    assert [t for t, _ in lecar.weight_log] == [10, 20, 30, 40, 50]


def test_counters_add_up():
    for keys, capacity in random_traces(count=10, seed=10):
        lecar = LeCaR(capacity)
        hits = run(lecar, keys)
        evictions = hits.count(False) - len(lecar)
        assert lecar.n_evict_lru + lecar.n_evict_lfu + lecar.n_same_victim == evictions


def test_capacity_invariant_and_belady_never_worse():
    for keys, capacity in random_traces(count=60, seed=2):
        belady = run(Belady(capacity, keys), keys).count(False)
        for seed in range(1, 6):
            assert belady <= run(LeCaR(capacity, seed=seed), keys).count(False)
            assert belady <= run(lecar_lcs(capacity, seed=seed), keys).count(False)


# --- agreement with the brute-force reference, for every option ---

OPTION_SETS = [
    {},                                                    # SY default
    {"half_history": True, **LCS},                         # libCacheSim
    {"weight_clamp": 0.0},
    {"restore_freq": False},
    {"weight_floor": 1e-10},
    {"same_victim_history": True},
    {"half_history": True, "weight_floor": 1e-10, "same_victim_history": True},
]


@pytest.mark.parametrize("options", OPTION_SETS)
def test_matches_naive_reference(options):
    options = dict(options)
    half_history = options.pop("half_history", False)
    for keys, capacity in random_traces(count=20, seed=9):
        if half_history:
            options["history_size"] = capacity // 2
        for seed in (1, 2, 3):
            lecar = LeCaR(capacity, seed=seed, **options)
            naive = NaiveLeCaR(capacity, seed=seed, **options)
            assert run(lecar, keys) == run(naive, keys)
            assert lecar.w_lru == pytest.approx(naive.w_lru, abs=1e-12)



# --- LFU min tracking: scan and heap must give identical behaviour ---

@pytest.mark.parametrize("options", [{}, {"half_history": True, **LCS}, {"weight_clamp": 0.0},
                                     {"same_victim_history": True}])
def test_heap_and_scan_min_tracking_give_identical_hits(options):
    options = dict(options)
    half_history = options.pop("half_history", False)
    for keys, capacity in random_traces(count=30, seed=21):
        if half_history:
            options["history_size"] = capacity // 2
        for seed in (1, 2):
            scan = LeCaR(capacity, seed=seed, lfu_min_tracking="scan", **options)
            heap = LeCaR(capacity, seed=seed, lfu_min_tracking="heap", **options)
            assert run(scan, keys) == run(heap, keys)
            assert scan.w_lru == heap.w_lru
            assert scan.lfu.n_min_scans == heap.lfu.n_min_scans    # same lookups, other method


def test_heap_and_scan_identical_on_crafted_traces():
    for keys, capacity in [(lfu_favouring_trace(), 4), (lru_favouring_trace(), 6)]:
        for seed in range(1, 4):
            scan = LeCaR(capacity, seed=seed)
            heap = LeCaR(capacity, seed=seed, lfu_min_tracking="heap")
            assert run(scan, keys) == run(heap, keys)
