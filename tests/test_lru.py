from src.policies.lru import LRU
from tests.util import hit_string, random_traces, run


def test_hand_trace():
    # capacity 2:  A  B  A  C (evicts B)  B (evicts A)  A (evicts C)  B
    hits = run(LRU(2), ["A", "B", "A", "C", "B", "A", "B"])
    assert hit_string(hits) == "MMHMMMH"


def test_capacity_one():
    hits = run(LRU(1), [1, 1, 2, 1, 1])
    assert hit_string(hits) == "MHMMH"


def test_sequential_loop_larger_than_cache_always_misses():
    # The classic LRU worst case: looping over C+1 keys.
    hits = run(LRU(3), [1, 2, 3, 4] * 5)
    assert not any(hits)


def test_trace_that_fits_misses_once_per_key():
    keys = [1, 2, 3, 1, 2, 3, 3, 2, 1]
    hits = run(LRU(3), keys)
    assert hits.count(False) == 3


def test_victim_is_least_recently_used():
    lru = LRU(3)
    run(lru, [1, 2, 3, 1])
    assert lru.victim() == 2


def test_capacity_invariant_on_random_traces():
    for keys, capacity in random_traces():
        run(LRU(capacity), keys)    # run() asserts len <= capacity after every request


def test_metadata_entries_equals_resident_count():
    lru = LRU(4)
    run(lru, [1, 2, 3, 4, 5, 6])
    assert lru.metadata_entries() == 4
