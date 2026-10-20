import random

import pytest

from src.policies.lfu import LFU
from tests.reference import brute_force_lfu
from tests.util import hit_string, random_traces, run


def test_hand_trace():
    # capacity 2
    # A M {A:1}   A H {A:2}   B M {A:2,B:1}   C M evict B (count 1) {A:2,C:1}
    # A H {A:3}   B M evict C {A:3,B:1}       C M evict B {A:3,C:1}   A H
    hits = run(LFU(2), ["A", "A", "B", "C", "A", "B", "C", "A"])
    assert hit_string(hits) == "MHMMHMMH"


def test_tie_break_is_lru_within_bucket():
    lfu = LFU(3)
    run(lfu, [1, 2, 3])        # all count 1, inserted in order 1, 2, 3
    assert lfu.victim() == 1
    run(lfu, [1])              # 1 moves to count 2; among count-1 keys, 2 is oldest
    assert lfu.victim() == 2
    run(lfu, [4])              # evicts 2
    assert 2 not in lfu and 1 in lfu and 3 in lfu and 4 in lfu


def test_returning_key_starts_at_count_one():
    lfu = LFU(1)
    run(lfu, [1, 1, 1, 2])     # 1 evicted with count 3
    run(lfu, [1])              # 1 back with count 1, not 4
    assert lfu.frequency(1) == 1


def test_frequent_key_survives_a_scan():
    # LFU's strength over LRU: a popular key is not flushed by one-time keys.
    lfu = LFU(2)
    run(lfu, ["hot"] * 5)
    run(lfu, list(range(10)))
    assert "hot" in lfu


def test_capacity_invariant_on_random_traces():
    for keys, capacity in random_traces():
        run(LFU(capacity), keys)


def test_matches_brute_force_reference():
    for keys, capacity in random_traces(seed=1):
        assert run(LFU(capacity), keys) == brute_force_lfu(capacity, keys)


@pytest.mark.parametrize("min_tracking", ["scan", "heap"])
def test_victim_correct_after_arbitrary_removals(min_tracking):
    # The LeCaR usage pattern: remove any key (LeCaR's LRU victim), touch, insert with
    # count 1 or a restored count.
    rng = random.Random(7)
    lfu = LFU(1000, min_tracking)
    for step in range(3000):
        action = rng.random()
        if action < 0.5 or len(lfu) == 0:
            key = rng.randrange(200)
            if key in lfu:
                lfu.touch(key)
            else:
                lfu.insert(key, rng.choice([1, 1, rng.randint(2, 6)]))    # LeCaR restore_freq
        else:
            lfu.remove(rng.choice(list(lfu.freq_of)))    # may empty the min bucket
        if len(lfu):
            expected_min = min(lfu.freq_of.values())
            assert lfu.frequency(lfu.victim()) == expected_min


def test_heap_mode_matches_brute_force_reference():
    for keys, capacity in random_traces(seed=1):
        assert run(LFU(capacity, "heap"), keys) == brute_force_lfu(capacity, keys)


def test_heap_and_scan_pick_the_same_victims_and_heap_stays_bounded():
    rng = random.Random(11)
    scan, heap = LFU(500, "scan"), LFU(500, "heap")
    most_buckets = 0
    for step in range(20000):
        key = rng.randrange(400)
        if key in scan:
            if rng.random() < 0.3:
                scan.remove(key)
                heap.remove(key)
            else:
                scan.touch(key)
                heap.touch(key)
        else:
            freq = rng.choice([1, 1, rng.randint(2, 40)])
            scan.insert(key, freq)
            heap.insert(key, freq)
        if len(scan):
            assert scan.victim() == heap.victim()
        # The rebuild check runs at each push, and buckets can empty between pushes, so the
        # bound is in terms of the most buckets seen, which is at most the capacity.
        most_buckets = max(most_buckets, len(heap.buckets))
        assert len(heap.count_heap) <= 2 * most_buckets + 17 <= 2 * heap.capacity + 17
    assert heap.n_heap_rebuild_items > 0    # the rebuild path was exercised


def test_heap_pops_stale_counts_lazily():
    lfu = LFU(10, "heap")
    lfu.insert("a", 5)
    lfu.insert("b", 6)
    lfu.insert("c", 7)
    assert lfu.victim() == "a"
    lfu.remove("a")                     # bucket 5 emptied: its heap entry is now stale
    lfu.remove("b")                     # bucket 6 emptied too
    assert sorted(lfu.count_heap) == [5, 6, 7]    # nothing removed from the heap yet
    assert lfu.victim() == "c"          # lookup pops 5 and 6, finds 7
    assert lfu.n_heap_pops == 2 and lfu.count_heap[0] == 7


def test_unknown_min_tracking_is_rejected():
    with pytest.raises(ValueError):
        LFU(3, "sorted-list")
