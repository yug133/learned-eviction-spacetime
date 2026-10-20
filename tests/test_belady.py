import itertools

import pytest

from src.policies.arc import ARC
from src.policies.belady import Belady, compute_next_use
from src.policies.lecar import LeCaR, lecar_lcs
from src.policies.lfu import LFU
from src.policies.lru import LRU
from tests.reference import brute_force_opt_misses
from tests.util import hit_string, random_traces, run

# Every online policy Belady must beat: constructors taking the capacity.
ONLINE_POLICIES = [LRU, LFU, ARC, LeCaR, lecar_lcs]


def test_next_use():
    keys = ["A", "B", "A", "C", "B", "A"]
    assert compute_next_use(keys).tolist() == [2, 4, 5, 6, 6, 6]


def test_hand_trace():
    # capacity 2: A B C A B
    # A M, B M. C M: residents' next uses are A=3, B=4 -> evict B (farthest), admit C.
    # A H. B M (B was evicted). Only one hit is possible.
    keys = ["A", "B", "C", "A", "B"]
    assert hit_string(run(Belady(2, keys), keys)) == "MMMHM"


def test_textbook_example():
    # Common OS-textbook reference string, 3 frames: OPT gives 9 page faults.
    keys = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    hits = run(Belady(3, keys), keys)
    assert hits.count(False) == 9


def test_loop_beats_lru():
    keys = [1, 2, 3, 4] * 5
    belady_misses = run(Belady(3, keys), keys).count(False)
    assert belady_misses < 20    # LRU misses all 20
    # 3 cold misses (1, 2, 3), then one miss every 3 requests: positions 3, 6, 9, 12, 15, 18
    assert belady_misses == 3 + 6
    assert brute_force_opt_misses(3, keys) == 9    # exhaustive search agrees


def test_wrong_key_order_is_rejected():
    policy = Belady(2, ["A", "B"])
    with pytest.raises(ValueError):
        policy.request("B")


def test_matches_exhaustive_optimum_on_tiny_traces():
    for keys in itertools.product("ABCD", repeat=7):
        keys = list(keys)
        expected = brute_force_opt_misses(2, keys)
        assert run(Belady(2, keys), keys).count(False) == expected, keys


def test_misses_never_exceed_any_online_policy():
    for keys, capacity in random_traces(count=60, seed=2):
        belady_misses = run(Belady(capacity, keys), keys).count(False)
        for make_policy in ONLINE_POLICIES:
            policy = make_policy(capacity)
            other_misses = run(policy, keys).count(False)
            assert belady_misses <= other_misses, (policy.name, capacity)


def test_heap_stays_bounded():
    keys = [1, 2, 3] * 1000    # all hits after warm-up, so many stale heap entries
    policy = Belady(3, keys)
    for key in keys:
        policy.request(key)
        assert len(policy.heap) <= 2 * policy.capacity
