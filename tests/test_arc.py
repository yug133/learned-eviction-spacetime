from src.policies.arc import ARC
from src.policies.belady import Belady
from src.policies.lru import LRU
from tests.util import hit_string, random_traces, run


def test_hand_trace():
    # capacity 2, worked through Fig. 4 of the paper:
    # A M  T1=[A]
    # B M  T1=[A,B]
    # A H  T1=[B] T2=[A]
    # C M  Case IV-B, REPLACE: |T1|=1 > p=0 -> B to B1.     T1=[C] T2=[A] B1=[B]
    # B M  Case II: p=1. REPLACE: |T1|=1 not > 1 -> A to B2. T1=[C] T2=[B] B2=[A]
    # A M  Case III: p=0. REPLACE: |T1|=1 > 0 -> C to B1.    T2=[B,A] B1=[C]
    # B H
    arc = ARC(2)
    hits = run(arc, ["A", "B", "A", "C", "B", "A", "B"])
    assert hit_string(hits) == "MMHMMMH"
    assert list(arc.t2) == ["A", "B"] and list(arc.b1) == ["C"] and arc.p == 0


def test_p_adapts_on_ghost_hits():
    arc = ARC(2)
    run(arc, ["A", "B", "A", "C", "B"])    # the B request is a B1 ghost hit
    assert arc.p == 1


def test_trace_that_fits_misses_once_per_key():
    keys = [1, 2, 3, 1, 1, 2, 4, 3, 4, 2, 1]
    hits = run(ARC(4), keys)
    assert hits.count(False) == len(set(keys))


def test_trace_that_fits_random():
    for keys, _ in random_traces(seed=3):
        unique = len(set(keys))
        assert run(ARC(unique), keys).count(False) == unique


def test_scan_resistance():
    # Keys seen twice live in T2 and survive a one-pass scan; LRU loses them.
    keys = [1, 2, 1, 2] + list(range(10, 30)) + [1, 2]
    assert run(ARC(4), keys)[-2:] == [True, True]
    assert run(LRU(4), keys)[-2:] == [False, False]


def test_list_size_invariants_on_random_traces():
    for keys, capacity in random_traces(seed=4):
        arc = ARC(capacity)
        for key in keys:
            arc.request(key)
            assert len(arc.t1) + len(arc.t2) <= capacity
            assert len(arc.t1) + len(arc.b1) <= capacity
            assert arc.metadata_entries() <= 2 * capacity
            assert 0 <= arc.p <= capacity
            lists = [arc.t1, arc.t2, arc.b1, arc.b2]
            assert sum(len(x) for x in lists) == len(set().union(*lists))    # disjoint


def test_belady_never_worse():
    for keys, capacity in random_traces(count=60, seed=2):
        belady = run(Belady(capacity, keys), keys).count(False)
        assert belady <= run(ARC(capacity), keys).count(False)
