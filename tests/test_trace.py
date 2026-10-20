import random

import numpy as np
import pytest

from src.trace import compute_days, load_trace
from tests.reference import brute_force_day


def test_load_integer_trace(tmp_path):
    path = tmp_path / "t.tr"
    path.write_text("100 7 512\n100 8 10\n86499 7 512\n86500 9 1\n")
    trace = load_trace(path)
    assert trace.times.tolist() == [100, 100, 86499, 86500]
    assert trace.ids.tolist() == [7, 8, 7, 9]
    assert trace.days.tolist() == [0, 0, 0, 1]    # 86500 - 100 = 86400 -> day 1
    assert len(trace) == 4
    assert trace.unique_objects() == 3
    assert trace.num_days() == 2


def test_load_without_size_column(tmp_path):
    path = tmp_path / "t.tr"
    path.write_text("5 1\n6 2\n")
    trace = load_trace(path)
    assert trace.ids.tolist() == [1, 2]


def test_string_ids_are_mapped_to_integers(tmp_path):
    path = tmp_path / "t.tr"
    path.write_text("1 abc 3\n2 xyz 3\n3 abc 3\n")
    trace = load_trace(path)
    assert trace.ids.tolist() == [0, 1, 0]


def test_external_t0_shifts_day_boundaries():
    times = np.array([86000, 86500], dtype=np.int64)
    assert compute_days(times).tolist() == [0, 0]
    assert compute_days(times, t0=50).tolist() == [0, 1]
    assert [brute_force_day(t, 50) for t in times.tolist()] == [0, 1]
    assert [brute_force_day(t, 200) for t in times.tolist()] == [0, 0]    # the old wrong value


def test_compute_days_matches_counting_reference():
    rng = random.Random(0)
    t0 = 1_500_000_000
    times = sorted(t0 + rng.randrange(14 * 86400) for _ in range(2000))
    times += [t0 + 86399, t0 + 86400, t0 + 13 * 86400]    # exact boundaries
    times.sort()
    days = compute_days(np.array(times, dtype=np.int64), t0=t0)
    assert days.tolist() == [brute_force_day(t, t0) for t in times]


def test_unsorted_trace_is_rejected(tmp_path):
    path = tmp_path / "t.tr"
    path.write_text("10 1\n5 2\n")
    with pytest.raises(ValueError):
        load_trace(path)
