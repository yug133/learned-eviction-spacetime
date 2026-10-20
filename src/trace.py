"""
trace.py - load a sampled trace (written by sample.py) into numpy arrays.

Each line is "time id [size]". We keep time and id; size is ignored because the cache
capacity is measured in objects (unit-size objects), see CLAUDE.md section 3.

Note: always import this as `src.trace`. A bare `import trace` gets Python's standard
library module of the same name.
"""

from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86400


@dataclass
class Trace:
    times: np.ndarray    # int64, one entry per request
    ids: np.ndarray      # int64 object ids, one entry per request
    days: np.ndarray     # int64 day index, 0 = first day

    def __len__(self):
        return len(self.ids)

    def unique_objects(self):
        return len(np.unique(self.ids))

    def num_days(self):
        return int(self.days[-1]) + 1 if len(self.days) else 0


def compute_days(times, t0=None):
    """Day index of every request: (time - t0) // 86400. t0 defaults to the first timestamp.

    Pass the FULL trace's first timestamp as t0 so day boundaries match across samples.
    """
    if t0 is None:
        t0 = times[0] if len(times) else 0
    return (times - t0) // SECONDS_PER_DAY


def load_trace(path, t0=None):
    """Read a sampled trace file into a Trace."""
    try:
        data = np.loadtxt(path, dtype=np.int64, usecols=(0, 1), ndmin=2)
        times = data[:, 0].copy()
        ids = data[:, 1].copy()
    except ValueError:
        # Non-integer ids: map each distinct id string to a small integer, in first-seen order.
        times, ids = _load_with_id_mapping(path)

    if len(times) > 1 and np.any(np.diff(times) < 0):
        raise ValueError(f"{path}: timestamps are not in non-decreasing order")
    return Trace(times=times, ids=ids, days=compute_days(times, t0))


def _load_with_id_mapping(path):
    id_to_int = {}
    times = []
    ids = []
    with open(path, "rb") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(int(parts[0]))
            ids.append(id_to_int.setdefault(parts[1], len(id_to_int)))
    return np.array(times, dtype=np.int64), np.array(ids, dtype=np.int64)
