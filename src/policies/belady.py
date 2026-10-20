"""
belady.py - Belady's OPT (MIN): evict the resident key whose next use is farthest away.

Offline only: it needs the whole future request sequence, so it is a lower bound on the
miss ratio, not a deployable policy. On a miss the key is always admitted (demand paging,
same as every other policy here); a key never used again simply becomes the next victim.

Preprocessing: one backward pass over the trace builds next_use[i] = index of the next
request for the same key after position i, or N (= "never") if there is none.

Eviction: a max-heap of (next use, key) with lazy deletion. Python's heapq is a min-heap,
so we store -next_use. A hit pushes a fresh entry and leaves the old one in the heap; an
entry is stale if it no longer matches resident[key]. Stale entries are skipped when
popped, and the heap is rebuilt from `resident` when it grows past 2C entries, so its
size stays O(C).

Time:  O(N) preprocessing; O(log C) amortised per request (heap push/pop; each rebuild
       costs O(C) and happens at most once every C requests).
Space: O(N) for next_use (int64 numpy array, 8 bytes per request) + O(C) for the cache
       and heap.
"""

import heapq

import numpy as np

from src.policies.base import Policy


def compute_next_use(keys):
    """next_use[i] = smallest j > i with keys[j] == keys[i], or len(keys) if none."""
    n = len(keys)
    next_use = np.empty(n, dtype=np.int64)
    last_seen = {}    # key -> index of its nearest request to the right of i
    for i in range(n - 1, -1, -1):
        key = keys[i]
        next_use[i] = last_seen.get(key, n)
        last_seen[key] = i
    return next_use


class Belady(Policy):
    name = "Belady"

    def __init__(self, capacity, keys, next_use=None):
        """keys: the full request sequence that request() will then be called with, in order.
        next_use: compute_next_use(keys), if already computed (it depends only on the trace,
        so one array can be shared by every cache size)."""
        super().__init__(capacity)
        self.keys = keys
        self.next_use = next_use if next_use is not None else compute_next_use(keys)
        self.position = 0     # index of the request being processed
        self.resident = {}    # key -> its current next use
        self.heap = []        # entries (-next_use, key)

    def request(self, key):
        i = self.position
        if key != self.keys[i]:
            raise ValueError(f"Belady expected key {self.keys[i]!r} at position {i}, got {key!r}")
        self.position += 1
        next_use = int(self.next_use[i])

        if key in self.resident:
            hit = True
        else:
            hit = False
            if len(self.resident) >= self.capacity:
                self._evict()

        self.resident[key] = next_use
        heapq.heappush(self.heap, (-next_use, key))
        if len(self.heap) > 2 * self.capacity:
            self._rebuild_heap()
        return hit

    def _evict(self):
        while True:
            neg_next_use, key = heapq.heappop(self.heap)
            if self.resident.get(key) == -neg_next_use:    # entry is current, not stale
                del self.resident[key]
                return

    def _rebuild_heap(self):
        self.heap = [(-nu, k) for k, nu in self.resident.items()]
        heapq.heapify(self.heap)

    # --- interface ---

    def __len__(self):
        return len(self.resident)

    def __contains__(self, key):
        return key in self.resident

    def metadata_entries(self):
        # online structures only; the O(N) next_use array is reported separately
        return len(self.resident) + len(self.heap)

    def preprocessing_entries(self):
        return len(self.next_use)
