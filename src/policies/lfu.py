"""
lfu.py - Least Frequently Used, O(1) version with frequency buckets.

Data structures:
  freq_of : dict  key -> access count of that resident key
  buckets : dict  count -> OrderedDict of the keys with that count (oldest first)
  min_freq: the smallest count among resident keys (None = unknown, see below)

Tie-break: among keys with the smallest count, evict the one that has been in that bucket
longest, i.e. the least recently used among them (LRU within a bucket).

Frequency is counted only while a key is resident: an evicted key that comes back starts
again at 1 (no ghost frequency is remembered).

Time:  O(1) per request. Every removal is followed by insert() with count 1, which makes
       min_freq = 1 again, so it is always known when victim() needs it. This holds both
       here and in default LeCaR (which removes arbitrary keys: its LRU expert's victims).
       Exception: LeCaR with restore_freq=True inserts a returning key with a count > 1.
       If the removal just before emptied the min bucket, min_freq is unknown and the
       next victim() finds it with a scan over the distinct counts: O(F), F <= C.
       (libCacheSim's LeCaR has the same O(F) scan: update_LFU_min_freq, LeCaR.c:646.)
Space: O(C) - each resident key appears once in freq_of and once in one bucket.

Finding the min count when it is unknown: min_tracking chooses how.
  "scan" (default here): min(self.buckets), O(F) per lookup, no other cost. A stand-alone
          LFU never looks the min up (see Time above), so scan costs it nothing.
          LeCaR (restore_freq) does look it up and uses "heap" by default (lecar.py).
  "heap": a min-heap of distinct counts with lazy deletion. A count is pushed when its
          bucket is created (O(log F)); when a bucket empties nothing happens, its entry
          just becomes stale. A lookup pops stale entries off the top until the top count
          has a bucket. Stale entries below the top can pile up, so when a push leaves the
          heap with more than 2F + 16 entries (F = buckets at that moment) it is rebuilt
          from the live buckets (O(F), but only after at least F + 16 pushes: amortised
          O(1) per push). The heap never holds more than 2C + 17 entries. Lookup amortised
          O(log F); the price is a heap push on every bucket creation, which happens on
          hits too (a key moving to a count no other key has).
Both give exactly the same victims; only the cost differs. Counters: n_min_scans and
n_buckets_scanned (scan), n_heap_pushes, n_heap_pops and n_heap_rebuild_items (heap).
"""

import heapq
from collections import OrderedDict

from src.policies.base import Policy


class LFU(Policy):
    name = "LFU"

    def __init__(self, capacity, min_tracking="scan"):
        super().__init__(capacity)
        if min_tracking not in ("scan", "heap"):
            raise ValueError(f"min_tracking must be 'scan' or 'heap', got {min_tracking!r}")
        self.freq_of = {}
        self.buckets = {}
        self.min_freq = None
        self.use_heap = min_tracking == "heap"
        self.count_heap = []          # heap mode: distinct counts, possibly stale
        self.n_min_scans = 0          # times victim() had to look for the min count
        self.n_buckets_scanned = 0    # scan mode: total buckets looked at by those scans
        self.n_heap_pushes = 0        # heap mode: counts pushed (one per bucket creation)
        self.n_heap_pops = 0          # heap mode: stale counts popped off the top
        self.n_heap_rebuild_items = 0 # heap mode: entries written by rebuilds

    def request(self, key):
        if key in self.freq_of:
            self.touch(key)
            return True
        if len(self.freq_of) >= self.capacity:
            self.remove(self.victim())
        self.insert(key)
        return False

    # --- helpers, also used by LeCaR ---

    def victim(self):
        """The key LFU would evict next: lowest count, oldest within that count."""
        if self.min_freq is None:
            self.n_min_scans += 1
            if self.use_heap:
                while self.count_heap[0] not in self.buckets:    # stale: bucket emptied
                    heapq.heappop(self.count_heap)
                    self.n_heap_pops += 1
                self.min_freq = self.count_heap[0]
            else:
                self.n_buckets_scanned += len(self.buckets)
                self.min_freq = min(self.buckets)    # O(F) scan; see the docstring
        return next(iter(self.buckets[self.min_freq]))

    def touch(self, key):
        """Count one more access to a resident key: move it from bucket f to bucket f+1."""
        f = self.freq_of[key]
        self._remove_from_bucket(key, f)
        if self.min_freq == f and f not in self.buckets:
            self.min_freq = f + 1    # the key we just moved is now the minimum
        self._add_to_bucket(key, f + 1)

    def insert(self, key, freq=1):
        """Add a new resident key with the given count (default 1). Does not check capacity."""
        self._add_to_bucket(key, freq)
        if freq == 1 or len(self.freq_of) == 1:
            self.min_freq = freq    # smallest possible count, or the only key
        elif self.min_freq is not None and freq < self.min_freq:
            self.min_freq = freq
        # otherwise min_freq stays as it is (possibly None = unknown)

    def remove(self, key):
        f = self.freq_of[key]
        self._remove_from_bucket(key, f)
        if f == self.min_freq and f not in self.buckets:
            self.min_freq = None    # min bucket emptied; next victim() recomputes it

    def frequency(self, key):
        return self.freq_of[key]

    def _add_to_bucket(self, key, f):
        self.freq_of[key] = f
        if f not in self.buckets:
            self.buckets[f] = OrderedDict()
            if self.use_heap:
                self._push_count(f)
        self.buckets[f][key] = None

    def _push_count(self, f):
        """Heap mode: record that a bucket for count f now exists."""
        heapq.heappush(self.count_heap, f)
        self.n_heap_pushes += 1
        if len(self.count_heap) > 2 * len(self.buckets) + 16:
            self.count_heap = list(self.buckets)    # rebuild: drop every stale entry
            heapq.heapify(self.count_heap)
            self.n_heap_rebuild_items += len(self.count_heap)

    def _remove_from_bucket(self, key, f):
        del self.freq_of[key]
        bucket = self.buckets[f]
        del bucket[key]
        if not bucket:
            del self.buckets[f]

    # --- interface ---

    def __len__(self):
        return len(self.freq_of)

    def __contains__(self, key):
        return key in self.freq_of

    def metadata_entries(self):
        # one freq_of entry per key + one bucket entry per key + one entry per non-empty bucket
        # (+ the heap of counts in heap mode)
        return 2 * len(self.freq_of) + len(self.buckets) + len(self.count_heap)
