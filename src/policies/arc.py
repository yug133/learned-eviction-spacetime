"""
arc.py - Adaptive Replacement Cache (Megiddo & Modha, FAST 2003), the standard algorithm.

Four LRU lists, each an OrderedDict (first = LRU end, last = MRU end):
  T1: resident keys seen once recently        B1: ghost keys recently evicted from T1
  T2: resident keys seen at least twice       B2: ghost keys recently evicted from T2
Ghost lists hold keys only (no data). |T1| + |T2| <= C and |T1|+|T2|+|B1|+|B2| <= 2C.

p is the adaptive target size for T1. A hit in ghost list B1 means T1 was too small, so p
grows; a hit in B2 means T2 was too small, so p shrinks. REPLACE then evicts from T1 or T2
depending on whether |T1| is above or below p.

The case numbers below follow Fig. 4 of the paper.

Time:  O(1) per request (constant number of dict/linked-list operations).
Space: O(2C) - up to C resident keys plus up to C ghost keys.
"""

from collections import OrderedDict

from src.policies.base import Policy


class ARC(Policy):
    name = "ARC"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.p = 0.0    # target size of T1, between 0 and C

    def request(self, key):
        c = self.capacity

        # Case I: hit in T1 or T2 -> move to MRU of T2.
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True

        # Case II: miss, but key is in ghost list B1 -> favour recency: grow p.
        if key in self.b1:
            delta = 1 if len(self.b1) >= len(self.b2) else len(self.b2) / len(self.b1)
            self.p = min(self.p + delta, c)
            self._replace(key)
            del self.b1[key]
            self.t2[key] = None
            return False

        # Case III: miss, but key is in ghost list B2 -> favour frequency: shrink p.
        if key in self.b2:
            delta = 1 if len(self.b2) >= len(self.b1) else len(self.b1) / len(self.b2)
            self.p = max(self.p - delta, 0)
            self._replace(key)
            del self.b2[key]
            self.t2[key] = None
            return False

        # Case IV: complete miss (not in any list).
        l1 = len(self.t1) + len(self.b1)
        total = l1 + len(self.t2) + len(self.b2)
        if l1 == c:
            # Case A: L1 = T1 + B1 is full.
            if len(self.t1) < c:
                self.b1.popitem(last=False)    # drop LRU ghost of B1
                self._replace(key)
            else:
                self.t1.popitem(last=False)    # B1 is empty: evict LRU of T1 outright
        elif total >= c:
            # Case B: L1 < C and the directory holds at least C keys.
            if total == 2 * c:
                self.b2.popitem(last=False)    # drop LRU ghost of B2
            self._replace(key)
        self.t1[key] = None
        return False

    def _replace(self, key):
        """Evict one resident key into its ghost list: from T1 if T1 is over its target p,
        otherwise from T2."""
        t1_size = len(self.t1)
        if t1_size >= 1 and ((key in self.b2 and t1_size == self.p) or t1_size > self.p):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = None
        else:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = None

    # --- interface ---

    def __len__(self):
        return len(self.t1) + len(self.t2)

    def __contains__(self, key):
        return key in self.t1 or key in self.t2

    def metadata_entries(self):
        return len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
