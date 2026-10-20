"""
lru.py - Least Recently Used.

Data structure: collections.OrderedDict, which is a hash map plus a doubly linked list.
The list order is recency: first = least recently used, last = most recently used.

Time:  O(1) per request (dict lookup, move_to_end, popitem(last=False) are all O(1)).
Space: O(C) - one entry per resident key, C = capacity.
"""

from collections import OrderedDict

from src.policies.base import Policy


class LRU(Policy):
    name = "LRU"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.order = OrderedDict()    # key -> None, ordered from LRU to MRU

    def request(self, key):
        if key in self.order:
            self.order.move_to_end(key)    # now the most recently used
            return True
        if len(self.order) >= self.capacity:
            self.order.popitem(last=False)    # evict the least recently used
        self.order[key] = None
        return False

    # --- helpers, also used by LeCaR, which keeps an LRU ordering of its own cache ---

    def victim(self):
        """The key LRU would evict next (the least recently used)."""
        return next(iter(self.order))

    def touch(self, key):
        """Mark a resident key as most recently used."""
        self.order.move_to_end(key)

    def insert(self, key):
        """Add a key as most recently used. Does not check capacity."""
        self.order[key] = None

    def remove(self, key):
        del self.order[key]

    # --- interface ---

    def __len__(self):
        return len(self.order)

    def __contains__(self, key):
        return key in self.order

    def metadata_entries(self):
        return len(self.order)
