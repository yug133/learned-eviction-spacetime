"""
base.py - the interface every eviction policy implements.

capacity is a number of objects (all objects have unit size).
"""


class Policy:
    name = "base"

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity

    def request(self, key):
        """Process one request. Return True on a hit, False on a miss.

        On a miss the key is always admitted, evicting one resident key if the cache is full.
        """
        raise NotImplementedError

    def __len__(self):
        """Number of keys currently resident in the cache."""
        raise NotImplementedError

    def __contains__(self, key):
        """True if key is currently resident in the cache."""
        raise NotImplementedError

    def metadata_entries(self):
        """Entries held in all of the policy's data structures right now
        (cache + ghost/history lists + frequency structures). Used for the memory metric."""
        raise NotImplementedError
