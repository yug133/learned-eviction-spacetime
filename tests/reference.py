"""Brute-force reference implementations, used only to check the real policies.

They are deliberately naive (full scans, plain lists) and share no code with src/, so a
bug in an OrderedDict / bucket / heap structure would show up as a disagreement.
"""

import math
import random

SECONDS_PER_DAY = 86400


def brute_force_day(t, t0):
    """Day index by counting whole days, instead of integer division."""
    day = 0
    while t0 + (day + 1) * SECONDS_PER_DAY <= t:
        day += 1
    return day


def brute_force_lru(capacity, keys):
    """LRU by scanning for the smallest last-access time."""
    last_access = {}
    hits = []
    for t, key in enumerate(keys):
        hits.append(key in last_access)
        if key not in last_access and len(last_access) >= capacity:
            del last_access[min(last_access, key=last_access.get)]
        last_access[key] = t
    return hits


def brute_force_lfu(capacity, keys):
    """Reference LFU by full scan: evict min (count, time placed in current count)."""
    count = {}
    since = {}    # time the key reached its current count = LRU order within a bucket
    hits = []
    for t, key in enumerate(keys):
        if key in count:
            count[key] += 1
            since[key] = t
            hits.append(True)
            continue
        if len(count) >= capacity:
            victim = min(count, key=lambda k: (count[k], since[k]))
            del count[victim], since[victim]
        count[key] = 1
        since[key] = t
        hits.append(False)
    return hits


def brute_force_opt_misses(capacity, keys):
    """Exact minimum misses by trying every eviction choice (tiny traces only)."""
    best = None

    def search(i, cache, misses):
        nonlocal best
        if best is not None and misses >= best:
            return
        if i == len(keys):
            best = misses
            return
        key = keys[i]
        if key in cache:
            search(i + 1, cache, misses)
        elif len(cache) < capacity:
            search(i + 1, cache | {key}, misses + 1)
        else:
            for victim in cache:
                search(i + 1, (cache - {victim}) | {key}, misses + 1)

    search(0, frozenset(), 0)
    return best


class NaiveLeCaR:
    """LeCaR written straight from the spec with scans and lists.

    With weight_floor = 0 it uses the paper's Algorithm 2 form (reward the other expert),
    so agreement with src/policies/lecar.py (which penalises the blamed expert) also
    checks that the two forms are equivalent.
    """

    def __init__(self, capacity, seed=1, learning_rate=0.45, initial_weights=(0.5, 0.5),
                 history_size=None, restore_freq=True, weight_clamp=0.01, weight_floor=0.0,
                 same_victim_history=False):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.lr = learning_rate
        self.d = 0.005 ** (1 / capacity)
        self.w_lru, self.w_lfu = initial_weights
        self.history_size = capacity if history_size is None else history_size
        self.restore_freq = restore_freq
        self.clamp = weight_clamp
        self.floor = weight_floor
        self.same_victim_history = same_victim_history
        self.now = 0
        self.last_access = {}    # resident key -> time of last access
        self.count = {}          # resident key -> access count
        self.count_since = {}    # resident key -> time it reached its current count
        self.history = {"lru": [], "lfu": []}    # lists of [key, eviction time, count]

    def request(self, key):
        self.now += 1
        if key in self.count:
            self.last_access[key] = self.now
            self.count[key] += 1
            self.count_since[key] = self.now
            return True

        new_count = 1
        for blamed in ("lru", "lfu"):
            entries = [e for e in self.history[blamed] if e[0] == key]
            if entries:
                _, evicted_at, old_count = entries[0]
                self.history[blamed].remove(entries[0])
                self._update(blamed, self.d ** (self.now - evicted_at))
                if self.restore_freq:
                    new_count = old_count + 1
                break

        if len(self.count) >= self.capacity:
            lru_victim = min(self.count, key=lambda k: self.last_access[k])
            lfu_victim = min(self.count, key=lambda k: (self.count[k], self.count_since[k]))
            if lru_victim == lfu_victim and not self.same_victim_history:
                victim, expert = lru_victim, None
            elif self.rng.random() < self.w_lru:
                victim, expert = lru_victim, "lru"
            else:
                victim, expert = lfu_victim, "lfu"
            if expert is not None and self.history_size > 0:
                if len(self.history[expert]) >= self.history_size:
                    self.history[expert].pop(0)
                self.history[expert].append([victim, self.now, self.count[victim]])
            del self.last_access[victim], self.count[victim], self.count_since[victim]

        self.last_access[key] = self.now
        self.count[key] = new_count
        self.count_since[key] = self.now
        return False

    name = "NaiveLeCaR"

    def __len__(self):
        return len(self.count)

    def __contains__(self, key):
        return key in self.count

    def _update(self, blamed, reward):
        if self.floor == 0:
            # Paper Algorithm 2: the OTHER expert's weight is multiplied up.
            if blamed == "lru":
                self.w_lfu *= math.exp(self.lr * reward)
            else:
                self.w_lru *= math.exp(self.lr * reward)
        else:
            # libCacheSim LeCaR.c:757-760: blamed weight decays, floor added to both.
            if blamed == "lru":
                self.w_lru = self.w_lru * math.exp(-self.lr * reward) + self.floor
                self.w_lfu += self.floor
            else:
                self.w_lfu = self.w_lfu * math.exp(-self.lr * reward) + self.floor
                self.w_lru += self.floor
        total = self.w_lru + self.w_lfu
        self.w_lru /= total
        self.w_lfu = 1 - self.w_lru
        # SY lecar.py:146-149
        if self.clamp > 0 and self.w_lru >= 1 - self.clamp:
            self.w_lru, self.w_lfu = 1 - self.clamp, self.clamp
        elif self.clamp > 0 and self.w_lfu >= 1 - self.clamp:
            self.w_lru, self.w_lfu = self.clamp, 1 - self.clamp
