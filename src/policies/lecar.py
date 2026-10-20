"""
lecar.py - LeCaR: Learning Cache Replacement (Vietri et al., HotStorage 2018).

Two experts, LRU and LFU, kept over the SAME cache contents. On each eviction LeCaR picks
one expert at random with probability (w_lru, w_lfu) and evicts that expert's victim.
The evicted key goes into that expert's eviction history (a FIFO of ghost keys).

Learning (regret minimisation): a miss on a key found in H_lru means LRU's eviction is
regretted (and symmetrically for H_lfu). With reward r = d**t:
  Paper, Algorithm 2:  w_lfu = w_lfu * exp(lr * r), then renormalise   (reward the other)
  SY and LCS:          w_lru = w_lru * exp(-lr * r), then renormalise  (penalise the blamed)
After renormalising the two are identical: both multiply w_lru / w_lfu by exp(-lr * r).
We use the penalise form, which is the one the LCS weight floor is defined on.

Default configuration = the authors' own code (SY below). lecar_lcs() builds the
libCacheSim configuration, used as an ablation.

Sources
-------
  Paper  Vietri et al., "Driving Cache Replacement with ML-based LeCaR", HotStorage 2018,
         p. 3 (Algorithms 1 and 2 and the text beside them).
  SY     github.com/sylab/LeCaR, code/alg/lecar.py, commit e4044de34f84157f765de487642a5803c22686be
         (SyLab, FIU: the paper's lab; the repo README cites the paper).
  CU     github.com/sylab/cacheus, code/algs/lecar.py, commit 1eec63ce166502be33ddd1f35bc041ed73a24f4d
         (same lab, CACHEUS, FAST 2021).
  LCS    github.com/1a1a11a/libCacheSim, libCacheSim/cache/eviction/LeCaR.c (+ cache/cache.c),
         commit 5b4dd398707f7c9bb0c738a4c417cb78f1155d95 (simulator used by S3-FIFO [5]).

  Decision               Ours (default)       Paper p.3        SY lines          CU lines   LCS lines
  ---------------------  -------------------  ---------------  ----------------  ---------  ------------------
  lr, d, initial w       0.45, 0.005**(1/C),  text, Alg. 2     49, 52, 55        58, 61, 64 130-132
                         0.5 / 0.5
  t in d**t              now - eviction time  Alg. 2 "time     124, 160-161,     153,       243, 253, 501;
                         (requests)           spent in         167-168, 184      190-191    cache.c 232
                                              History"
  History size (each)    C                    unclear (*1)     44: C             53: C//2   518, 532: C/2
  Frequency on return    stored count + 1     not stated       155, 158, 166     188, 196   249, 259, 321: 1
  Same victim            no history           Alg. 1: drawn    114-115, 84-85    143-144    471-473, 485-487,
                                              expert's hist.                                541-544
  Weight bound           clamp [0.01, 0.99]   none             146-149           176-179    757-760: +1e-10
                         after each update                                                  floor (*2)
  Random draw            rng.random() < w_lru "w/ prob"        101-102, 110 (*3) -          440, 489 (*4)
  LFU tie-break          FIFO in min bucket   not stated       20-23: key id     24-27: id  640-645, 735: FIFO

  (*1) Paper: "The number of entries in the history (as with ARC) is equal to the number
       of entries in the cache"; Alg. 1 tests each history for "full" without defining it.
  (*2) LCS adds 1e-10 to both weights, but its denominator adds 2e-10 more, so its weights
       sum to 1 - ~1e-10. We renormalise to exactly 1.
  (*3) SY draws before checking for a same victim (so every eviction consumes a random
       number); we draw only when the victims differ. Same distribution, different stream.
  (*4) LCS draws (next_rand() % 100) / 100, in steps of 0.01, so any weight > 0 is chosen
       at least 1% of the time. Not copied.

Deviations from SY in the default configuration (all documented, none affects the rule):
  - LFU ties: FIFO within the min-count bucket (as LCS), not SY's key-id order. Key-id
    order is arbitrary (it depends on how ids were assigned), FIFO is the usual LFU rule.
  - Weights are float64; SY uses numpy float32 (lines 58-59, 142).
  - The clamp is not applied to initial_weights, only after an update (as in SY).
  - Random stream: Python random.Random(seed), seeds 1-5; SY uses np.random.seed(123).

Counters (named after LCS's n_hit_lru_history / n_hit_lfu_history, lines 49-50):
  n_hit_lru_history / n_hit_lfu_history  misses on a key in H_lru / H_lfu (regrets)
  n_evict_lru / n_evict_lfu              evictions into H_lru / H_lfu
  n_same_victim                          evictions where both experts nominated one key
  n_history_dropped                      history entries dropped because it was full
  lfu.n_min_scans / lfu.n_buckets_scanned  O(F) min-count scans (see Time below)

Time:  O(1) per request, except for one case. A restored key (restore_freq=True, the
       default) is inserted with count > 1. If the eviction just before it emptied the
       min-count bucket, the min count is unknown and the next eviction finds it with a
       scan over the distinct counts: O(F) worst case, F <= C distinct counts among the
       resident keys. lfu.n_min_scans counts how often that happens, so the amortised
       cost can be reported. A heap of counts would bound this at O(log C), but it adds
       a heap push to most count changes, i.e. to the hit path. Both are available:
       lfu_min_tracking="heap" (default since 21 Oct 2026) or "scan" (min-heap of distinct
       counts with lazy deletion, amortised O(log F) per lookup; see lfu.py). They give
       identical hits.
       (LCS also scans: update_LFU_min_freq, LeCaR.c lines 646-655.)
       Measured, 0.1% Wikipedia sample, seed 1, median of 3 alternating runs, 20 Oct 2026
       (experiments/time_min_tracking.py -> results/wiki_0.001/min_tracking.csv):
         cache    scan ns/req  buckets/req   heap ns/req  heap ops/req   heap slower
         0.1%          1,943         0.28         2,102          0.68          8.2%
         1%            1,833        10.96         1,969          1.41          7.4%
         10%           1,308         9.45         1,399          1.17          7.0%
       On that sample the scan wins: ~11 bucket visits per request are cheap because
       min() over the dict keys runs in C, while each heap push is a Python-level call on
       bucket creation (often on a hit).
       Re-measured on the 1% sample (the paper's results), seed 1, 21 Oct 2026
       (results/wiki_0.01/min_tracking.csv): F grows with the cache, and so does the scan:
         cache    scan ns/req  buckets/req   heap ns/req  heap ops/req   heap vs scan
         0.1%          1,745         0.81         1,936          0.97        +11%
         1%            2,216        41.27         1,878          1.34        -15%
         10%           1,656        19.24         1,593          1.20        -3.8%
       Crossover: the heap is faster once the scan visits ~20+ buckets per request, so it
       is the DEFAULT since 21 Oct 2026 (it loses ~11% at the 0.1% size, where scans are
       rare). CPython only: a C implementation may rank them differently (not measured).
       Worst cases: scan O(F) per lookup; heap O(log F) amortised per lookup plus
       O(log F) per bucket creation.
       With restore_freq=False (lecar_lcs) every insert has count 1: always O(1).
Space: O(C) for the cache, kept in two structures (LRU order + LFU buckets), plus up to
       2C history entries (C per expert; C in total for lecar_lcs). metadata_entries()
       counts exactly.
"""

import math
import random
from collections import OrderedDict

from src.policies.base import Policy
from src.policies.lfu import LFU
from src.policies.lru import LRU


class LeCaR(Policy):
    name = "LeCaR"

    def __init__(self, capacity, seed=1, learning_rate=0.45, discount=None,
                 initial_weights=(0.5, 0.5), history_size=None, restore_freq=True,
                 weight_clamp=0.01, weight_floor=0.0, same_victim_history=False,
                 lfu_min_tracking="heap", log_every=0):
        """
        seed                : seed of the random generator used for the expert draw
        discount            : d in the reward d**t; default 0.005 ** (1 / capacity)
        initial_weights     : (w_lru, w_lfu). With restore_freq=False, weight_clamp=0 and
                              weight_floor=0, (1, 0) and (0, 1) freeze LeCaR as pure LRU /
                              pure LFU, because a zero weight stays zero.
        history_size        : cap on EACH history; default capacity (SY)
        restore_freq        : a key returning from a history gets (stored count + 1)
                              instead of 1 (SY, CU). False = LCS.
        weight_clamp        : after each update, a weight >= 1 - clamp is set to 1 - clamp
                              and the other to clamp (SY: 0.01). 0 = off.
        weight_floor        : added to both weights on each update (LCS: 1e-10). 0 = off.
        same_victim_history : if True, a victim both experts agree on still goes into the
                              drawn expert's history (paper Alg. 1 read literally)
        lfu_min_tracking    : "heap" (default) or "scan": how the LFU structure finds its
                              min count when it is unknown (same results, different cost;
                              lfu.py and the measurements in the Time section above)
        log_every           : if > 0, append (request index, w_lru) to weight_log every
                              log_every requests, for the weight-over-time plot
        """
        super().__init__(capacity)
        self.rng = random.Random(seed)
        self.learning_rate = learning_rate
        self.discount = discount if discount is not None else 0.005 ** (1 / capacity)
        self.w_lru, self.w_lfu = initial_weights
        self.history_size = history_size if history_size is not None else capacity
        self.restore_freq = restore_freq
        self.weight_clamp = weight_clamp
        self.weight_floor = weight_floor
        self.same_victim_history = same_victim_history
        self.log_every = log_every

        self.lru = LRU(capacity)    # recency order of the cache (used as a data structure)
        self.lfu = LFU(capacity, lfu_min_tracking)    # frequency buckets of the same keys
        # key -> (eviction time, count at eviction), oldest first
        self.h_lru = OrderedDict()  # keys evicted by the LRU expert
        self.h_lfu = OrderedDict()  # keys evicted by the LFU expert
        self.now = 0                # request counter (virtual time)
        self.weight_log = []

        self.n_hit_lru_history = 0
        self.n_hit_lfu_history = 0
        self.n_evict_lru = 0
        self.n_evict_lfu = 0
        self.n_same_victim = 0
        self.n_history_dropped = 0

    def request(self, key):
        self.now += 1
        if self.log_every and self.now % self.log_every == 0:
            self.weight_log.append((self.now, self.w_lru))

        if key in self.lru:    # hit: update both experts' views of the cache
            self.lru.touch(key)
            self.lfu.touch(key)
            return True

        # Miss. If the key was evicted by an expert recently, that eviction is regretted.
        freq = 1
        if key in self.h_lru:
            evicted_at, old_freq = self.h_lru.pop(key)
            self.n_hit_lru_history += 1
            self._update_weights("lru", self.now - evicted_at)
            if self.restore_freq:
                freq = old_freq + 1
        elif key in self.h_lfu:
            evicted_at, old_freq = self.h_lfu.pop(key)
            self.n_hit_lfu_history += 1
            self._update_weights("lfu", self.now - evicted_at)
            if self.restore_freq:
                freq = old_freq + 1

        if len(self.lru) >= self.capacity:
            self._evict()
        self.lru.insert(key)
        self.lfu.insert(key, freq)
        return False

    def _update_weights(self, blamed, t):
        """Regret update: blamed is "lru" or "lfu", the expert whose eviction was wrong.
        Penalise it (SY lines 141-144), add the optional floor (LCS lines 757-760),
        renormalise, then apply the optional clamp (SY lines 146-149)."""
        penalty = math.exp(-self.learning_rate * self.discount ** t)
        floor = self.weight_floor
        if blamed == "lru":
            w_lru, w_lfu = self.w_lru * penalty + floor, self.w_lfu + floor
        else:
            w_lru, w_lfu = self.w_lru + floor, self.w_lfu * penalty + floor
        self.w_lru = w_lru / (w_lru + w_lfu)
        self.w_lfu = 1 - self.w_lru

        c = self.weight_clamp
        if c > 0:
            if self.w_lru >= 1 - c:
                self.w_lru, self.w_lfu = 1 - c, c
            elif self.w_lfu >= 1 - c:
                self.w_lru, self.w_lfu = c, 1 - c

    def _evict(self):
        lru_victim = self.lru.victim()
        lfu_victim = self.lfu.victim()
        same = lru_victim == lfu_victim
        if same:
            self.n_same_victim += 1

        if same and not self.same_victim_history:
            victim, history = lru_victim, None    # both agree: nobody to blame
        elif self.rng.random() < self.w_lru:
            victim, history = lru_victim, self.h_lru
            self.n_evict_lru += 1
        else:
            victim, history = lfu_victim, self.h_lfu
            self.n_evict_lfu += 1

        freq = self.lfu.frequency(victim)
        self.lru.remove(victim)
        self.lfu.remove(victim)
        if history is not None and self.history_size > 0:
            if len(history) >= self.history_size:
                history.popitem(last=False)    # drop the oldest history entry (FIFO)
                self.n_history_dropped += 1
            history[victim] = (self.now, freq)

    # --- interface ---

    def __len__(self):
        return len(self.lru)

    def __contains__(self, key):
        return key in self.lru

    def metadata_entries(self):
        return (self.lru.metadata_entries() + self.lfu.metadata_entries()
                + len(self.h_lru) + len(self.h_lfu))


def lecar_lcs(capacity, seed=1, **overrides):
    """LeCaR configured as in libCacheSim (LCS): frequency reset to 1 on return, each
    history C // 2, 1e-10 weight floor, no clamp. The 0.01-step random draw of LCS is
    not reproduced (see the sources table). Keyword overrides are passed to LeCaR."""
    # lfu_min_tracking="scan": without restore_freq every insert has count 1, so the min
    # count is always known and never looked up; a heap would only add pushes. (LCS's own
    # LeCaR also scans: LeCaR.c lines 646-655.)
    options = dict(restore_freq=False, history_size=capacity // 2,
                   weight_floor=1e-10, weight_clamp=0.0, lfu_min_tracking="scan")
    options.update(overrides)
    policy = LeCaR(capacity, seed=seed, **options)
    policy.name = "LeCaR-LCS"
    return policy
