"""Shared helpers for the policy tests."""

import random


def run(policy, keys):
    """Feed keys to policy. Return the hit/miss list (True = hit).

    Also checks the invariant that the resident count never exceeds capacity.
    """
    hits = []
    for key in keys:
        hits.append(policy.request(key))
        assert len(policy) <= policy.capacity, f"{policy.name} holds {len(policy)} > {policy.capacity}"
        assert key in policy, f"{policy.name} did not admit {key} after requesting it"
    return hits


def hit_string(hits):
    """[True, False] -> 'HM', so expected sequences are easy to read in the tests."""
    return "".join("H" if h else "M" for h in hits)


def random_traces(count=30, length=400, seed=0):
    """Small random traces with a skewed key popularity, plus their cache sizes.

    Yields (keys, capacity). A mix of uniform and Zipf-like key choice so that recency and
    frequency both matter in some traces.
    """
    rng = random.Random(seed)
    for i in range(count):
        num_keys = rng.randint(5, 60)
        capacity = rng.randint(1, max(1, num_keys // 2))
        if i % 2 == 0:
            keys = [rng.randrange(num_keys) for _ in range(length)]
        else:
            weights = [1 / (k + 1) for k in range(num_keys)]
            keys = rng.choices(range(num_keys), weights=weights, k=length)
        yield keys, capacity


def lru_favouring_trace():
    """Working set of 6 keys that changes every 300 requests (old frequent keys go cold)."""
    rng = random.Random(0)
    keys = []
    for phase in range(30):
        keys += [phase * 100 + rng.randrange(6) for _ in range(300)]
    return keys


def lfu_favouring_trace():
    """3 hot keys (warmed up to count 2) interleaved with one-time keys; use capacity 4."""
    keys = [0, 1, 2] * 2
    for i in range(3000):
        keys += [i % 3, 1000 + i]
    return keys
