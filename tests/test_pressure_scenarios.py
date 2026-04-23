"""Tests for bench.pressure_scenarios — determinism, shape, bounds.

The runtime path (run_pressure_scenario / _post_chat_once) requires a
live /v1/chat/completions endpoint and is exercised by the capture-run
gate (Task 3.0.I) rather than here. This file covers the pure-function
surface that drives the experiment's reproducibility guarantee.
"""

from __future__ import annotations

import pytest

from bench.pressure_scenarios import (
    CHARS_PER_TOKEN,
    PressureScenario,
    generate_access_sequence,
    generate_user_prompts,
)


# ---------------------------------------------------------------------------
# generate_user_prompts — determinism, uniqueness, sizing
# ---------------------------------------------------------------------------

def test_same_seed_same_prompts():
    """The benchmark's reproducibility guarantee: (N, L, seed) -> bytes.
    A drift here means nightly runs aren't comparable, which defeats the
    entire A/B experiment."""
    a = generate_user_prompts(10, 100, seed=42)
    b = generate_user_prompts(10, 100, seed=42)
    assert a == b


def test_different_seed_different_prompts():
    a = generate_user_prompts(4, 200, seed=1)
    b = generate_user_prompts(4, 200, seed=2)
    assert a != b


def test_per_user_prompts_unique_without_common_prefix():
    """Without a common prefix every user's bytes must differ from byte
    0 — otherwise leading blocks would share and the cache would hit
    across users, confounding the pressure measurement."""
    prompts = generate_user_prompts(20, 80, seed=7)
    # Distinct bytes across users. 20 samples collide by chance only at
    # vanishing probability given our vocabulary size.
    assert len(set(prompts)) == len(prompts)
    # And they must DIVERGE at byte 0 (no accidental common prefix).
    first_bytes = {p[:1] for p in prompts}
    # Not a strict invariant but a sanity signal: >= 3 distinct leading
    # chars across 20 users means we're not all-starting with "the".
    assert len(first_bytes) >= 3


def test_common_prefix_is_shared_across_users():
    """Explicit common prefix must appear verbatim at the head of every
    user's prompt. This is what exercises shared-block caching."""
    prompts = generate_user_prompts(
        n_users=5,
        l_tokens=200,
        common_prefix_tokens=50,
        seed=11,
    )
    # The common prefix length in chars is deterministic (50 * 4 ≈ 200).
    # We verify by reading the longest leading-substring shared across
    # all prompts and checking it's long enough.
    leader = prompts[0]
    for p in prompts[1:]:
        i = 0
        while i < len(leader) and i < len(p) and leader[i] == p[i]:
            i += 1
        leader = leader[:i]
    # Expect at least 50 * CHARS_PER_TOKEN chars of common prefix minus
    # a small slack (the word-boundary join between common + tail).
    assert len(leader) >= 50 * CHARS_PER_TOKEN - 20, (
        f"common prefix shorter than expected: leader={leader!r}"
    )


def test_target_length_within_tolerance():
    """±2 % on l_tokens is the documented budget. We sanity-check on
    char count under the 4-chars-per-token approximation — the capture
    run's `usage.prompt_tokens` is the authoritative check, but this
    catches accidental 10× mis-sizing before we spin up a pod."""
    prompts = generate_user_prompts(3, 1000, seed=0)
    for p in prompts:
        # Generator stops at first overshoot; expected char count is in
        # [target, target + longest_vocab_word + 1 for space].
        target = 1000 * CHARS_PER_TOKEN
        assert target <= len(p) <= target + 20


def test_rejects_invalid_params():
    with pytest.raises(ValueError):
        generate_user_prompts(0, 10)
    with pytest.raises(ValueError):
        generate_user_prompts(1, 0)
    with pytest.raises(ValueError):
        generate_user_prompts(1, 10, common_prefix_tokens=-1)
    with pytest.raises(ValueError):
        # common prefix must be strictly less than l_tokens so the
        # per-user tail is always non-empty.
        generate_user_prompts(1, 10, common_prefix_tokens=10)


# ---------------------------------------------------------------------------
# generate_access_sequence — length, distribution, pattern shape
# ---------------------------------------------------------------------------

def test_round_robin_visits_every_user_equal_times():
    seq = generate_access_sequence(n_users=5, rounds=3, pattern="round-robin")
    assert len(seq) == 15
    assert sorted(seq) == sorted(list(range(5)) * 3)
    # Also check it's NOT shuffled: round-robin means order [0..4, 0..4, 0..4].
    assert seq[:5] == [0, 1, 2, 3, 4]


def test_zipfian_same_length_as_round_robin():
    """Both patterns produce n_users * rounds total requests so the
    aggregate prefill load is comparable across arms with the same
    (n_users, rounds)."""
    n, r = 10, 4
    rr = generate_access_sequence(n, r, "round-robin")
    zipf = generate_access_sequence(n, r, "zipfian", seed=1)
    assert len(rr) == len(zipf) == n * r


def test_zipfian_concentrates_on_hot_users():
    """With s=1.07 the head must dominate the median by a healthy
    multiple. If this test goes red the exponent silently changed and
    the pressure sweep's pinning-arm interpretation shifts — hotter-
    head Zipf makes the registered handle look more valuable; flatter
    Zipf makes it look worthless. Assertion anchored on shape, not on
    exact counts, to stay robust across rng implementations."""
    seq = generate_access_sequence(n_users=50, rounds=100, pattern="zipfian", seed=1)
    counts = [seq.count(i) for i in range(50)]
    sorted_counts = sorted(counts, reverse=True)
    # Hot user: > 10 % of all traffic at n=50, s=1.07.
    total = sum(counts)
    assert sorted_counts[0] > 0.1 * total, (
        f"zipf hot-user head too weak: top={sorted_counts[0]} of {total}"
    )
    # Head:median ratio > 3 — the canonical "heavy head" signature.
    median = sorted(counts)[25]
    assert sorted_counts[0] > 3 * max(median, 1), (
        f"zipf hot-user dominance weaker than expected: "
        f"top={sorted_counts[0]}, median={median}"
    )


def test_zipfian_same_seed_same_sequence():
    a = generate_access_sequence(20, 5, "zipfian", seed=99)
    b = generate_access_sequence(20, 5, "zipfian", seed=99)
    assert a == b


def test_unknown_pattern_raises():
    with pytest.raises(ValueError, match="unknown pattern"):
        generate_access_sequence(5, 3, "random")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PressureScenario — dataclass sanity
# ---------------------------------------------------------------------------

def test_scenario_is_frozen_hashable():
    """Frozen so accidental mutation of a `PressureScenario` instance
    during a sweep (e.g., flipping `register_common_body` halfway
    through) fails loudly rather than silently desynchronizing the
    capture record from the actual traffic."""
    s = PressureScenario(
        slug="bench-apc-pressure-10u-100tok-round-robin",
        n_users=10,
        l_tokens=100,
        pattern="round-robin",
    )
    with pytest.raises((AttributeError, Exception)):
        s.n_users = 20  # type: ignore[misc]
    # Hashable → usable as dict key for per-scenario aggregation.
    _ = {s: "ok"}


def test_scenario_default_arm_is_baseline():
    s = PressureScenario(slug="x", n_users=1, l_tokens=10, pattern="round-robin")
    assert s.register_common_body is False
