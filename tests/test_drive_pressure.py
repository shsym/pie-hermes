"""Tests for bench.drive_pressure — preset matrix, cost preflight, adhoc parser.

Does NOT exercise the actual HTTP sweep (that's 3.0.I). Covers the
non-network-bound surface: sweep-preset invariants, cost estimator
monotonicity, and the `--scenario` parser.
"""

from __future__ import annotations

import pytest

from bench.drive_pressure import (
    PRESET_SWEEPS,
    _estimate_cost_usd,
    _parse_adhoc_scenario,
)
from bench.pressure_scenarios import PressureScenario


# ---------------------------------------------------------------------------
# Preset matrix — invariants
# ---------------------------------------------------------------------------


def test_every_preset_scenario_has_unique_slug():
    """Two scenarios in the same preset with the same slug would write
    over each other's capture directory. The preset author likely meant
    to vary the slug (e.g., forgot '-pinned'); fail loudly."""
    for preset, scenarios in PRESET_SWEEPS.items():
        slugs = [s.slug for s in scenarios]
        assert len(slugs) == len(set(slugs)), (
            f"preset {preset!r} has duplicate slugs: {slugs}"
        )


def test_phase30_initial_preset_matches_plan():
    """Locks the blessed sweep against the plan doc. If the plan
    changes, update both — but the test failing means someone edited
    one without the other."""
    initial = {s.slug: s for s in PRESET_SWEEPS["phase-3.0-initial"]}
    # From docs/plans/2026-04-23-phase-3.0-apc-pressure.md Task 3.0.I
    # sweep matrix (6 scenarios).
    expected_slugs = {
        "bench-apc-pressure-50u-4ktok-round-robin",
        "bench-apc-pressure-200u-4ktok-round-robin",
        "bench-apc-pressure-500u-4ktok-round-robin",
        "bench-apc-pressure-500u-4ktok-round-robin-pinned",
        "bench-apc-pressure-500u-4ktok-zipfian",
        "bench-apc-pressure-200u-16ktok-round-robin",
    }
    assert set(initial.keys()) == expected_slugs


def test_pinned_preset_entry_has_common_prefix_registered():
    """The A/B arm's value comes from pinning a SHARED common prefix —
    a pinned scenario with no common prefix is an empty A/B signal."""
    pinned = PRESET_SWEEPS["phase-3.0-initial"][3]
    assert "pinned" in pinned.slug
    assert pinned.register_common_body is True
    assert pinned.common_prefix_tokens > 0


# ---------------------------------------------------------------------------
# Cost preflight
# ---------------------------------------------------------------------------


def _scen(n, L, rounds=3):
    return PressureScenario(
        slug=f"t-{n}u-{L}L",
        n_users=n, l_tokens=L, rounds=rounds, pattern="round-robin",
    )


def test_cost_increases_with_users_rounds_and_tokens():
    """Sanity monotonicity: if any of (n_users, l_tokens, rounds) grows,
    cost grows. Guards against a regression where someone optimizes the
    estimator and accidentally makes it non-monotonic."""
    base = _estimate_cost_usd(
        [_scen(10, 1000)],
        hourly_usd=3.0, prefill_tps=30_000, seconds_per_request_overhead=0.05,
    )
    more_users = _estimate_cost_usd(
        [_scen(20, 1000)],
        hourly_usd=3.0, prefill_tps=30_000, seconds_per_request_overhead=0.05,
    )
    more_tokens = _estimate_cost_usd(
        [_scen(10, 2000)],
        hourly_usd=3.0, prefill_tps=30_000, seconds_per_request_overhead=0.05,
    )
    more_rounds = _estimate_cost_usd(
        [_scen(10, 1000, rounds=6)],
        hourly_usd=3.0, prefill_tps=30_000, seconds_per_request_overhead=0.05,
    )
    assert more_users > base
    assert more_tokens > base
    assert more_rounds > base


def test_cost_is_below_preset_budget():
    """The plan doc caps the phase-3.0-initial preset at ≤$8 with a
    16× headroom warning. This test anchors the estimate under that
    ceiling so a casual 'add one more scenario' to the preset cannot
    silently blow through the approved budget."""
    est = _estimate_cost_usd(
        PRESET_SWEEPS["phase-3.0-initial"],
        hourly_usd=3.0, prefill_tps=30_000, seconds_per_request_overhead=0.05,
    )
    assert est <= 8.0, (
        f"phase-3.0-initial preset estimate ${est:.2f} exceeds $8 plan budget; "
        "either shrink the sweep or re-open the budget conversation"
    )


# ---------------------------------------------------------------------------
# --scenario parser
# ---------------------------------------------------------------------------


def test_adhoc_parser_minimum_form():
    s = _parse_adhoc_scenario("100,2000,round-robin")
    assert s.n_users == 100
    assert s.l_tokens == 2000
    assert s.pattern == "round-robin"
    assert s.register_common_body is False
    assert s.rounds == 3


def test_adhoc_parser_pinned_and_custom_rounds():
    s = _parse_adhoc_scenario("50,1000,zipfian,pinned,5")
    assert s.n_users == 50
    assert s.register_common_body is True
    assert s.common_prefix_tokens > 0
    assert s.rounds == 5


def test_adhoc_parser_rejects_incomplete_spec():
    with pytest.raises(ValueError):
        _parse_adhoc_scenario("100,2000")
