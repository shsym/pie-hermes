"""Tests for bench.replay_pressure — loader, schedule, scenario spec.

The HTTP firing path is exercised against a live pod (the sweep's
ground truth); these tests cover the pure-function surface that pins
the replay's determinism guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.replay_pressure import (
    ReplayScenario,
    _schedule,
    load_conversation,
)


def _write_capture(path: Path, turns: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"kwargs": t, "ok": True} for t in turns]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_conversation_preserves_turn_order(tmp_path):
    """Multi-turn session-KV reuse depends on turn N+1 being a prefix-
    extension of turn N. Reordering turns silently would destroy that
    invariant and produce garbage hit rates."""
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "user", "content": "q1"}]},
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"},
                      {"role": "user", "content": "q2"}]},
    ])
    turns = load_conversation(cap)
    assert len(turns) == 2
    assert len(turns[0]["messages"]) == 2
    assert len(turns[1]["messages"]) == 4
    # Turn 1 must be a prefix-extension of turn 0 (same system/user0).
    assert turns[1]["messages"][:2] == turns[0]["messages"]


def test_load_conversation_skips_malformed_rows(tmp_path):
    """A capture interrupted mid-write can leave a truncated final line.
    Loader should drop it rather than abort the replay."""
    cap = tmp_path / "capture.jsonl"
    cap.write_text(
        '{"kwargs":{"messages":[{"role":"user","content":"ok"}]}}\n'
        'not a json line\n'
        '{"kwargs":{"messages":[]}}\n'  # empty messages -> skipped
        '{"not_kwargs":true}\n'          # no kwargs -> skipped
    )
    turns = load_conversation(cap)
    assert len(turns) == 1


def test_load_conversation_errors_when_no_usable_turns(tmp_path):
    """If the source file has zero usable turns, fail fast — replaying
    nothing is never the caller's intent."""
    cap = tmp_path / "capture.jsonl"
    cap.write_text('{"not_kwargs":true}\n')
    with pytest.raises(ValueError, match="no usable turns"):
        load_conversation(cap)


def test_load_conversation_errors_when_source_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_conversation(tmp_path / "absent.jsonl")


# ---------------------------------------------------------------------------
# Schedule shape — sequential vs interleaved
# ---------------------------------------------------------------------------


def test_sequential_schedule_groups_by_user():
    """Sequential: user u finishes all rounds*turns before user u+1
    starts. This is the per-user hit-rate upper bound — session stays
    warm across the user's own turns."""
    s = _schedule(n_users=3, n_turns=2, rounds=2, pattern="sequential")
    # 3 users × 2 rounds × 2 turns = 12 items.
    assert len(s) == 12
    # User 0 occupies positions 0..3, user 1 occupies 4..7, etc.
    assert [t[0] for t in s[:4]] == [0, 0, 0, 0]
    assert [t[0] for t in s[4:8]] == [1, 1, 1, 1]
    # Within a user, (round, turn) order is (0,0), (0,1), (1,0), (1,1).
    assert [(t[1], t[2]) for t in s[:4]] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_interleaved_schedule_groups_by_round_turn():
    """Interleaved: all users fire (round=r, turn=t) before advancing.
    Each switch from user i to user i+1 evicts i's session (single-slot).
    By the time user 0 reaches turn=1, its session has been overwritten
    N-1 times — expected per-user hit rate ≈ 0."""
    s = _schedule(n_users=3, n_turns=2, rounds=1, pattern="interleaved")
    # 3 × 2 = 6 items.
    assert len(s) == 6
    # Positions 0..2 are (r=0, t=0) for users 0..2.
    assert [(t[0], t[1], t[2]) for t in s[:3]] == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    # Positions 3..5 are (r=0, t=1) for users 0..2.
    assert [(t[0], t[1], t[2]) for t in s[3:]] == [(0, 0, 1), (1, 0, 1), (2, 0, 1)]


def test_both_patterns_fire_same_total_requests():
    """Cost-preflight sanity: switching pattern must not change the
    total request budget — otherwise an A/B between patterns at the
    same (N, turns, rounds) would muddle the comparison by changing
    both the access shape and the aggregate load."""
    seq = _schedule(10, 3, 2, "sequential")
    ilv = _schedule(10, 3, 2, "interleaved")
    assert len(seq) == len(ilv) == 10 * 3 * 2


def test_unknown_pattern_raises():
    with pytest.raises(ValueError, match="unknown pattern"):
        _schedule(1, 1, 1, "random")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReplayScenario dataclass
# ---------------------------------------------------------------------------


def test_scenario_is_frozen():
    """Accidental mutation during a sweep would desync meta.json from
    actual traffic."""
    s = ReplayScenario(
        slug="x", source_capture_path=Path("/dev/null"),
        n_users=1, pattern="sequential",
    )
    with pytest.raises((AttributeError, Exception)):
        s.n_users = 10  # type: ignore[misc]


def test_scenario_bearer_prefix_default_namespaces_replay_traffic():
    """Bearer must be recognizably ours in pod logs — a collision with
    real traffic's bearer would poison session attribution."""
    s = ReplayScenario(
        slug="x", source_capture_path=Path("/dev/null"),
        n_users=1, pattern="sequential",
    )
    assert s.bearer_prefix == "replay"
