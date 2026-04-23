"""Tests for bench.analyze_pressure — capture → stats reducer.

We only cover the pure aggregation path; rendering is a string operation
best verified by reading the output in a real capture. The main invariants
to lock down are (a) the reducer counts hits by round correctly, (b) it
tolerates both streaming and non-streaming pie_cache placement, and (c) a
row without pressure_* metadata doesn't crash the pipeline — that happens
when a non-pressure scenario's capture is accidentally pointed at this
analyzer.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.analyze_pressure import aggregate, _iter_rows, format_report


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(*, user, round_idx, skipped, tool_imported=0, ttft=None, streaming=False,
         ok=True, mode="direct"):
    pc = {
        "mode": mode,
        "prefill_tokens_skipped": skipped,
        "prompt_tokens_total": 1000,
        "tool_result_tokens_imported": tool_imported,
    }
    row: dict = {
        "kwargs": {},
        "ok": ok,
        "pressure_user_idx": user,
        "pressure_round": round_idx,
    }
    if ttft is not None:
        row["ttft_s"] = ttft
    if streaming:
        row["chunks"] = [
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "pie_cache": pc},
        ]
    else:
        row["response"] = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "pie_cache": pc,
        }
    return row


def test_aggregate_counts_hits_per_round(tmp_path):
    """Round 0: cold cache, no hits. Round 1: first 2 users hit, last 2 miss.
    Round 2: all hit. This is the shape we expect from a well-behaved APC
    under moderate pressure."""
    cap = tmp_path / "capture.jsonl"
    _write_rows(cap, [
        _row(user=0, round_idx=0, skipped=0),
        _row(user=1, round_idx=0, skipped=0),
        _row(user=2, round_idx=0, skipped=0),
        _row(user=3, round_idx=0, skipped=0),
        _row(user=0, round_idx=1, skipped=500),
        _row(user=1, round_idx=1, skipped=500),
        _row(user=2, round_idx=1, skipped=0),
        _row(user=3, round_idx=1, skipped=0),
        _row(user=0, round_idx=2, skipped=500, streaming=True),
        _row(user=1, round_idx=2, skipped=500, streaming=True),
        _row(user=2, round_idx=2, skipped=500, streaming=True),
        _row(user=3, round_idx=2, skipped=500, streaming=True),
    ])
    st = aggregate(_iter_rows(cap))
    assert st.n_rows == 12
    assert st.n_errors == 0
    assert st.rounds[0] == {"n": 4, "hits": 0, "skipped_tokens": 0}
    assert st.rounds[1] == {"n": 4, "hits": 2, "skipped_tokens": 1000}
    assert st.rounds[2] == {"n": 4, "hits": 4, "skipped_tokens": 2000}


def test_aggregate_reads_pie_cache_from_streaming_and_nonstreaming(tmp_path):
    """The inferlet places pie_cache on response.pie_cache for non-
    streaming and on the final SSE chunk for streaming. Both must
    surface through the same aggregation code."""
    cap = tmp_path / "capture.jsonl"
    _write_rows(cap, [
        _row(user=0, round_idx=0, skipped=100, streaming=False),
        _row(user=0, round_idx=0, skipped=100, streaming=True),
    ])
    st = aggregate(_iter_rows(cap))
    assert st.rounds[0]["hits"] == 2
    assert st.total_prefill_tokens_skipped == 200


def test_aggregate_tracks_per_user_distribution(tmp_path):
    """Zipfian arms produce skewed per-user hit rates — the analyzer
    must surface that asymmetry so hot-vs-cold survivability is
    visible in the report."""
    cap = tmp_path / "capture.jsonl"
    rows = []
    # user 0 = hot, 5 hits.
    for _ in range(5):
        rows.append(_row(user=0, round_idx=0, skipped=500))
    # user 1 = cold, 1 miss.
    rows.append(_row(user=1, round_idx=0, skipped=0))
    _write_rows(cap, rows)
    st = aggregate(_iter_rows(cap))
    assert st.per_user_hits.get(0) == 5
    assert st.per_user_total.get(0) == 5
    assert st.per_user_hits.get(1, 0) == 0
    assert st.per_user_total.get(1) == 1


def test_aggregate_handles_errors_without_crashing(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_rows(cap, [
        _row(user=0, round_idx=0, skipped=100),
        {"ok": False, "error": "timeout", "pressure_user_idx": 1,
         "pressure_round": 0, "kwargs": {}},
    ])
    st = aggregate(_iter_rows(cap))
    assert st.n_rows == 2
    assert st.n_errors == 1
    # The error row must NOT count as a hit or participate in totals.
    assert st.rounds[0] == {"n": 1, "hits": 1, "skipped_tokens": 100}


def test_aggregate_tolerates_rows_without_pressure_metadata(tmp_path):
    """If someone accidentally points this at a non-pressure capture,
    the reducer should produce a sparse stats object rather than
    crashing with a KeyError."""
    cap = tmp_path / "capture.jsonl"
    _write_rows(cap, [
        {"kwargs": {}, "ok": True,
         "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                      "pie_cache": {"mode": "direct",
                                    "prefill_tokens_skipped": 42,
                                    "prompt_tokens_total": 1000}}},
    ])
    st = aggregate(_iter_rows(cap))
    assert st.n_rows == 1
    assert st.rounds == {}          # no round idx → no round bucket
    assert st.per_user_total == {}  # no user idx → no per-user bucket
    # Totals still accumulate.
    assert st.total_prefill_tokens_skipped == 42


def test_format_report_does_not_crash_on_empty_stats(tmp_path):
    """Rendering an empty / error-only capture is part of the failure
    path — the analyzer should still produce a readable report."""
    cap = tmp_path / "capture.jsonl"
    _write_rows(cap, [])
    st = aggregate(_iter_rows(cap))
    meta = {"scenario_slug": "empty", "n_users": 10, "l_tokens": 4000,
            "pattern": "round-robin", "rounds": 3}
    out = format_report(meta, st)
    assert "empty" in out
    # No ttft data -> fallback line present.
    assert "no TTFT data" in out


def test_format_report_computes_pressure_ratio_when_capacity_given():
    st = aggregate([])  # empty but meta carries enough
    meta = {"scenario_slug": "x", "n_users": 500, "l_tokens": 4000,
            "pattern": "round-robin", "rounds": 3}
    out = format_report(meta, st, kv_capacity_tokens=180_000)
    # 500 * 4000 / 180000 ≈ 11.11× — must appear verbatim.
    assert "11.11×" in out
