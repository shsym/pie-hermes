"""Tests for the probe-expectation analyzer (bench.check_probes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.check_probes import (
    ProbeReport,
    _all_tool_calls,
    _args_match,
    _final_reply_text,
    check_probe,
    check_run,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _capture_row(message_content: str | None, tool_calls: list[dict] | None = None) -> dict:
    """Build a single capture.jsonl row with one assistant message."""
    return {
        "kwargs": {"messages": [], "tools": []},
        "response": {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
        },
    }


def _write_capture(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# -----------------------------------------------------------------------------
# Helper-function unit tests
# -----------------------------------------------------------------------------

def test_final_reply_text_returns_last_nonempty_assistant_content():
    msgs = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "final"},
    ]
    assert _final_reply_text(msgs) == "final"


def test_final_reply_text_skips_trailing_empty():
    msgs = [
        {"role": "assistant", "content": "real reply"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "  "},
    ]
    assert _final_reply_text(msgs) == "real reply"


def test_final_reply_text_returns_empty_when_all_blank():
    assert _final_reply_text([{"role": "assistant", "content": ""}]) == ""
    assert _final_reply_text([]) == ""


def test_all_tool_calls_flattens_across_messages_and_parses_args():
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "read_context",
                          "arguments": '{"section_id": "agents.md#x"}'}},
        ]},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "todo",
                          "arguments": '{"merge": false}'}},
        ]},
    ]
    calls = _all_tool_calls(msgs)
    assert calls == [
        ("read_context", {"section_id": "agents.md#x"}),
        ("todo", {"merge": False}),
    ]


def test_all_tool_calls_handles_invalid_json_args_gracefully():
    msgs = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "borked", "arguments": "not json"}},
    ]}]
    calls = _all_tool_calls(msgs)
    assert calls == [("borked", {"__raw__": "not json"})]


def test_args_match_partial_with_nested():
    assert _args_match({"a": 1}, {"a": 1, "b": 2}) is True
    assert _args_match({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}}) is True
    assert _args_match({"a": 1}, {"a": 2}) is False
    assert _args_match({"a": 1}, {}) is False


# -----------------------------------------------------------------------------
# check_probe behavior
# -----------------------------------------------------------------------------

def test_passes_when_must_contain_all_satisfied(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("user is @alice_demo and rollout is 1.42")])
    r = check_probe("p", cap, {"must_contain_all": ["@alice_demo", "1.42"]})
    assert r.passed and not r.failures


def test_fails_when_must_contain_all_violated(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("hello there")])
    r = check_probe("p", cap, {"must_contain_all": ["@alice_demo", "1.42"]})
    assert not r.passed
    assert len(r.failures) == 2
    assert all("missing" in f for f in r.failures)


def test_fails_when_must_not_contain_violated(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("the answer is 4 because forbidden_word")])
    r = check_probe("p", cap, {"must_not_contain": ["forbidden_word"]})
    assert not r.passed
    assert "forbidden" in r.failures[0]


def test_fails_when_must_call_tool_violated(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("answer", tool_calls=None)])
    r = check_probe("p", cap, {"must_call_tool": ["read_context"]})
    assert not r.passed


def test_passes_when_must_not_call_tool_satisfied(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("answer", tool_calls=None)])
    r = check_probe("p", cap, {"must_not_call_tool": ["read_context"]})
    assert r.passed


def test_fails_when_must_not_call_tool_violated(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [
        _capture_row(None, tool_calls=[
            {"function": {"name": "read_context", "arguments": '{"section_id": "x"}'}},
        ]),
        _capture_row("done"),
    ])
    r = check_probe("p", cap, {"must_not_call_tool": ["read_context"]})
    assert not r.passed
    assert "was called" in r.failures[0]


def test_must_call_tool_with_args_partial_match(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [
        _capture_row(None, tool_calls=[
            {"function": {"name": "read_context",
                          "arguments": '{"section_id": "agents.md#coding-style", "extra": 7}'}},
        ]),
        _capture_row("done"),
    ])
    r = check_probe("p", cap, {"must_call_tool_with_args": {
        "read_context": {"section_id": "agents.md#coding-style"},
    }})
    assert r.passed


def test_must_call_tool_with_args_fails_on_wrong_value(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [
        _capture_row(None, tool_calls=[
            {"function": {"name": "read_context",
                          "arguments": '{"section_id": "agents.md#wrong"}'}},
        ]),
    ])
    r = check_probe("p", cap, {"must_call_tool_with_args": {
        "read_context": {"section_id": "agents.md#right"},
    }})
    assert not r.passed


def test_unknown_expectation_key_fails_loudly(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("ok")])
    r = check_probe("p", cap, {"must_contains_all": ["typo"]})  # typo: "contains"
    assert not r.passed
    assert "unsupported expectation keys" in r.failures[0]


def test_missing_capture_file_fails(tmp_path):
    r = check_probe("p", tmp_path / "absent.jsonl", {"must_contain_all": ["x"]})
    assert not r.passed
    assert "missing" in r.failures[0]


# -----------------------------------------------------------------------------
# Phase-3.0 must_import_kv — detect registered-handle match on role:"tool"
# bodies via pie_cache.tool_result_tokens_imported. The probe must pass
# whether telemetry rides the non-streaming response top-level field or the
# final streaming chunk's field (Phase 2.1 / Phase 3.0 wire parity).
# -----------------------------------------------------------------------------

def _capture_row_with_pie_cache(pie_cache: dict, *, streaming: bool = False) -> dict:
    """Capture row that carries `pie_cache` on either the response body
    (non-streaming) or the final chunk (streaming). Mirrors the inferlet's
    emission pattern — see src/handler.rs:1084-1095."""
    if streaming:
        return {
            "kwargs": {"messages": [], "tools": []},
            "chunks": [
                {"choices": [{"index": 0, "delta": {"content": "hc_"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "pie_cache": pie_cache},
            ],
        }
    return {
        "kwargs": {"messages": [], "tools": []},
        "response": {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hc_"},
                "finish_reason": "stop",
            }],
            "pie_cache": pie_cache,
        },
    }


def test_must_import_kv_passes_on_nonstreaming_response(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row_with_pie_cache(
        {"mode": "direct", "tool_result_tokens_imported": 42}
    )])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert r.passed, r.failures


def test_must_import_kv_passes_on_streaming_final_chunk(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row_with_pie_cache(
        {"mode": "direct", "tool_result_tokens_imported": 42},
        streaming=True,
    )])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert r.passed, r.failures


def test_must_import_kv_fails_when_field_absent(tmp_path):
    """Pre-Phase-3.0 inferlet build: pie_cache present but without the new
    field. Analyzer must fail loudly so probe-red is visible."""
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row_with_pie_cache({"mode": "direct"})])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert not r.passed
    assert any("tool_result_tokens_imported" in f for f in r.failures)


def test_must_import_kv_fails_when_below_min_tokens(tmp_path):
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row_with_pie_cache(
        {"mode": "direct", "tool_result_tokens_imported": 0}
    )])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert not r.passed
    assert any("min_tokens" in f for f in r.failures)


def test_must_import_kv_fails_when_no_pie_cache_at_all(tmp_path):
    """Capture hook wrote rows but no telemetry anywhere — mis-configured
    build. Surface it so we don't silently pass."""
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [_capture_row("hc_")])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert not r.passed
    assert any("no pie_cache telemetry" in f for f in r.failures)


def test_must_import_kv_uses_max_across_turns(tmp_path):
    """Multi-turn probes emit one pie_cache per chat-completion; only one of
    the turns (the post-tool-call turn) carries a body that matches a
    registered handle. The probe passes when ANY turn sees a match."""
    cap = tmp_path / "capture.jsonl"
    _write_capture(cap, [
        _capture_row_with_pie_cache({"mode": "direct", "tool_result_tokens_imported": 0}),
        _capture_row_with_pie_cache({"mode": "session_kv", "tool_result_tokens_imported": 88}),
    ])
    r = check_probe("p", cap, {"must_import_kv": {"min_tokens": 1}})
    assert r.passed, r.failures


# -----------------------------------------------------------------------------
# check_run cross-run integration
# -----------------------------------------------------------------------------

def test_check_run_only_evaluates_probes_present_in_scenarios(tmp_path, monkeypatch):
    """If a scenario in PROBE_EXPECTATIONS isn't in the run, skip it.
    This makes the analyzer safe to invoke against any capture-run."""
    from bench import check_probes as cp

    monkeypatch.setattr(cp, "PROBE_EXPECTATIONS", {
        "scenario-A": {"must_contain_all": ["A"]},
        "scenario-B": {"must_contain_all": ["B"]},
    })

    run_dir = tmp_path / "run"
    _write_capture(run_dir / "scenarios" / "scenario-A" / "capture.jsonl",
                   [_capture_row("contains A token")])
    # scenario-B intentionally absent

    reports = check_run(run_dir)
    assert len(reports) == 1
    assert reports[0].slug == "scenario-A"
    assert reports[0].passed
