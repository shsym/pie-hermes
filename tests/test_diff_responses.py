"""Tests for the behavioral diff analyzer (bench.diff_responses)."""

from __future__ import annotations

import json
from pathlib import Path

from bench.diff_responses import ScenarioDiff, diff_runs, diff_scenario


# -----------------------------------------------------------------------------
# Fixtures — build a pair of capture-run directories on disk
# -----------------------------------------------------------------------------

def _write_scenario(root: Path, slug: str, *,
                    exit_code: int = 0,
                    n_calls: int = 1,
                    assistant_content: str | None = "ok",
                    tool_calls: list[dict] | None = None) -> None:
    scen = root / "scenarios" / slug
    scen.mkdir(parents=True, exist_ok=True)
    (scen / "meta.json").write_text(json.dumps({
        "slug": slug, "exit_code": exit_code, "n_completions_calls": n_calls,
    }))
    row = {
        "response": {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                },
            }],
        },
    }
    (scen / "capture.jsonl").write_text(json.dumps(row) + "\n")


# -----------------------------------------------------------------------------
# diff_scenario — per-scenario comparisons
# -----------------------------------------------------------------------------

def test_exact_match_across_text_tools_and_meta(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", assistant_content="hello world")
    _write_scenario(cand, "s", assistant_content="hello world")
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert d.text_exact_match
    assert d.text_similarity == 1.0
    assert not d.has_structural_diff


def test_text_only_diff_does_not_register_as_structural(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", assistant_content="hello there friend")
    _write_scenario(cand, "s", assistant_content="hello there, pal")
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert not d.text_exact_match
    assert 0.5 < d.text_similarity < 1.0
    assert not d.has_structural_diff


def test_exit_code_diff_is_structural(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", exit_code=0)
    _write_scenario(cand, "s", exit_code=1)
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert d.exit_code_diff == (0, 1)
    assert d.has_structural_diff


def test_n_calls_diff_is_structural(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", n_calls=1)
    _write_scenario(cand, "s", n_calls=2)
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert d.n_calls_diff == (1, 2)
    assert d.has_structural_diff


def test_tool_name_sequence_diff_is_structural(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", tool_calls=[
        {"function": {"name": "read_file", "arguments": '{"path":"a"}'}},
    ])
    _write_scenario(cand, "s", tool_calls=[
        {"function": {"name": "list_dir",  "arguments": '{"path":"a"}'}},
    ])
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert d.tool_name_seq_diff == (["read_file"], ["list_dir"])
    assert d.has_structural_diff


def test_tool_arg_keys_diff_is_structural_when_name_sequence_matches(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", tool_calls=[
        {"function": {"name": "read_file", "arguments": '{"path":"a","offset":10}'}},
    ])
    _write_scenario(cand, "s", tool_calls=[
        {"function": {"name": "read_file", "arguments": '{"path":"a"}'}},
    ])
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    # name sequence agrees → arg-key delta is reported
    assert d.tool_name_seq_diff is None
    assert len(d.tool_arg_keys_diff) == 1
    assert d.tool_arg_keys_diff[0]["base_only"] == ["offset"]
    assert d.has_structural_diff


def test_tool_arg_keys_diff_suppressed_when_name_sequence_diverges(tmp_path):
    """If tool names disagree, arg-key noise is suppressed — the name diff
    is the headline and would make the index-wise key comparison misleading."""
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", tool_calls=[
        {"function": {"name": "read_file", "arguments": '{"path":"a","offset":10}'}},
    ])
    _write_scenario(cand, "s", tool_calls=[
        {"function": {"name": "list_dir",  "arguments": '{"path":"a"}'}},
    ])
    d = diff_scenario("s", base / "scenarios" / "s", cand / "scenarios" / "s")
    assert d.tool_name_seq_diff is not None
    assert d.tool_arg_keys_diff == []


# -----------------------------------------------------------------------------
# diff_runs — cross-run driver
# -----------------------------------------------------------------------------

def test_diff_runs_reports_coverage_asymmetries(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "shared")
    _write_scenario(base, "base-only")
    _write_scenario(cand, "shared")
    _write_scenario(cand, "cand-only")
    result = diff_runs(base, cand)
    assert result["base_only_scenarios"] == ["base-only"]
    assert result["cand_only_scenarios"] == ["cand-only"]
    assert [d.slug for d in result["diffs"]] == ["shared"]


def test_diff_runs_only_filter(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "a")
    _write_scenario(base, "b")
    _write_scenario(cand, "a")
    _write_scenario(cand, "b")
    result = diff_runs(base, cand, only=["a"])
    assert [d.slug for d in result["diffs"]] == ["a"]


def test_diff_runs_handles_streaming_capture(tmp_path):
    """The analyzer must reassemble streaming chunks (not just non-streaming
    `choices[0].message`). Writes a base as non-streaming and cand as
    streaming — same content — and expects an exact match."""
    base, cand = tmp_path / "base", tmp_path / "cand"
    _write_scenario(base, "s", assistant_content="hi")
    scen = cand / "scenarios" / "s"
    scen.mkdir(parents=True)
    (scen / "meta.json").write_text(json.dumps({
        "slug": "s", "exit_code": 0, "n_completions_calls": 1,
    }))
    row = {
        "chunks": [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ],
    }
    (scen / "capture.jsonl").write_text(json.dumps(row) + "\n")
    result = diff_runs(base, cand)
    d = result["diffs"][0]
    assert d.text_exact_match
    assert not d.has_structural_diff
