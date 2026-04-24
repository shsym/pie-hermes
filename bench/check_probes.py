#!/usr/bin/env python3
"""Verify probe expectations against captured traffic.

Loads a captures/runs/<RUN_ID>/ directory and, for every scenario whose
slug appears in `bench.driver.PROBE_EXPECTATIONS`, asserts the declared
expectations against the captured assistant messages.

Schema of an expectation entry (see bench/driver.py:PROBE_EXPECTATIONS):

    must_contain_all:        list[str]   final assistant reply must contain ALL
    must_not_contain:        list[str]   final assistant reply must contain NONE
    must_call_tool:          list[str]   each name appeared in tool_calls of
                                         at least one assistant message
    must_not_call_tool:      list[str]   none of these names ever appeared
    must_call_tool_with_args dict[name,
                                  partial-args dict] each named tool must have
                                         been called with at least the named
                                         keys/values in args (partial match)
    must_import_kv           dict      at least one captured pie_cache.
                                         tool_result_tokens_imported value
                                         must be >= min_tokens. Phase-3.0
                                         detection gate (VENDOR_SOURCE #8).
                                         Shape: {"min_tokens": int}.

Usage:
    python -m bench.check_probes captures/runs/<RUN_ID>           # exit 0 = all pass
    python -m bench.check_probes captures/runs/<RUN_ID> --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.driver import PROBE_EXPECTATIONS


@dataclass
class ProbeReport:
    slug: str
    passed: bool
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Capture extraction — pull the assistant messages out of capture.jsonl
# ---------------------------------------------------------------------------

def _reassemble_streaming_message(chunks: list[dict]) -> dict:
    """Reassemble a single assistant message from streaming chunks.

    OpenAI streaming sends incremental deltas: `choices[0].delta` carries
    partial `content` strings and partial `tool_calls` entries indexed by
    `index`. The final chunk sets `finish_reason`. We rebuild a message
    that looks like the non-streaming `choices[0].message` shape.
    """
    content_parts: list[str] = []
    # Tool calls accumulate by their `index` field; arguments arrive as
    # multi-chunk strings that must be concatenated in order.
    tc_by_idx: dict[int, dict] = {}
    for c in chunks:
        choices = c.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            # `index` is required by the OpenAI streaming spec — it's how
            # distinct tool calls within the same assistant turn are
            # disambiguated, and how argument-string chunks are joined to
            # the right call. Silently defaulting to 0 would coalesce every
            # index-less chunk into slot 0 and concatenate unrelated argument
            # strings into invalid JSON. Instead, allocate a fresh slot per
            # index-less chunk so the anomaly surfaces as an obviously-extra
            # call (and the analyzer's wire-shape check catches it) rather
            # than as silently merged args.
            raw_idx = tc.get("index")
            if raw_idx is None:
                raw_idx = max(tc_by_idx.keys(), default=-1) + 1
            slot = tc_by_idx.setdefault(raw_idx, {"function": {"name": "", "arguments": ""}})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
    return {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
        "tool_calls": [tc_by_idx[i] for i in sorted(tc_by_idx.keys())] or None,
    }


def _iter_assistant_messages(rows: list[dict]) -> list[dict]:
    """Return assistant messages across every chat-completions response in
    chronological order (one per row). Handles both non-streaming
    (`response.choices[0].message`) and streaming (`chunks` array) captures."""
    msgs: list[dict] = []
    for r in rows:
        # Non-streaming path
        resp = r.get("response") or {}
        for ch in resp.get("choices", []) or []:
            m = ch.get("message")
            if m and m.get("role") == "assistant":
                msgs.append(m)
        # Streaming path: capture stores chunks under `chunks` (or
        # `stream_chunks` in older builds).
        chunks = r.get("chunks") or r.get("stream_chunks") or []
        if chunks:
            msgs.append(_reassemble_streaming_message(chunks))
    return msgs


def _final_reply_text(asst_msgs: list[dict]) -> str:
    """The 'final reply' = last assistant message whose content is non-empty
    (skips intermediate tool-call-only messages with empty content)."""
    for m in reversed(asst_msgs):
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, list):
            text = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if text.strip():
                return text
    return ""


def _all_tool_calls(asst_msgs: list[dict]) -> list[tuple[str, dict]]:
    """Flatten every tool call across every assistant message. Args are
    deserialized from JSON strings when possible; left as the raw value
    otherwise so the analyzer surfaces wire-shape bugs."""
    out: list[tuple[str, dict]] = []
    for m in asst_msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, ValueError):
                    args = {"__raw__": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            if name:
                out.append((name, args))
    return out


# ---------------------------------------------------------------------------
# Per-expectation checks
# ---------------------------------------------------------------------------

def _check_must_contain_all(reply: str, needles: list[str]) -> list[str]:
    return [f"final reply missing {n!r}" for n in needles if n not in reply]


def _check_must_not_contain(reply: str, forbidden: list[str]) -> list[str]:
    return [f"final reply contains forbidden {n!r}" for n in forbidden if n in reply]


def _check_must_call_tool(calls: list[tuple[str, dict]], required: list[str]) -> list[str]:
    seen = {name for name, _ in calls}
    return [f"required tool not called: {name!r}" for name in required if name not in seen]


def _check_must_not_call_tool(calls: list[tuple[str, dict]], forbidden: list[str]) -> list[str]:
    seen = {name for name, _ in calls}
    return [f"forbidden tool was called: {name!r}" for name in forbidden if name in seen]


def _args_match(expected: dict, actual: dict) -> bool:
    """Partial match: every key in `expected` is present in `actual` with
    the same value (recursive for nested dicts)."""
    for k, v in expected.items():
        if k not in actual:
            return False
        if isinstance(v, dict) and isinstance(actual[k], dict):
            if not _args_match(v, actual[k]):
                return False
        elif actual[k] != v:
            return False
    return True


def _check_must_call_tool_with_args(
    calls: list[tuple[str, dict]],
    spec: dict[str, dict],
) -> list[str]:
    failures: list[str] = []
    for name, expected_args in spec.items():
        matches = [args for n, args in calls if n == name and _args_match(expected_args, args)]
        if not matches:
            actual_for_name = [args for n, args in calls if n == name]
            failures.append(
                f"tool {name!r} not called with expected args {expected_args!r}; "
                f"actual calls: {actual_for_name!r}"
            )
    return failures


def _iter_pie_cache_entries(rows: list[dict]) -> list[dict]:
    """Return every `pie_cache` telemetry object observed in the capture.

    Two wire shapes emit telemetry:
      - Non-streaming: `response.pie_cache` (top-level field on the chat
        completion response).
      - Streaming: the final SSE chunk carries `pie_cache` alongside
        `finish_reason`. The capture hook stores all chunks under `chunks`
        (or `stream_chunks` on older builds); we scan each chunk.

    We collect ALL occurrences so a multi-turn probe (one chat-completion
    per turn) can be inspected across turns.
    """
    out: list[dict] = []
    for r in rows:
        resp = r.get("response") or {}
        pc = resp.get("pie_cache")
        if isinstance(pc, dict):
            out.append(pc)
        for ch in r.get("chunks") or r.get("stream_chunks") or []:
            pc = ch.get("pie_cache")
            if isinstance(pc, dict):
                out.append(pc)
    return out


def _check_must_import_kv(rows: list[dict], spec: dict[str, Any]) -> list[str]:
    min_tokens = spec.get("min_tokens", 1)
    entries = _iter_pie_cache_entries(rows)
    if not entries:
        return [
            "must_import_kv: no pie_cache telemetry observed in capture "
            "(inferlet build predates Phase 3.0 telemetry wiring, or the "
            "capture hook did not record chunks/response fields)"
        ]
    observed = [
        e.get("tool_result_tokens_imported")
        for e in entries
        if e.get("tool_result_tokens_imported") is not None
    ]
    if not observed:
        return [
            "must_import_kv: pie_cache present but tool_result_tokens_imported "
            "field is absent on every entry (Phase-3.0 telemetry not populated; "
            "see VENDOR_SOURCE.md #8)"
        ]
    best = max(int(v) for v in observed)
    if best < min_tokens:
        return [
            f"must_import_kv: max tool_result_tokens_imported={best} "
            f"< min_tokens={min_tokens}; inferlet did not detect a matching "
            "context_section handle for any inbound role:\"tool\" body"
        ]
    return []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_SUPPORTED_KEYS = {
    "must_contain_all",
    "must_not_contain",
    "must_call_tool",
    "must_not_call_tool",
    "must_call_tool_with_args",
    "must_import_kv",
}


def check_probe(slug: str, capture_path: Path, expectation: dict[str, Any]) -> ProbeReport:
    unknown = set(expectation.keys()) - _SUPPORTED_KEYS
    if unknown:
        return ProbeReport(slug=slug, passed=False,
                           failures=[f"unsupported expectation keys: {sorted(unknown)}"])

    if not capture_path.exists():
        return ProbeReport(slug=slug, passed=False,
                           failures=[f"capture file missing: {capture_path}"])

    rows = [json.loads(l) for l in capture_path.read_text().splitlines() if l.strip()]
    asst = _iter_assistant_messages(rows)
    reply = _final_reply_text(asst)
    calls = _all_tool_calls(asst)

    failures: list[str] = []
    failures += _check_must_contain_all(reply, expectation.get("must_contain_all") or [])
    failures += _check_must_not_contain(reply, expectation.get("must_not_contain") or [])
    failures += _check_must_call_tool(calls, expectation.get("must_call_tool") or [])
    failures += _check_must_not_call_tool(calls, expectation.get("must_not_call_tool") or [])
    failures += _check_must_call_tool_with_args(
        calls, expectation.get("must_call_tool_with_args") or {}
    )
    if "must_import_kv" in expectation:
        failures += _check_must_import_kv(rows, expectation["must_import_kv"])

    return ProbeReport(slug=slug, passed=not failures, failures=failures)


def check_run(run_dir: Path) -> list[ProbeReport]:
    scen_root = run_dir / "scenarios"
    if not scen_root.exists():
        raise SystemExit(f"no scenarios/ dir in {run_dir}")
    reports: list[ProbeReport] = []
    for slug, expectation in PROBE_EXPECTATIONS.items():
        capture_path = scen_root / slug / "capture.jsonl"
        if not capture_path.exists():
            # Probe wasn't part of this run — silently skip. Analyzer is
            # cross-run-safe; only assert against probes that actually ran.
            continue
        reports.append(check_probe(slug, capture_path, expectation))
    return reports


def _format_text(reports: list[ProbeReport]) -> str:
    if not reports:
        return "no probes ran in this capture (PROBE_EXPECTATIONS slugs not in scenarios/)"
    lines = []
    width = max(len(r.slug) for r in reports)
    for r in reports:
        marker = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{marker}] {r.slug:<{width}}")
        for f in r.failures:
            lines.append(f"         - {f}")
    n_fail = sum(1 for r in reports if not r.passed)
    summary = f"{len(reports) - n_fail}/{len(reports)} probes passed"
    return "\n".join(lines) + "\n" + summary


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bench.check_probes")
    p.add_argument("run_dir", type=Path, help="captures/runs/<RUN_ID> directory")
    p.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = p.parse_args(argv)

    reports = check_run(args.run_dir)
    if args.json:
        print(json.dumps([
            {"slug": r.slug, "passed": r.passed, "failures": r.failures}
            for r in reports
        ], indent=2))
    else:
        print(_format_text(reports))

    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
