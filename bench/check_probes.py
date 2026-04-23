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

def _iter_assistant_messages(rows: list[dict]) -> list[dict]:
    """Return assistant messages across every chat-completions response in
    chronological order (one per row)."""
    msgs: list[dict] = []
    for r in rows:
        resp = r.get("response") or {}
        for ch in resp.get("choices", []) or []:
            m = ch.get("message")
            if m and m.get("role") == "assistant":
                msgs.append(m)
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_SUPPORTED_KEYS = {
    "must_contain_all",
    "must_not_contain",
    "must_call_tool",
    "must_not_call_tool",
    "must_call_tool_with_args",
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
