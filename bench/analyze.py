#!/usr/bin/env python3
"""Analyze a captures/runs/<RUN_ID>/ directory produced by driver.py.

For each scenario within a run, prints a per-turn composition breakdown
(system / user / assistant / tool_result / tool_schemas in char counts
and estimated tokens at 4 chars/token — accurate to within ~10% for
English+JSON payloads). Aggregates at the run level and, when --compare
is given, across multiple runs.

Usage::

    python scenarios/analyze.py captures/runs/<RUN_ID>
    python scenarios/analyze.py --compare captures/runs/<RUN_A> captures/runs/<RUN_B>

Writes an ``analysis.txt`` inside the target run directory (unless
--no-write is given). Same text is printed to stdout."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4.0


# -----------------------------------------------------------------------------
# Per-call composition
# -----------------------------------------------------------------------------

@dataclass
class CallBreakdown:
    # Raw char counts
    sys_chars: int = 0
    user_chars: int = 0
    asst_chars: int = 0
    tool_result_chars: int = 0
    tool_schema_chars: int = 0
    n_tools: int = 0
    n_msgs: int = 0
    stream: bool = False
    finish_reason: str | None = None
    error: str | None = None

    @property
    def total_chars(self) -> int:
        return (self.sys_chars + self.user_chars + self.asst_chars
                + self.tool_result_chars + self.tool_schema_chars)

    @property
    def est_tokens(self) -> int:
        return int(self.total_chars / CHARS_PER_TOKEN)


def _msg_chars(msg: dict) -> int:
    c = msg.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return 0


def breakdown_row(row: dict) -> CallBreakdown:
    kw = row.get("kwargs") or {}
    msgs = kw.get("messages") or []
    tools = kw.get("tools") or []

    b = CallBreakdown(
        n_tools=len(tools),
        n_msgs=len(msgs),
        stream=bool(kw.get("stream")),
        error=row.get("error"),
    )
    for m in msgs:
        role = m.get("role")
        if role == "system":
            b.sys_chars += _msg_chars(m)
        elif role == "user":
            b.user_chars += _msg_chars(m)
        elif role == "assistant":
            b.asst_chars += _msg_chars(m)
            for tc in (m.get("tool_calls") or []):
                b.asst_chars += len(json.dumps(tc, ensure_ascii=False))
        elif role == "tool":
            b.tool_result_chars += _msg_chars(m)
    b.tool_schema_chars = sum(len(json.dumps(t, ensure_ascii=False)) for t in tools)

    # Resolve finish_reason: non-stream .response.choices[0].finish_reason or
    # the last streamed chunk with one set.
    resp = row.get("response")
    if isinstance(resp, dict) and resp.get("choices"):
        b.finish_reason = resp["choices"][0].get("finish_reason")
    elif row.get("chunks"):
        for c in reversed(row["chunks"]):
            choices = (c.get("choices") or [])
            if choices and choices[0].get("finish_reason"):
                b.finish_reason = choices[0]["finish_reason"]
                break
    return b


# -----------------------------------------------------------------------------
# Scenario + run aggregation
# -----------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    slug: str
    description: str
    tags: list[str]
    meta: dict
    calls: list[CallBreakdown] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def max_prompt_tokens(self) -> int:
        return max((c.est_tokens for c in self.calls), default=0)

    @property
    def turn1_tokens(self) -> int:
        return self.calls[0].est_tokens if self.calls else 0

    @property
    def prompt_growth(self) -> int:
        """Tokens added from turn 1 to final turn."""
        if len(self.calls) < 2:
            return 0
        return self.calls[-1].est_tokens - self.calls[0].est_tokens


def load_run(run_dir: Path) -> tuple[dict, list[ScenarioResult]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text())

    results = []
    scen_root = run_dir / "scenarios"
    for scen_dir in sorted(scen_root.iterdir()) if scen_root.exists() else []:
        if not scen_dir.is_dir():
            continue
        meta_path = scen_dir / "meta.json"
        capture_path = scen_dir / "capture.jsonl"
        if not meta_path.exists() or not capture_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        rows = [json.loads(l) for l in capture_path.read_text().splitlines() if l.strip()]
        breakdowns = [breakdown_row(r) for r in rows]
        results.append(ScenarioResult(
            slug=meta["slug"],
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            meta=meta,
            calls=breakdowns,
        ))
    return manifest, results


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def _tok(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


def format_run(manifest: dict, scenarios: list[ScenarioResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"RUN: {manifest['run_id']}")
    lines.append(f"backend: {manifest['backend'].get('provider','?')} / "
                 f"{manifest['backend'].get('model','?')} "
                 f"[pod={manifest['backend'].get('pod_id','-')} "
                 f"inferlet={manifest['backend'].get('inferlet','-')}]")
    lines.append(f"hermes: {manifest.get('hermes_version','?')}  "
                 f"git: {manifest['git'].get('branch','')}@{manifest['git'].get('sha','')[:8]}"
                 f"{' (dirty)' if manifest['git'].get('dirty') else ''}")
    lines.append("=" * 90)

    lines.append(f"\n{'scenario':<24} {'tag':<8} {'turns':>5} {'t1_tok':>7} {'max_tok':>7} "
                 f"{'growth':>7} {'finish (last)':<15} {'exit':>4}")
    lines.append("-" * 90)
    for s in scenarios:
        tag = s.tags[0] if s.tags else ""
        finish = s.calls[-1].finish_reason if s.calls else "-"
        lines.append(
            f"{s.slug:<24} {tag:<8} {s.n_calls:>5} {s.turn1_tokens:>7,} "
            f"{s.max_prompt_tokens:>7,} {s.prompt_growth:>+7,} {str(finish):<15} "
            f"{s.meta.get('exit_code','?'):>4}"
        )

    # Per-turn detail for every scenario
    lines.append("\n" + "=" * 90)
    lines.append("PER-TURN BREAKDOWN (tokens ≈ chars/4)")
    lines.append("=" * 90)
    for s in scenarios:
        lines.append(f"\n## {s.slug}  [{','.join(s.tags)}]  — {s.description}")
        lines.append(f"   query: {s.meta.get('query','')[:120]}{'…' if len(s.meta.get('query',''))>120 else ''}")
        lines.append(f"   {'turn':>4} {'total':>8} {'Δ':>7} {'sys':>6} {'user':>5} "
                     f"{'asst':>6} {'toolres':>8} {'schema':>7} {'n_msg':>5} {'finish':<12}")
        prev = 0
        for i, c in enumerate(s.calls):
            delta = c.est_tokens - prev if i > 0 else 0
            prev = c.est_tokens
            lines.append(
                f"   {i:>4} {c.est_tokens:>8,} {('+' + format(delta, ',')) if i > 0 else '-':>7} "
                f"{_tok(c.sys_chars):>6,} {_tok(c.user_chars):>5,} "
                f"{_tok(c.asst_chars):>6,} {_tok(c.tool_result_chars):>8,} "
                f"{_tok(c.tool_schema_chars):>7,} {c.n_msgs:>5} {str(c.finish_reason or '-'):<12}"
            )

    # Aggregate
    lines.append("\n" + "=" * 90)
    lines.append("AGGREGATES")
    lines.append("=" * 90)
    all_calls = [c for s in scenarios for c in s.calls]
    t1_calls = [s.calls[0] for s in scenarios if s.calls]
    turn_deltas: list[int] = []
    for s in scenarios:
        for i in range(1, len(s.calls)):
            turn_deltas.append(s.calls[i].est_tokens - s.calls[i - 1].est_tokens)

    def _stat(name: str, vals: list[int]) -> None:
        if not vals:
            return
        lines.append(
            f"  {name:<28} n={len(vals):<4} min={min(vals):>7,}  "
            f"median={int(statistics.median(vals)):>7,}  "
            f"mean={int(statistics.mean(vals)):>7,}  max={max(vals):>7,}"
        )

    _stat("prompt tokens (all calls)", [c.est_tokens for c in all_calls])
    _stat("prompt tokens (turn 1)",    [c.est_tokens for c in t1_calls])
    _stat("per-turn Δ (turn-to-turn)", turn_deltas)
    _stat("tool_result tokens/turn",   [_tok(c.tool_result_chars) for c in all_calls])

    # Shallow vs deep comparison if tags present
    by_tag: dict[str, list[ScenarioResult]] = {}
    for s in scenarios:
        for t in s.tags or ["untagged"]:
            by_tag.setdefault(t, []).append(s)
    if "shallow" in by_tag and "deep" in by_tag:
        def _max_tok(rs: list[ScenarioResult]) -> int:
            return max((s.max_prompt_tokens for s in rs), default=0)
        def _median_tok(rs: list[ScenarioResult]) -> int:
            vals = [s.max_prompt_tokens for s in rs]
            return int(statistics.median(vals)) if vals else 0
        def _avg_calls(rs: list[ScenarioResult]) -> float:
            return statistics.mean(s.n_calls for s in rs) if rs else 0.0
        lines.append("\n  shallow vs deep:")
        lines.append(f"    shallow: {len(by_tag['shallow'])} scen, "
                     f"median peak={_median_tok(by_tag['shallow']):,} tok, "
                     f"avg calls/scen={_avg_calls(by_tag['shallow']):.1f}")
        lines.append(f"    deep:    {len(by_tag['deep'])} scen, "
                     f"median peak={_median_tok(by_tag['deep']):,} tok, "
                     f"avg calls/scen={_avg_calls(by_tag['deep']):.1f}")
        lines.append(f"    peak-prompt ratio (deep / shallow): "
                     f"{_max_tok(by_tag['deep']) / max(1, _max_tok(by_tag['shallow'])):.2f}×")

    return "\n".join(lines)


def format_compare(runs: list[tuple[Path, dict, list[ScenarioResult]]]) -> str:
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("CROSS-RUN COMPARISON — peak prompt tokens per scenario")
    lines.append("=" * 90)
    all_slugs = sorted({s.slug for _, _, scens in runs for s in scens})
    header = f"{'scenario':<24}" + "".join(f" {r[0].name[:28]:>30}" for r in runs)
    lines.append(header)
    lines.append("-" * len(header))
    for slug in all_slugs:
        row = f"{slug:<24}"
        for _, _, scens in runs:
            match = next((s for s in scens if s.slug == slug), None)
            row += f" {(str(match.max_prompt_tokens) + ' tok') if match else '-':>30}"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run dir(s) under captures/runs/")
    p.add_argument("--compare", action="store_true",
                   help="also emit a cross-run peak-tokens table")
    p.add_argument("--no-write", action="store_true",
                   help="don't write analysis.txt into the run dir(s)")
    args = p.parse_args(argv)

    loaded: list[tuple[Path, dict, list[ScenarioResult]]] = []
    for raw in args.paths:
        path = Path(raw).resolve()
        manifest, scens = load_run(path)
        loaded.append((path, manifest, scens))

    for path, manifest, scens in loaded:
        text = format_run(manifest, scens)
        print(text)
        if not args.no_write:
            (path / "analysis.txt").write_text(text + "\n")
            print(f"\n[analysis written to {path / 'analysis.txt'}]")

    if args.compare and len(loaded) >= 2:
        text = format_compare(loaded)
        print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
