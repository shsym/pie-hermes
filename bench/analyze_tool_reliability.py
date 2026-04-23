"""Tool-call reliability analyzer for bench captures.

Walks capture.jsonl files in a run directory (captures/runs/<ts>_<tag>/)
and emits four metrics per model×scenario:

  1. syntax_valid_rate   — fraction of emitted tool_calls whose `function.arguments`
                            parses as JSON.
  2. selection_correct   — on scenarios with an unambiguous expected tool,
                            did turn-1's FIRST tool_call use the expected name?
                            (n/a for scenarios without a mapping.)
  3. completion          — did the final turn finish with `finish_reason == "stop"`
                            (scenario reached a user-facing answer, not stuck in
                            a tool-call loop or truncated).
  4. no_tool_discipline  — on scenarios where no tool is warranted
                            (e.g. shallow-hello), did turn-1 refrain from
                            emitting tool_calls?

Capture format (per bench/capture_hook.py, streaming mode):
  {session_id, ts_start, ts_end, stream: true,
   kwargs: {model, messages, tools, ...},
   chunks: [<OpenAI chat.completion.chunk>, ...]}

Usage:
  python -m bench.analyze_tool_reliability <run_dir> [<run_dir> ...]
  python -m bench.analyze_tool_reliability --out /tmp/report.json <run_dir>

Outputs:
  - Markdown table to stdout (per model × scenario, plus per-model rollup).
  - Optional JSON to --out for machine-readable consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Scenario → expected tool name. Only list scenarios where the expected tool
# is unambiguous from the prompt (used for the selection metric). Scenarios
# not listed here are skipped for selection but still counted for the other
# three metrics.
EXPECTED_TOOL: dict[str, str] = {
    "shallow-file-head": "read_file",
    "shallow-terminal": "terminal",
    "deep-git-audit": "terminal",          # first-turn tool
    "deep-multi-read": "read_file",
    "deep-search-read": "search_files",
    "deep-breadth-audit": "search_files",
}

# Scenarios where NO tool call should occur. Used for the discipline metric.
NO_TOOL_SCENARIOS: set[str] = {"shallow-hello"}

# Scenarios we deliberately skip (Phase-2 substrate probes; not about tool-use
# reliability of the base model).
EXCLUDED_SCENARIOS: set[str] = {
    "gateway-ephemeral-chat",
    "gateway-ephemeral-chat-3turn",
    "probe-ephemeral-handle-recall",
    "probe-read-context-one-call",
    "probe-skills-handle-recall",
}


@dataclass
class TurnStats:
    """Stats for one capture turn (= one chat.completions.create call)."""
    tool_calls_emitted: int = 0
    tool_calls_valid_json: int = 0
    first_tool_name: str | None = None
    finish_reason: str | None = None
    content_len: int = 0


@dataclass
class ScenarioStats:
    scenario: str
    model: str
    run_tag: str
    turns: list[TurnStats] = field(default_factory=list)
    error: str | None = None

    # Derived metrics (filled by compute())
    syntax_valid_rate: float | None = None      # None = no tool_calls observed
    selection_correct: bool | None = None       # None = no expected-tool mapping
    completion: bool | None = None
    no_tool_discipline: bool | None = None      # None = not a no-tool scenario

    def compute(self) -> None:
        total_tc = sum(t.tool_calls_emitted for t in self.turns)
        total_valid = sum(t.tool_calls_valid_json for t in self.turns)
        self.syntax_valid_rate = (total_valid / total_tc) if total_tc else None

        # Selection: compare turn-1 first tool to expected.
        if self.scenario in EXPECTED_TOOL and self.turns:
            first_tool = self.turns[0].first_tool_name
            self.selection_correct = first_tool == EXPECTED_TOOL[self.scenario]

        # Completion: final turn's finish_reason should be "stop".
        if self.turns:
            self.completion = self.turns[-1].finish_reason == "stop"

        # Discipline: shallow-hello etc — turn 1 must not emit tool_calls.
        if self.scenario in NO_TOOL_SCENARIOS and self.turns:
            self.no_tool_discipline = self.turns[0].tool_calls_emitted == 0


def parse_chunks(chunks: list[dict]) -> TurnStats:
    """Reassemble streaming chunks into turn-level stats."""
    ts = TurnStats()
    tool_args_accum: dict[int, list[str]] = {}
    tool_name_by_index: dict[int, str] = {}
    first_tool_name: str | None = None

    for chunk in chunks:
        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError):
            continue
        delta = choice.get("delta") or {}
        fr = choice.get("finish_reason")
        if fr:
            ts.finish_reason = fr
        content = delta.get("content")
        if content:
            ts.content_len += len(content)
        tcs = delta.get("tool_calls") or []
        for tc in tcs:
            idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments") or ""
            if name:
                tool_name_by_index[idx] = name
                if first_tool_name is None:
                    first_tool_name = name
            if args:
                tool_args_accum.setdefault(idx, []).append(args)

    ts.first_tool_name = first_tool_name
    ts.tool_calls_emitted = len(tool_args_accum) if tool_args_accum else len(tool_name_by_index)
    for idx, parts in tool_args_accum.items():
        joined = "".join(parts)
        try:
            json.loads(joined)
            ts.tool_calls_valid_json += 1
        except (json.JSONDecodeError, ValueError):
            pass

    return ts


def parse_non_stream(record: dict) -> TurnStats:
    """Non-streaming record (chunks empty; full response under 'response')."""
    ts = TurnStats()
    resp = record.get("response") or {}
    try:
        choice = resp["choices"][0]
    except (KeyError, IndexError):
        return ts
    ts.finish_reason = choice.get("finish_reason")
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    ts.content_len = len(content)
    for tc in msg.get("tool_calls") or []:
        ts.tool_calls_emitted += 1
        fn = tc.get("function") or {}
        if fn.get("name") and ts.first_tool_name is None:
            ts.first_tool_name = fn["name"]
        args = fn.get("arguments") or ""
        try:
            json.loads(args)
            ts.tool_calls_valid_json += 1
        except (json.JSONDecodeError, ValueError):
            pass
    return ts


def analyze_scenario(path: Path, run_tag: str) -> ScenarioStats:
    scenario = path.parent.name
    stats = ScenarioStats(scenario=scenario, model="", run_tag=run_tag)
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not stats.model:
                    stats.model = rec.get("kwargs", {}).get("model") or "unknown"
                if rec.get("stream"):
                    stats.turns.append(parse_chunks(rec.get("chunks") or []))
                else:
                    stats.turns.append(parse_non_stream(rec))
    except Exception as e:
        stats.error = f"{type(e).__name__}: {e}"
    stats.compute()
    return stats


def discover_scenarios(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "scenarios").glob("*/capture.jsonl"))


def render_markdown(all_stats: list[ScenarioStats]) -> str:
    lines = ["# Tool-call reliability report", ""]

    by_model: dict[str, list[ScenarioStats]] = {}
    for s in all_stats:
        by_model.setdefault(f"{s.model}  [{s.run_tag}]", []).append(s)

    for model, stats in by_model.items():
        lines += [f"## {model}", ""]
        lines += [
            "| Scenario | Turns | TC emitted | Syntax OK | Selection | Completion | Discipline | Notes |",
            "|---|---:|---:|---|---|---|---|---|",
        ]
        syntax_rates = []
        completions = []
        selections = []
        disciplines = []
        for s in sorted(stats, key=lambda x: x.scenario):
            if s.scenario in EXCLUDED_SCENARIOS:
                continue
            turns = len(s.turns)
            tc = sum(t.tool_calls_emitted for t in s.turns)
            valid = sum(t.tool_calls_valid_json for t in s.turns)
            synt = (
                f"{valid}/{tc} ({100*valid/tc:.0f}%)" if tc else "—"
            )
            if s.selection_correct is None:
                sel = "—"
            else:
                sel = "✅" if s.selection_correct else f"❌ got={s.turns[0].first_tool_name}"
                selections.append(bool(s.selection_correct))
            if s.completion is None:
                comp = "—"
            else:
                comp = "✅" if s.completion else f"❌ fr={s.turns[-1].finish_reason}"
                completions.append(bool(s.completion))
            if s.no_tool_discipline is None:
                disc = "—"
            else:
                disc = "✅" if s.no_tool_discipline else "❌ emitted tools"
                disciplines.append(bool(s.no_tool_discipline))
            notes = s.error or ""
            lines.append(
                f"| {s.scenario} | {turns} | {tc} | {synt} | {sel} | {comp} | {disc} | {notes} |"
            )
            if tc:
                syntax_rates.append(valid / tc)

        def rate(xs):
            return f"{100*sum(xs)/len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"

        lines += [
            "",
            "### Rollup",
            f"- syntax_valid aggregate: {rate(syntax_rates) if syntax_rates else 'n/a'}",
            f"- selection correct: {rate(selections)}",
            f"- completion: {rate(completions)}",
            f"- no-tool discipline: {rate(disciplines)}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path, help="captures/runs/<ts>_<tag> directories")
    ap.add_argument("--out", type=Path, help="write JSON summary here")
    args = ap.parse_args(argv)

    all_stats: list[ScenarioStats] = []
    for rd in args.run_dirs:
        if not rd.is_dir():
            print(f"skip (not dir): {rd}", file=sys.stderr)
            continue
        run_tag = rd.name
        for p in discover_scenarios(rd):
            all_stats.append(analyze_scenario(p, run_tag))

    if not all_stats:
        print("No capture.jsonl files found.", file=sys.stderr)
        return 1

    print(render_markdown(all_stats))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([asdict(s) for s in all_stats], indent=2, default=str)
        )
        print(f"\nJSON: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
