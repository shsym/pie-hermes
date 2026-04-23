#!/usr/bin/env python3
"""Behavioral diff between two capture-runs.

Given a baseline run (e.g. Phase 1 / task #25 round2) and a candidate run
(e.g. Phase 2.0), walk every scenario that appears in both and compare:

    - exit_code           (meta.json)
    - n_completions_calls (meta.json)
    - final assistant reply text (from capture.jsonl; exact + char-ngram similarity)
    - tool_call sequence (names, in order, across all assistant messages)
    - tool_call arg keys  (set of top-level argument keys per tool invocation)

Scenarios present in only one run are reported separately so the operator
can see coverage asymmetries.

Caveat: inference is non-deterministic unless the capture was taken at
temp=0. `--temp-tolerance` controls what counts as a "material" text
diff — at default temperatures, expect the report to flag many text
changes as informational-only. Structural axes (exit_code, tool_call
names, n_calls) are the reliable signals at any temperature.

Usage:
    python -m bench.diff_responses BASELINE_DIR CANDIDATE_DIR
    python -m bench.diff_responses BASELINE_DIR CANDIDATE_DIR --json
    python -m bench.diff_responses BASELINE_DIR CANDIDATE_DIR --only shallow-hello deep-git-audit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Reuse the streaming reassembly from check_probes so the two analyzers
# see the same view of assistant messages.
from bench.check_probes import _all_tool_calls, _final_reply_text, _iter_assistant_messages


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDiff:
    slug: str
    # Structural — exact diffs are material at any temperature.
    exit_code_diff: tuple[int, int] | None = None          # (base, cand) if different
    n_calls_diff: tuple[int, int] | None = None
    tool_name_seq_diff: tuple[list[str], list[str]] | None = None
    tool_arg_keys_diff: list[dict] = field(default_factory=list)
    # Textual — informational unless temp=0.
    text_exact_match: bool = False
    text_similarity: float = 0.0                            # 0.0 – 1.0
    base_reply_len: int = 0
    cand_reply_len: int = 0

    @property
    def has_structural_diff(self) -> bool:
        return bool(
            self.exit_code_diff
            or self.n_calls_diff
            or self.tool_name_seq_diff
            or self.tool_arg_keys_diff
        )


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _tool_name_sequence(calls: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in calls]


def _tool_arg_key_sets(calls: list[tuple[str, dict]]) -> list[tuple[str, frozenset[str]]]:
    return [(name, frozenset(args.keys())) for name, args in calls]


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # SequenceMatcher on character n-grams is fine for the short replies
    # our scenarios produce; it's the same metric Python's difflib exposes.
    return SequenceMatcher(None, a, b).ratio()


def diff_scenario(slug: str, base_dir: Path, cand_dir: Path) -> ScenarioDiff:
    r = ScenarioDiff(slug=slug)

    base_meta = json.loads((base_dir / "meta.json").read_text())
    cand_meta = json.loads((cand_dir / "meta.json").read_text())
    if base_meta.get("exit_code") != cand_meta.get("exit_code"):
        r.exit_code_diff = (base_meta.get("exit_code"), cand_meta.get("exit_code"))
    if base_meta.get("n_completions_calls") != cand_meta.get("n_completions_calls"):
        r.n_calls_diff = (
            base_meta.get("n_completions_calls"),
            cand_meta.get("n_completions_calls"),
        )

    base_rows = _load_rows(base_dir / "capture.jsonl")
    cand_rows = _load_rows(cand_dir / "capture.jsonl")
    base_asst = _iter_assistant_messages(base_rows)
    cand_asst = _iter_assistant_messages(cand_rows)

    base_calls = _all_tool_calls(base_asst)
    cand_calls = _all_tool_calls(cand_asst)
    base_names = _tool_name_sequence(base_calls)
    cand_names = _tool_name_sequence(cand_calls)
    if base_names != cand_names:
        r.tool_name_seq_diff = (base_names, cand_names)

    # Arg-key comparison is positional — same index ⇒ same tool name ⇒ compare
    # key sets. If the tool name sequence diverges we already reported it;
    # skip arg-key noise in that case.
    if not r.tool_name_seq_diff:
        base_keys = _tool_arg_key_sets(base_calls)
        cand_keys = _tool_arg_key_sets(cand_calls)
        for i, (b, c) in enumerate(zip(base_keys, cand_keys)):
            if b[1] != c[1]:
                r.tool_arg_keys_diff.append({
                    "index": i,
                    "tool": b[0],
                    "base_keys": sorted(b[1]),
                    "cand_keys": sorted(c[1]),
                    "base_only": sorted(b[1] - c[1]),
                    "cand_only": sorted(c[1] - b[1]),
                })

    base_reply = _final_reply_text(base_asst)
    cand_reply = _final_reply_text(cand_asst)
    r.base_reply_len = len(base_reply)
    r.cand_reply_len = len(cand_reply)
    r.text_exact_match = base_reply == cand_reply
    r.text_similarity = _text_similarity(base_reply, cand_reply)
    return r


# ---------------------------------------------------------------------------
# Cross-run driver
# ---------------------------------------------------------------------------

def _scenario_dirs(run_dir: Path) -> dict[str, Path]:
    scen = run_dir / "scenarios"
    if not scen.is_dir():
        return {}
    return {d.name: d for d in scen.iterdir() if d.is_dir()}


def diff_runs(base_run: Path, cand_run: Path,
              only: list[str] | None = None) -> dict[str, Any]:
    base_scens = _scenario_dirs(base_run)
    cand_scens = _scenario_dirs(cand_run)
    shared = sorted(base_scens.keys() & cand_scens.keys())
    base_only = sorted(base_scens.keys() - cand_scens.keys())
    cand_only = sorted(cand_scens.keys() - base_scens.keys())

    if only:
        shared = [s for s in shared if s in set(only)]

    diffs = [diff_scenario(slug, base_scens[slug], cand_scens[slug]) for slug in shared]
    return {
        "base_run": str(base_run),
        "cand_run": str(cand_run),
        "base_only_scenarios": base_only,
        "cand_only_scenarios": cand_only,
        "diffs": diffs,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_diff(d: ScenarioDiff) -> str:
    lines = [f"  [{'STRUCT' if d.has_structural_diff else 'text-only' if not d.text_exact_match else 'match'}] {d.slug}"]
    if d.exit_code_diff:
        lines.append(f"      exit_code:   base={d.exit_code_diff[0]!r}  cand={d.exit_code_diff[1]!r}")
    if d.n_calls_diff:
        lines.append(f"      n_calls:     base={d.n_calls_diff[0]}  cand={d.n_calls_diff[1]}")
    if d.tool_name_seq_diff:
        b, c = d.tool_name_seq_diff
        lines.append(f"      tool_names:  base={b!r}")
        lines.append(f"                   cand={c!r}")
    for ak in d.tool_arg_keys_diff:
        lines.append(f"      tool[{ak['index']}] {ak['tool']!r} args differ:")
        if ak["base_only"]:
            lines.append(f"          - base only: {ak['base_only']}")
        if ak["cand_only"]:
            lines.append(f"          + cand only: {ak['cand_only']}")
    if not d.text_exact_match:
        lines.append(f"      reply text:  base_len={d.base_reply_len}  cand_len={d.cand_reply_len}  similarity={d.text_similarity:.3f}")
    return "\n".join(lines)


def fmt_report(result: dict[str, Any]) -> str:
    base = result["base_run"]
    cand = result["cand_run"]
    diffs: list[ScenarioDiff] = result["diffs"]
    out = [f"base: {base}", f"cand: {cand}"]
    if result["base_only_scenarios"]:
        out.append(f"base-only scenarios (skipped): {result['base_only_scenarios']}")
    if result["cand_only_scenarios"]:
        out.append(f"cand-only scenarios (skipped): {result['cand_only_scenarios']}")
    out.append("")
    n_struct = sum(1 for d in diffs if d.has_structural_diff)
    n_text_only = sum(1 for d in diffs if not d.has_structural_diff and not d.text_exact_match)
    n_match = sum(1 for d in diffs if d.text_exact_match and not d.has_structural_diff)
    out.append(f"{len(diffs)} shared scenarios: {n_match} exact-match, {n_text_only} text-diff only, {n_struct} structural-diff")
    out.append("")
    for d in diffs:
        out.append(_fmt_diff(d))
    return "\n".join(out)


def _diff_to_json_safe(d: ScenarioDiff) -> dict:
    return {
        "slug": d.slug,
        "has_structural_diff": d.has_structural_diff,
        "text_exact_match": d.text_exact_match,
        "text_similarity": round(d.text_similarity, 4),
        "base_reply_len": d.base_reply_len,
        "cand_reply_len": d.cand_reply_len,
        "exit_code_diff": list(d.exit_code_diff) if d.exit_code_diff else None,
        "n_calls_diff": list(d.n_calls_diff) if d.n_calls_diff else None,
        "tool_name_seq_diff": (
            {"base": d.tool_name_seq_diff[0], "cand": d.tool_name_seq_diff[1]}
            if d.tool_name_seq_diff else None
        ),
        "tool_arg_keys_diff": d.tool_arg_keys_diff,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bench.diff_responses")
    p.add_argument("base", type=Path, help="baseline capture-run dir")
    p.add_argument("cand", type=Path, help="candidate capture-run dir")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p.add_argument("--only", nargs="+", default=None,
                   help="restrict to these scenario slugs")
    p.add_argument("--struct-only", action="store_true",
                   help="non-zero exit only on structural diffs (ignore text-only)")
    args = p.parse_args(argv)

    if not args.base.is_dir() or not args.cand.is_dir():
        raise SystemExit("base and cand must both be existing directories")

    result = diff_runs(args.base, args.cand, only=args.only)
    if args.json:
        print(json.dumps({
            **{k: v for k, v in result.items() if k != "diffs"},
            "diffs": [_diff_to_json_safe(d) for d in result["diffs"]],
        }, indent=2))
    else:
        print(fmt_report(result))

    diffs: list[ScenarioDiff] = result["diffs"]
    if args.struct_only:
        return 0 if not any(d.has_structural_diff for d in diffs) else 1
    return 0 if all(d.text_exact_match and not d.has_structural_diff for d in diffs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
