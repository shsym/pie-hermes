"""Analyze a Phase-3.0 pressure capture — eviction cliff, hit rate, TTFT.

Reads a `captures/runs/<RUN_ID>/scenarios/<slug>/capture.jsonl` produced
by `bench.pressure_scenarios.run_pressure_scenario` and turns the raw
row stream into:

  - per-round hit rate (fraction of requests where
    `pie_cache.prefill_tokens_skipped > 0` among the tail tokens)
  - pressure ratio vs nominal capacity (best-effort; requires a
    `--kv-capacity-tokens` hint since pie-vllm doesn't expose the
    number over the SDK — see Task 3.0.E memo)
  - TTFT P50 / P95 across the run
  - arm-over-arm delta when two captures are compared side by side
    (baseline vs pinned).

Design notes:
  - No matplotlib dependency — ASCII-plot style output so the report
    pastes cleanly into the task memo and doesn't introduce a new
    runtime dep for a one-off measurement.
  - We only examine signals the inferlet ALREADY emits today. Anything
    that would require a pie-vllm patch (global eviction count,
    block-manager hit rate) is left as a TODO in the output, not an
    import error, so the analyzer keeps running when those signals are
    absent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Capture row extraction
# ---------------------------------------------------------------------------


def _iter_rows(path: Path) -> Iterable[dict]:
    """Yield parsed capture rows. Silently skips blank lines and rows
    whose JSON is truncated (shouldn't happen but the driver writes
    append-mode, and an interrupted sweep can leave a half-flushed
    terminal row — dropping it is safer than aborting the whole
    analysis)."""
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pie_cache_from_row(row: dict) -> dict | None:
    """Extract `pie_cache` telemetry from either a non-streaming row
    (response.pie_cache) or the streaming final chunk (chunks[*].
    pie_cache). The inferlet attaches it only once per request, so we
    return the first non-empty hit."""
    resp = row.get("response") or {}
    pc = resp.get("pie_cache")
    if isinstance(pc, dict):
        return pc
    for ch in row.get("chunks") or []:
        pc = ch.get("pie_cache")
        if isinstance(pc, dict):
            return pc
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class PressureStats:
    n_rows: int = 0
    n_errors: int = 0
    ttft_s: list[float] = field(default_factory=list)
    # Per-round counters. Key = round index.
    rounds: dict[int, dict[str, int]] = field(default_factory=dict)
    # Per-user hit counters across ALL rounds (used for distribution).
    per_user_hits: dict[int, int] = field(default_factory=dict)
    per_user_total: dict[int, int] = field(default_factory=dict)
    # Totals.
    total_prompt_tokens: int = 0
    total_prefill_tokens_skipped: int = 0
    total_tool_result_tokens_imported: int = 0
    # Highest mode observed — diagnostic for "did we even exercise the
    # non-session path?" (session_kv implies the session path hit,
    # which is reuse-within-session, not APC-like cross-session reuse).
    modes: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)


def _bump(d: dict, key, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def aggregate(rows: Iterable[dict]) -> PressureStats:
    st = PressureStats()
    for row in rows:
        st.n_rows += 1
        if not row.get("ok", True):
            st.n_errors += 1
            continue

        ttft = row.get("ttft_s")
        if isinstance(ttft, (int, float)):
            st.ttft_s.append(float(ttft))

        user_idx = row.get("pressure_user_idx")
        round_idx = row.get("pressure_round")
        pc = _pie_cache_from_row(row) or {}
        mode = pc.get("mode") or "?"
        _bump(st.modes, mode)
        fr = pc.get("fallback_reason")
        if fr:
            _bump(st.fallback_reasons, fr)

        prompt_total = int(pc.get("prompt_tokens_total") or 0)
        skipped = int(pc.get("prefill_tokens_skipped") or 0)
        tool_imported = int(pc.get("tool_result_tokens_imported") or 0)
        st.total_prompt_tokens += prompt_total
        st.total_prefill_tokens_skipped += skipped
        st.total_tool_result_tokens_imported += tool_imported

        # "Hit" = we saved at least one leading block's worth of
        # prefill. Threshold of >0 rather than "≥ page_size" because
        # partial-block hits are still real savings and we don't know
        # the page size from here without a model probe.
        is_hit = skipped > 0
        if round_idx is not None:
            r = st.rounds.setdefault(
                round_idx, {"n": 0, "hits": 0, "skipped_tokens": 0}
            )
            r["n"] += 1
            if is_hit:
                r["hits"] += 1
            r["skipped_tokens"] += skipped

        if user_idx is not None:
            _bump(st.per_user_total, user_idx)
            if is_hit:
                _bump(st.per_user_hits, user_idx)

    return st


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _ascii_bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def format_report(
    meta: dict,
    st: PressureStats,
    *,
    kv_capacity_tokens: int | None = None,
) -> str:
    """Markdown-flavored text report. Written to stdout and/or
    `pressure-report.md` inside the scenario's capture dir."""
    lines: list[str] = []
    slug = meta.get("scenario_slug") or meta.get("slug") or "(unnamed)"
    arm = meta.get("arm") or "baseline"
    n_users = meta.get("n_users")
    l_tokens = meta.get("l_tokens")
    pattern = meta.get("pattern")
    rounds = meta.get("rounds")

    lines.append(f"# Pressure report — {slug}")
    lines.append("")
    lines.append(f"- arm           : **{arm}**")
    lines.append(f"- n_users       : {n_users}")
    lines.append(f"- l_tokens      : {l_tokens}")
    lines.append(f"- pattern       : {pattern}")
    lines.append(f"- rounds        : {rounds}")
    lines.append(f"- requests      : {st.n_rows} (errors: {st.n_errors})")

    # Pressure ratio. Only meaningful if the caller supplies capacity;
    # approximate it as (n_users × l_tokens) / kv_capacity.
    if kv_capacity_tokens and n_users and l_tokens:
        working_set = int(n_users) * int(l_tokens)
        ratio = working_set / kv_capacity_tokens
        lines.append(
            f"- working set   : {working_set:,} tokens "
            f"({ratio:.2f}× of assumed capacity {kv_capacity_tokens:,})"
        )
    else:
        lines.append(
            "- working set   : (pass --kv-capacity-tokens to get pressure ratio; "
            "pie-vllm doesn't expose it over the SDK — see task #42 3.0.E memo)"
        )
    lines.append("")

    # Hit rate per round — the eviction cliff.
    lines.append("## Per-round hit rate (prefill_tokens_skipped > 0)")
    lines.append("")
    for r_idx in sorted(st.rounds.keys()):
        r = st.rounds[r_idx]
        frac = (r["hits"] / r["n"]) if r["n"] else 0.0
        lines.append(
            f"  round {r_idx}  n={r['n']:<5d}  "
            f"hits={r['hits']:<5d}  rate={frac:6.1%}  "
            f"|{_ascii_bar(frac)}|  "
            f"saved_tok={r['skipped_tokens']:,}"
        )
    lines.append("")

    # Inferlet-path mix. Non-trivial counts on session_kv mean the
    # session-auto-derivation path caught some of these — expected for
    # same-bearer runs; surprising for one-shot anonymous traffic.
    lines.append("## Mode breakdown")
    lines.append("")
    for mode, n in sorted(st.modes.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {mode:<24} {n}")
    if st.fallback_reasons:
        lines.append("")
        lines.append("## Fallback reasons")
        lines.append("")
        for reason, n in sorted(st.fallback_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:<24} {n}")
    lines.append("")

    # TTFT — the user-visible latency cost of a miss.
    lines.append("## TTFT (streaming first-content chunk)")
    lines.append("")
    if st.ttft_s:
        lines.append(f"  mean   : {statistics.fmean(st.ttft_s):.3f} s")
        lines.append(f"  median : {_percentile(st.ttft_s, 0.5):.3f} s")
        lines.append(f"  p95    : {_percentile(st.ttft_s, 0.95):.3f} s")
        lines.append(f"  p99    : {_percentile(st.ttft_s, 0.99):.3f} s")
        lines.append(f"  max    : {max(st.ttft_s):.3f} s")
    else:
        lines.append("  (no TTFT data — run with stream=True to measure)")
    lines.append("")

    # Token-level summary.
    lines.append("## Tokens")
    lines.append("")
    skip_frac = (
        st.total_prefill_tokens_skipped / st.total_prompt_tokens
        if st.total_prompt_tokens else 0.0
    )
    lines.append(f"  prompt tokens total     : {st.total_prompt_tokens:,}")
    lines.append(f"  prefill tokens skipped  : {st.total_prefill_tokens_skipped:,} "
                 f"({skip_frac:.1%})")
    lines.append(
        f"  tool-result tokens match: {st.total_tool_result_tokens_imported:,} "
        "(Phase-3.0 detection diagnostic; does NOT reflect actual savings)"
    )
    lines.append("")

    # Per-user hit distribution (quick shape check — are hot users
    # retaining cache while cold users thrash?).
    if st.per_user_total:
        rates = sorted(
            (
                st.per_user_hits.get(u, 0) / st.per_user_total[u]
                for u in st.per_user_total
                if st.per_user_total[u] > 0
            ),
            reverse=True,
        )
        if rates:
            lines.append("## Per-user hit-rate distribution")
            lines.append("")
            lines.append(
                f"  hottest user   : {rates[0]:.1%}\n"
                f"  median user    : {rates[len(rates) // 2]:.1%}\n"
                f"  coldest user   : {rates[-1]:.1%}"
            )
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_meta(scenario_dir: Path) -> dict:
    """Prefer scenario/meta.json; fall back to a minimal stub when the
    driver didn't emit one (e.g., running against a raw capture.jsonl
    produced by a bare invocation of run_pressure_scenario)."""
    meta_path = scenario_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"scenario_slug": scenario_dir.name}


def analyze_scenario(
    scenario_dir: Path,
    *,
    kv_capacity_tokens: int | None = None,
) -> tuple[PressureStats, str]:
    """Aggregate a single scenario's capture.jsonl + render a report."""
    cap = scenario_dir / "capture.jsonl"
    meta = _load_meta(scenario_dir)
    st = aggregate(_iter_rows(cap))
    report = format_report(meta, st, kv_capacity_tokens=kv_capacity_tokens)
    return st, report


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bench.analyze_pressure")
    p.add_argument(
        "scenario_dir",
        type=Path,
        help="captures/runs/<RUN_ID>/scenarios/<slug>/ directory",
    )
    p.add_argument(
        "--kv-capacity-tokens",
        type=int,
        default=None,
        help=(
            "Assumed KV-cache capacity in tokens (for pressure-ratio reporting). "
            "pie-vllm does not expose this over the SDK — pass the value you "
            "started the pod with, or query it out-of-band via "
            "QUERY_BACKEND_STATS on the Control Plane socket."
        ),
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the report to <scenario_dir>/pressure-report.md in addition to stdout.",
    )
    args = p.parse_args(argv)

    st, report = analyze_scenario(
        args.scenario_dir,
        kv_capacity_tokens=args.kv_capacity_tokens,
    )
    if args.write:
        (args.scenario_dir / "pressure-report.md").write_text(report)
    print(report, end="")
    return 0 if st.n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
