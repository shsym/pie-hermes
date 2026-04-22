#!/usr/bin/env python3
"""Project $ cost of Hermes sessions across vendor pricing.

Reads a pie-hermes run directory (same layout as analyze.py) and multiplies
est prompt/completion tokens by each model's input/output $/M pricing to
produce per-scenario and aggregate cost.

Usage: python bench/cost_projection.py <run_dir>

Tokens are estimated via chars/4 (same as analyze.py). Prices are a static
snapshot — edit PRICING below to refresh. All costs in USD per request /
session assuming zero prompt-cache hit (worst case).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.analyze import load_run, _tok  # reuse existing parser + tokenizer


# USD per 1M tokens, snapshot 2026-04. Edit to refresh.
# Format: model_name -> (input_$per_M, output_$per_M)
PRICING = {
    # Anthropic Claude 4.x family
    "claude-opus-4-7":    (15.00, 75.00),
    "claude-sonnet-4-6":  ( 3.00, 15.00),
    "claude-haiku-4-5":   ( 1.00,  5.00),
    # OpenAI
    "gpt-4o":             ( 2.50, 10.00),
    "gpt-4.1":            ( 3.00, 12.00),
    "gpt-4.1-mini":       ( 0.40,  1.60),
    # Self-hosted Qwen on A100-SXM4 @ $1.19/hr RunPod list
    #   Effective $/M derived from throughput, not vendor price.
    #   Rough: ~50 output-tok/s on Qwen2.5-72B-AWQ + 1 A100 → $0.24/hr/1k-tok/s
    #   ≈ $1.19/hr ÷ (50 tok/s × 3600 s/hr) = $6.61 per M output tok at 100% util
    #   Input is usually prefill-dominant & faster; use $0.50/M as ballpark.
    "qwen2.5-72b-awq-selfhost":  ( 0.50,  6.61),
}


def est_completion_tokens(scenario_dir: Path) -> dict[int, int]:
    """Extract est completion tokens per call from stream chunks.

    Returns {call_index: est_completion_tok}. Sums len(delta.content) across
    all chunks per request, divides by 4.
    """
    import json
    out = {}
    with (scenario_dir / "capture.jsonl").open() as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            total_chars = 0
            for ch in r.get("chunks") or []:
                if not isinstance(ch, dict):
                    continue
                for choice in ch.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # content
                    c = delta.get("content") or ""
                    total_chars += len(c)
                    # tool-call arguments (streamed JSON)
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        total_chars += len(fn.get("name") or "")
                        total_chars += len(fn.get("arguments") or "")
            out[i] = total_chars // 4
    return out


def main(run_dir: Path) -> int:
    manifest, scenarios = load_run(run_dir)

    # Prompt tokens per call come from analyze.py's breakdown.
    # Completion tokens come from the capture jsonl chunks.
    rows = []  # (slug, call_idx, prompt_tok, completion_tok)
    for s in scenarios:
        compl = est_completion_tokens(run_dir / "scenarios" / s.slug)
        for i, c in enumerate(s.calls):
            rows.append((s.slug, i, c.est_tokens, compl.get(i, 0)))

    total_prompt = sum(r[2] for r in rows)
    total_compl  = sum(r[3] for r in rows)

    # Per-scenario (one "session" = one scenario's total call cost)
    scen_totals = {}
    for slug, _i, p, c in rows:
        a, b = scen_totals.get(slug, (0, 0))
        scen_totals[slug] = (a + p, b + c)

    print(f"Run: {manifest['run_id']}")
    print(f"Backend: {manifest['backend'].get('model')} ({manifest['backend'].get('provider')})")
    print(f"Calls: {len(rows)}  total prompt={total_prompt:,} tok  total compl={total_compl:,} tok\n")

    # Per-model aggregate (worst case: no prompt-cache hits)
    print("=" * 82)
    print("COST PER SESSION (= one scenario's worth of calls), no prompt-cache")
    print("=" * 82)
    header = f"{'model':<28} {'in$/M':>7} {'out$/M':>7}   " + "  ".join(f"{s.slug[:14]:>14}" for s in scenarios) + f"   {'TOTAL':>8}"
    print(header)
    print("-" * len(header))
    for model, (pin, pout) in PRICING.items():
        per_scen = []
        for s in scenarios:
            p, c = scen_totals[s.slug]
            cost = p / 1e6 * pin + c / 1e6 * pout
            per_scen.append(cost)
        total = sum(per_scen)
        cells = "  ".join(f"${v:>13.4f}" for v in per_scen)
        print(f"{model:<28} {pin:>7.2f} {pout:>7.2f}   {cells}   ${total:>7.4f}")

    # Per-call average (what a single Hermes turn costs)
    print()
    print("=" * 82)
    print("COST PER CALL (averaged across all 28 calls)")
    print("=" * 82)
    avg_prompt = total_prompt / len(rows)
    avg_compl = total_compl / len(rows)
    print(f"avg prompt tok/call = {avg_prompt:,.0f}  (dominated by system + tool schemas)")
    print(f"avg compl  tok/call = {avg_compl:,.0f}\n")
    print(f"{'model':<28} {'$ per call':>12}   {'calls / $1':>12}   {'$1k Hermes sessions*':>22}")
    print("-" * 82)
    for model, (pin, pout) in PRICING.items():
        per_call = avg_prompt / 1e6 * pin + avg_compl / 1e6 * pout
        per_session = per_call * (len(rows) / len(scenarios))  # avg calls/scenario
        calls_per_dollar = 1 / per_call if per_call > 0 else 0
        sessions_1k_usd = 1000 / per_session if per_session > 0 else 0
        print(f"{model:<28} {per_call:>12.5f}   {calls_per_dollar:>12,.0f}   {sessions_1k_usd:>22,.0f}")
    print(f"\n* a 'session' here = {len(rows)/len(scenarios):.1f} avg calls, matching round-3 scenario mix")

    # Prompt-cache sensitivity — show Claude savings assuming 90% cache hit on sys+schemas
    print()
    print("=" * 82)
    print("WITH PROMPT CACHING (Anthropic ~90% discount on cached read)")
    print("=" * 82)
    # Assume ~12.8K of every prompt is cacheable (system + tool schemas). Growth is not.
    CACHEABLE_PER_CALL = 12800  # conservative; round-3 turn-1 is 12,841–12,897
    cacheable_tot = min(total_prompt, CACHEABLE_PER_CALL * len(rows))
    uncached_tot = total_prompt - cacheable_tot
    print(f"Assumed cacheable: first {CACHEABLE_PER_CALL:,} tok of every call (system + tool schemas)")
    print(f"  cacheable total   = {cacheable_tot:,} tok")
    print(f"  non-cached growth = {uncached_tot:,} tok\n")
    # Anthropic prompt caching:
    #   cache_write = first call of each cache prefix, billed at 1.25× input price
    #   cache_read  = subsequent calls, billed at 0.10× input price
    # Treat each scenario as a new cache prefix (independent session), so 7 writes total.
    N_WRITES = len(scenarios)
    N_READS  = len(rows) - N_WRITES
    for model, (pin, pout) in PRICING.items():
        if "claude" not in model:
            continue
        cache_write_cost = N_WRITES * CACHEABLE_PER_CALL / 1e6 * pin * 1.25
        cache_read_cost  = N_READS  * CACHEABLE_PER_CALL / 1e6 * pin * 0.10
        growth_cost      = uncached_tot / 1e6 * pin
        compl_cost       = total_compl / 1e6 * pout
        cached_total     = cache_write_cost + cache_read_cost + growth_cost + compl_cost
        nocache_total    = total_prompt / 1e6 * pin + total_compl / 1e6 * pout
        savings_pct = (1 - cached_total / nocache_total) * 100
        # Also show savings for a hypothetical single long session (1 write, 27 reads)
        single_write   = CACHEABLE_PER_CALL / 1e6 * pin * 1.25
        single_reads   = (len(rows) - 1) * CACHEABLE_PER_CALL / 1e6 * pin * 0.10
        single_total   = single_write + single_reads + growth_cost + compl_cost
        single_savings = (1 - single_total / nocache_total) * 100
        print(f"{model:<22} nocache ${nocache_total:>7.4f}  7-scens cached ${cached_total:>7.4f} ({savings_pct:>4.1f}% save)  1-session cached ${single_total:>7.4f} ({single_savings:>4.1f}% save)")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: cost_projection.py <run_dir>")
    sys.exit(main(Path(sys.argv[1])))
