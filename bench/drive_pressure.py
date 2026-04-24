"""CLI that drives one or more APC-pressure scenarios against a live
inferlet endpoint and writes capture-run directories compatible with
bench.analyze_pressure.

Phase 3.0 — Task H. Separate from bench.driver on purpose: bench.driver
shells out to `hermes chat -q` which injects its own system prompt and
tool schemas; the pressure scenarios need direct /v1/chat/completions
POSTs with precisely-shaped synthetic prompts. Sharing the driver's
capture-directory layout keeps existing analyzers usable against both
kinds of runs.

Preset sweeps:
  The sweep matrix from the plan doc (`docs/plans/2026-04-23-phase-
  3.0-apc-pressure.md`) is encoded under `PRESET_SWEEPS` so a single
  invocation can reproduce the decision-ready picture:

      uv run python -m bench.drive_pressure \\
          --preset phase-3.0-initial \\
          --base-url http://<pod>:<port> \\
          --model Qwen/Qwen2.5-72B-Instruct-AWQ

  The preset is the unit of budgetary approval. Ad-hoc sweeps use
  --scenario flags directly.

Budget guard:
  --max-cost-usd aborts before firing if the estimated cost exceeds
  the provided ceiling. The estimate is intentionally conservative (see
  _estimate_cost_usd doc) — better to refuse a legit run and force an
  explicit override than to silently overspend on a pressure sweep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from bench.pressure_scenarios import (
    PressureScenario,
    run_pressure_scenario,
)
from bench.analyze_pressure import analyze_scenario


# ---------------------------------------------------------------------------
# Presets — from the plan doc's sweep matrix
# ---------------------------------------------------------------------------


def _scen(
    slug: str,
    *,
    n_users: int,
    l_tokens: int,
    pattern: str,
    register: bool = False,
    rounds: int = 3,
    common_prefix_tokens: int = 0,
) -> PressureScenario:
    return PressureScenario(
        slug=slug,
        n_users=n_users,
        l_tokens=l_tokens,
        pattern=pattern,  # type: ignore[arg-type]
        common_prefix_tokens=common_prefix_tokens,
        rounds=rounds,
        register_common_body=register,
    )


# The canonical sweep documented in docs/plans/2026-04-23-phase-3.0-apc-
# pressure.md, Task 3.0.I. Keep this list in sync with the plan — the
# two names (plan row vs preset key) form the handshake between "plan
# approved" and "what the script will run tonight."
PRESET_SWEEPS: dict[str, list[PressureScenario]] = {
    "phase-3.0-initial": [
        _scen("bench-apc-pressure-50u-4ktok-round-robin",
              n_users=50, l_tokens=4_000, pattern="round-robin"),
        _scen("bench-apc-pressure-200u-4ktok-round-robin",
              n_users=200, l_tokens=4_000, pattern="round-robin"),
        _scen("bench-apc-pressure-500u-4ktok-round-robin",
              n_users=500, l_tokens=4_000, pattern="round-robin"),
        _scen("bench-apc-pressure-500u-4ktok-round-robin-pinned",
              n_users=500, l_tokens=4_000, pattern="round-robin",
              common_prefix_tokens=512, register=True),
        _scen("bench-apc-pressure-500u-4ktok-zipfian",
              n_users=500, l_tokens=4_000, pattern="zipfian"),
        _scen("bench-apc-pressure-200u-16ktok-round-robin",
              n_users=200, l_tokens=16_000, pattern="round-robin"),
    ],
    # Smoke — used to validate the runner against a fresh pod before
    # committing budget to the full sweep. Do NOT ship numbers from
    # this as a measurement; it's single-digit users.
    "phase-3.0-smoke": [
        _scen("bench-apc-pressure-3u-500tok-round-robin",
              n_users=3, l_tokens=500, pattern="round-robin", rounds=2),
    ],
}


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def _estimate_cost_usd(
    scenarios: Iterable[PressureScenario],
    *,
    hourly_usd: float,
    prefill_tps: int,
    seconds_per_request_overhead: float,
) -> float:
    """Upper-bound the cost of a sweep.

    Conservative on purpose — rounds up rather than down. Two sources
    of work per request:
      - prefill: `l_tokens` / `prefill_tps` seconds. This is the
        dominant cost at our sizes; decode is bounded by max_tokens=8.
      - fixed overhead: HTTP roundtrip, tokenizer, queue wait. Modeled
        as a constant `seconds_per_request_overhead` rather than
        network-stack-dependent math.

    Cache hits REDUCE prefill cost in practice; we intentionally ignore
    this in the estimate. A sweep that's "pay for everything" safe is
    also safe when APC is warm.
    """
    total_seconds = 0.0
    for s in scenarios:
        n_requests = s.n_users * s.rounds
        prefill_s = (s.l_tokens / max(prefill_tps, 1)) * n_requests
        overhead_s = seconds_per_request_overhead * n_requests
        total_seconds += prefill_s + overhead_s
    return (total_seconds / 3600.0) * hourly_usd


# ---------------------------------------------------------------------------
# Run-layout helpers (minimal reimpl so we don't import bench.driver's
# private helpers — that file is hermes-CLI-oriented and carries env
# handling we don't want)
# ---------------------------------------------------------------------------


def _now_iso_compact() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _build_run_dir(captures_root: Path, backend_tag: str, preset_tag: str) -> Path:
    run_id = f"{_now_iso_compact()}_{backend_tag}_{preset_tag}"
    d = captures_root / "runs" / run_id
    (d / "scenarios").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_sweep(
    scenarios: list[PressureScenario],
    *,
    base_url: str,
    model: str,
    run_dir: Path,
    kv_capacity_tokens: int | None,
) -> list[dict]:
    """Execute the sweep and return per-scenario summary dicts."""
    scenarios_root = run_dir / "scenarios"
    summaries: list[dict] = []

    for s in scenarios:
        scen_dir = scenarios_root / s.slug
        scen_dir.mkdir(parents=True, exist_ok=True)
        capture_file = scen_dir / "capture.jsonl"

        print(f"=== {s.slug} (n={s.n_users}, L={s.l_tokens}, "
              f"{s.pattern}, arm={'pinned' if s.register_common_body else 'baseline'}) ===",
              flush=True)
        t0 = time.monotonic()
        summary = run_pressure_scenario(
            s,
            base_url=base_url,
            model=model,
            capture_file=capture_file,
        )
        summary["wall_clock_s_actual"] = round(time.monotonic() - t0, 3)

        # meta.json next to capture.jsonl — analyze_pressure reads it.
        (scen_dir / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")

        # Render a per-scenario report immediately so a long sweep emits
        # intermediate results; operator can `less` the reports while
        # the rest of the sweep runs.
        _, report = analyze_scenario(scen_dir, kv_capacity_tokens=kv_capacity_tokens)
        (scen_dir / "pressure-report.md").write_text(report)

        print(f"    ok={summary['n_successes']}/{summary['n_requests']}  "
              f"wall={summary['wall_clock_s_actual']:.1f}s  "
              f"arm={summary['arm']}  register_ok={summary['register_ok']}",
              flush=True)
        summaries.append(summary)

    return summaries


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bench.drive_pressure")
    p.add_argument("--base-url", required=True,
                   help="Inferlet base URL, e.g. http://pod.example:18095")
    p.add_argument("--model", required=True,
                   help="Model name passed in the chat completions request.")

    # Selecting scenarios — mutually exclusive across preset and adhoc.
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--preset", choices=sorted(PRESET_SWEEPS.keys()),
                     help="Run a canonical sweep (see docs/plans/).")
    sel.add_argument("--scenario",
                     help=("Run a single ad-hoc scenario as "
                           "N,L,pattern[,pinned[,rounds]] — e.g. "
                           "500,4000,round-robin,pinned,3"))

    # Capture-dir + tags.
    p.add_argument("--captures-root", type=Path,
                   default=Path(os.environ.get("CAPTURES_ROOT", "captures")))
    p.add_argument("--backend-tag",
                   default=os.environ.get("BACKEND_TAG", "unknown-backend"))
    p.add_argument("--preset-tag",
                   default=None,
                   help="Run-id suffix (default: --preset name or 'adhoc').")

    # Budget / estimation knobs.
    p.add_argument("--max-cost-usd", type=float, default=None,
                   help="Refuse to run if estimated cost exceeds this.")
    p.add_argument("--hourly-usd", type=float, default=3.0,
                   help="GPU rental rate for cost estimation (default A100-80GB).")
    p.add_argument("--prefill-tps", type=int, default=30_000,
                   help="Assumed prefill throughput for cost estimation.")
    p.add_argument("--request-overhead-s", type=float, default=0.05,
                   help="Per-request fixed overhead for cost estimation.")
    p.add_argument("--kv-capacity-tokens", type=int, default=None,
                   help="Passed through to analyze_pressure for pressure-ratio reporting.")
    p.add_argument("--yes", action="store_true",
                   help="Skip interactive confirm-before-firing prompt.")

    args = p.parse_args(argv)

    if args.preset:
        scenarios = list(PRESET_SWEEPS[args.preset])
        preset_tag = args.preset_tag or args.preset
    else:
        scenarios = [_parse_adhoc_scenario(args.scenario)]
        preset_tag = args.preset_tag or "adhoc"

    # Cost preflight.
    est = _estimate_cost_usd(
        scenarios,
        hourly_usd=args.hourly_usd,
        prefill_tps=args.prefill_tps,
        seconds_per_request_overhead=args.request_overhead_s,
    )
    print(f"[preflight] scenarios={len(scenarios)}  est_cost_usd=${est:.2f}  "
          f"(assumes {args.hourly_usd}/hr, {args.prefill_tps} prefill_tps)")
    if args.max_cost_usd is not None and est > args.max_cost_usd:
        print(f"[abort] estimated cost ${est:.2f} > --max-cost-usd ${args.max_cost_usd:.2f}",
              file=sys.stderr)
        return 2

    # Confirmation step — live pod = real money. Skippable for
    # automation via --yes. We explicitly print before prompting so CI
    # / piped stdin sees the preflight even when the prompt never
    # flushes.
    if not args.yes:
        try:
            resp = input("[confirm] fire sweep against live pod? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("[abort] user did not confirm", file=sys.stderr)
            return 3

    run_dir = _build_run_dir(args.captures_root, args.backend_tag, preset_tag)
    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "run_id": run_dir.name,
        "kind": "pressure-sweep",
        "preset": args.preset,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "base_url": args.base_url,
        "model": args.model,
        "git": {
            "sha": _git("rev-parse", "HEAD", cwd=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
            "dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        },
        "cost_estimate": {
            "usd": round(est, 4),
            "hourly_usd": args.hourly_usd,
            "prefill_tps": args.prefill_tps,
            "request_overhead_s": args.request_overhead_s,
        },
        "scenarios": [
            {
                "slug": s.slug, "n_users": s.n_users, "l_tokens": s.l_tokens,
                "pattern": s.pattern, "rounds": s.rounds,
                "common_prefix_tokens": s.common_prefix_tokens,
                "register_common_body": s.register_common_body,
                "seed": s.seed,
            }
            for s in scenarios
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summaries = run_sweep(
        scenarios,
        base_url=args.base_url,
        model=args.model,
        run_dir=run_dir,
        kv_capacity_tokens=args.kv_capacity_tokens,
    )

    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["summaries"] = summaries
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[done] run_dir={run_dir}", flush=True)
    any_errors = any(s.get("n_errors", 0) > 0 for s in summaries)
    return 0 if not any_errors else 1


def _parse_adhoc_scenario(spec: str) -> PressureScenario:
    """Parse N,L,pattern[,pinned[,rounds]] into a PressureScenario.

    Minimal parser — the preset matrix is the blessed path. This exists
    for one-off probes ("what happens at 1000 users?") where paying
    the full plan-update / preset-bump cost is overkill.
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 3:
        raise ValueError(f"--scenario must be N,L,pattern[,pinned[,rounds]]: {spec!r}")
    n = int(parts[0])
    L = int(parts[1])
    pattern = parts[2]
    pinned = len(parts) >= 4 and parts[3].lower() in ("pinned", "true", "1")
    rounds = int(parts[4]) if len(parts) >= 5 else 3
    return _scen(
        f"bench-apc-pressure-{n}u-{L}tok-{pattern}{'-pinned' if pinned else ''}",
        n_users=n, l_tokens=L, pattern=pattern,
        register=pinned, rounds=rounds,
        common_prefix_tokens=min(L // 4, 512) if pinned else 0,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
