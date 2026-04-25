"""CLI for the replay-pressure sweep (Phase 3.0 follow-up).

Fires captured hermes-agent conversations at a live inferlet across
configurable N × pattern × rounds to measure session-KV behavior under
multi-tenant pressure. See `bench.replay_pressure` for the mechanism
notes and the single-slot session-cache discovery.

Preset shapes:
  phase-3.0-replay-sessions  — sequential vs interleaved at matched N,
                               so the per-user hit-rate gap directly
                               quantifies session-cache thrashing.
  phase-3.0-replay-smoke     — minimum 2-user replay for validating
                               the runner against a fresh pod.

Sweeps share the same budget / preflight / confirmation machinery as
`bench.drive_pressure` — kept separate because the replay-source path
is a positional concept (which existing capture to replay) that
doesn't map onto the synthetic-scenario CLI surface.
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

from bench.analyze_pressure import analyze_scenario
from bench.replay_pressure import (
    ReplayScenario,
    load_conversation,
    run_replay_scenario,
)


# Default source for replay — pulled from the pie-hermes sibling worktree
# since this worktree's captures/ is for task/81538's own runs. Overridable
# via --source. Using `deep-search-read` because it's multi-turn (3), has
# a realistic 12.7k-char system + 28 tool schemas, and exercises both
# read_file tool results (large tail tokens) and the conversation-growth
# pattern that session-KV is supposed to benefit.
DEFAULT_SOURCE = Path(
    "/Users/seung-seoblee/Workspace/pie-hermes/captures/runs/"
    "2026-04-21T17-42-49_pie-qwen2.5-72b-awq_round2-deep/"
    "scenarios/deep-search-read/capture.jsonl"
)


def _scen(slug: str, *, source: Path, n_users: int, pattern: str,
          rounds: int = 1, bearer_prefix: str = "replay") -> ReplayScenario:
    return ReplayScenario(
        slug=slug, source_capture_path=source,
        n_users=n_users, pattern=pattern, rounds=rounds,  # type: ignore[arg-type]
        bearer_prefix=bearer_prefix,
    )


def _build_presets(source: Path, bearer_prefix: str = "replay") -> dict[str, list[ReplayScenario]]:
    _s = lambda slug, **kw: _scen(slug, source=source, bearer_prefix=bearer_prefix, **kw)
    return {
        "phase-3.0-replay-sessions": [
            # Low N — both patterns should hit similarly (little contention
            # on session slot since few switches happen).
            _s("replay-N10-seq", n_users=10, pattern="sequential"),
            _s("replay-N10-ilv", n_users=10, pattern="interleaved"),
            # Mid N — thrash signal should appear in interleaved.
            _s("replay-N50-seq", n_users=50, pattern="sequential"),
            _s("replay-N50-ilv", n_users=50, pattern="interleaved"),
            # High N — pure thrash in interleaved; sequential still retains
            # per-user warm turns. Delta here is the session-cache-is-
            # single-slot evidence quantified.
            _s("replay-N200-seq", n_users=200, pattern="sequential"),
            _s("replay-N200-ilv", n_users=200, pattern="interleaved"),
        ],
        "phase-3.0-replay-smoke": [
            _s("replay-smoke-N2-seq", n_users=2, pattern="sequential"),
        ],
    }


def _now_iso_compact() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _build_run_dir(root: Path, backend_tag: str, preset_tag: str) -> Path:
    d = root / "runs" / f"{_now_iso_compact()}_{backend_tag}_{preset_tag}"
    (d / "scenarios").mkdir(parents=True, exist_ok=True)
    return d


def _estimate_cost_usd(
    scenarios: list[ReplayScenario],
    *,
    hourly_usd: float,
    avg_request_s: float,
) -> float:
    """Replay cost model is simpler than synthetic: per-request wall-clock
    is roughly constant (~2s from earlier data), independent of token
    length because the conversation is fixed. So budget = sum of
    (n_users * n_turns * rounds) * avg_request_s."""
    total_requests = 0
    for s in scenarios:
        n_turns = len(load_conversation(s.source_capture_path))
        total_requests += s.n_users * n_turns * s.rounds
    seconds = total_requests * avg_request_s
    return (seconds / 3600.0) * hourly_usd


def run_sweep(
    scenarios: list[ReplayScenario],
    *,
    base_url: str,
    run_dir: Path,
    kv_capacity_tokens: int | None,
) -> list[dict]:
    root = run_dir / "scenarios"
    out: list[dict] = []
    for s in scenarios:
        scen_dir = root / s.slug
        scen_dir.mkdir(parents=True, exist_ok=True)
        cap = scen_dir / "capture.jsonl"
        print(f"=== {s.slug}  (N={s.n_users}, pattern={s.pattern}, "
              f"rounds={s.rounds}) ===", flush=True)
        t0 = time.monotonic()
        summary = run_replay_scenario(s, base_url=base_url, capture_file=cap)
        summary["wall_clock_s_actual"] = round(time.monotonic() - t0, 3)
        (scen_dir / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")

        _, report = analyze_scenario(scen_dir, kv_capacity_tokens=kv_capacity_tokens)
        (scen_dir / "pressure-report.md").write_text(report)

        print(f"    ok={summary['n_successes']}/{summary['n_requests']}  "
              f"wall={summary['wall_clock_s_actual']:.1f}s", flush=True)
        out.append(summary)
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bench.drive_replay_pressure")
    p.add_argument("--base-url", required=True)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="capture.jsonl to replay. Default: pie-hermes "
                        "deep-search-read (3-turn, realistic tool use).")
    p.add_argument("--preset", default="phase-3.0-replay-sessions",
                   help="Preset name.")
    p.add_argument("--captures-root", type=Path,
                   default=Path(os.environ.get("CAPTURES_ROOT", "captures")))
    p.add_argument("--backend-tag",
                   default=os.environ.get("BACKEND_TAG", "unknown-backend"))
    p.add_argument("--preset-tag", default=None)
    p.add_argument("--max-cost-usd", type=float, default=None)
    p.add_argument("--hourly-usd", type=float, default=3.0)
    p.add_argument("--avg-request-s", type=float, default=2.0,
                   help="Per-request wall-clock estimate for cost preflight.")
    p.add_argument("--kv-capacity-tokens", type=int, default=None)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--bearer-prefix", default="replay",
                   help="Isolation prefix for `Authorization: Bearer <prefix>-uNNNN`. "
                        "Change between sweeps on the same pod to avoid session-cache "
                        "state from prior runs colliding on ExportNameExists errors "
                        "(session exports accumulate per turn and only the latest is "
                        "released on eviction — see session_cache.rs:160).")
    args = p.parse_args(argv)

    presets = _build_presets(args.source, bearer_prefix=args.bearer_prefix)
    if args.preset not in presets:
        print(f"[abort] unknown preset {args.preset!r} "
              f"(known: {sorted(presets)})", file=sys.stderr)
        return 2
    scenarios = presets[args.preset]
    preset_tag = args.preset_tag or args.preset

    try:
        est = _estimate_cost_usd(
            scenarios, hourly_usd=args.hourly_usd,
            avg_request_s=args.avg_request_s,
        )
    except FileNotFoundError as e:
        print(f"[abort] {e}", file=sys.stderr)
        return 2

    print(f"[preflight] scenarios={len(scenarios)}  est_cost_usd=${est:.2f}  "
          f"(assumes ${args.hourly_usd}/hr, {args.avg_request_s}s/request)",
          flush=True)
    if args.max_cost_usd is not None and est > args.max_cost_usd:
        print(f"[abort] est ${est:.2f} > --max-cost-usd ${args.max_cost_usd:.2f}",
              file=sys.stderr)
        return 2
    if not args.yes:
        try:
            r = input("[confirm] fire replay against live pod? [y/N] ").strip().lower()
        except EOFError:
            r = ""
        if r not in ("y", "yes"):
            print("[abort] user did not confirm", file=sys.stderr)
            return 3

    run_dir = _build_run_dir(args.captures_root, args.backend_tag, preset_tag)
    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "run_id": run_dir.name,
        "kind": "replay-pressure-sweep",
        "preset": args.preset,
        "source": str(args.source),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "base_url": args.base_url,
        "git": {
            "sha": _git("rev-parse", "HEAD", cwd=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
            "dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        },
        "cost_estimate": {
            "usd": round(est, 4),
            "hourly_usd": args.hourly_usd,
            "avg_request_s": args.avg_request_s,
        },
        "scenarios": [
            {
                "slug": s.slug, "n_users": s.n_users, "pattern": s.pattern,
                "rounds": s.rounds, "seed": s.seed,
                "source": str(s.source_capture_path),
            }
            for s in scenarios
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summaries = run_sweep(
        scenarios, base_url=args.base_url, run_dir=run_dir,
        kv_capacity_tokens=args.kv_capacity_tokens,
    )

    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["summaries"] = summaries
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[done] run_dir={run_dir}", flush=True)
    return 0 if all(s.get("n_errors", 0) == 0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
