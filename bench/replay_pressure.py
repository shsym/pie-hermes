"""Replay captured hermes-agent conversations against a live inferlet under
multi-user pressure.

Phase 3.0 — Task I' (follow-up). Where `bench.pressure_scenarios` fires
simplified synthetic prompts, this module replays REAL conversations that
hermes-agent previously executed, so the measured cache behavior reflects
the token shape production actually produces: 12k-char system prompt with
embedded tool schemas + context index + skills, multi-turn history
growing per turn with interleaved tool_call / tool_result messages.

Mechanism per request:
  - Load captured `/v1/chat/completions` `kwargs` verbatim from a prior
    scenario's capture.jsonl. Each captured row is a complete request body
    (messages, tools, max_tokens, temperature, stream).
  - Swap `Authorization: Bearer <...>` per replay user so the inferlet's
    `auto_session_id` derives a distinct session per user
    (`sha256(bearer + first_system_message[:500])`).
  - Preserve `stream=True` so `pie_cache` telemetry rides the final SSE
    chunk and the analyzer sees it.

Replay patterns:
  "sequential": fire user u's turns 0..T in order, then user u+1, etc.
     Each user's turn 1+ hits `session_kv` because their session just
     stored state on turn 0. Session cache is single-slot (see
     inferlets/hermes-openai-compat/src/session_cache.rs:108): the
     transition from user u to user u+1 evicts u's session, but u was
     done by then — clean. Serves as the hit-rate upper bound.
  "interleaved": fire turn 0 for all users, then turn 1 for all users,
     etc. Every switch from user i to user j evicts i's session. By the
     time user 1 arrives at turn 1, its session has been overwritten N-1
     times. Expected hit rate ≈ 0 %. This is what a multi-tenant pod
     actually sees when N agents share the stack.

Not covered here: variant-KV, warmup-populated block_cache, or
Phase 2.1 registered handles. Those need their own activation and are
orthogonal to the session-cache single-slot question.
"""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal


__all__ = [
    "ReplayScenario",
    "load_conversation",
    "run_replay_scenario",
]


# ---------------------------------------------------------------------------
# Scenario spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayScenario:
    """Parameters that determine a byte-identical replay.

    `source_capture_path` points at an existing
    `captures/runs/<RUN>/scenarios/<slug>/capture.jsonl` file whose rows
    ARE the conversation to replay. `scenario_slug` names the replay
    output directory and is advisory — determinism is driven by
    (source, n_users, pattern, rounds, seed).
    """

    slug: str
    source_capture_path: Path
    n_users: int
    pattern: Literal["sequential", "interleaved"]
    rounds: int = 1
    seed: int = 0xDEADBEEF
    # Each user's auto_session_id is `sha256(bearer + system[:500])`. To get
    # N distinct sessions with the same underlying conversation, we rotate
    # the bearer per user. This prefix keeps the bearer tokens recognizably
    # ours in pod logs and differentiates replay users from any real
    # traffic that happens to share the pod.
    bearer_prefix: str = "replay"


# ---------------------------------------------------------------------------
# Capture loading
# ---------------------------------------------------------------------------


def load_conversation(source: Path) -> list[dict]:
    """Load an ordered list of request-bodies from a capture.jsonl.

    Each entry is the `kwargs` dict that was POSTed to
    `/v1/chat/completions` on the original run. Rows without a usable
    `kwargs` (e.g. analyzer-only rows; malformed writes) are skipped so
    an interrupted source capture doesn't kill the replay.

    Order is preserved — critical for multi-turn replay where turn N+1's
    messages include turn N's assistant/tool output.
    """
    if not source.exists():
        raise FileNotFoundError(f"source capture not found: {source}")
    turns: list[dict] = []
    with source.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kw = row.get("kwargs")
            if not isinstance(kw, dict):
                continue
            if not kw.get("messages"):
                continue
            turns.append(kw)
    if not turns:
        raise ValueError(
            f"no usable turns in {source} — is this a capture.jsonl with "
            "kwargs records, or an analyzer-only file?"
        )
    return turns


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------


def _iter_sse_events(resp) -> Iterator[dict]:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _fire_one(
    base_url: str,
    body: dict,
    bearer: str,
    timeout_s: float = 120.0,
) -> dict:
    """POST one request, stream the response, return a capture row
    shaped like `bench.pressure_scenarios._post_chat_once` output so the
    existing analyzer works unchanged."""
    # Force stream=True so pie_cache rides the final chunk. The original
    # captured body may have been non-streaming; telemetry still emits
    # either way but streaming is what the analyzer's _iter_pie_cache_
    # entries currently checks first.
    body = dict(body)
    body["stream"] = True

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer}",
        },
    )
    t0 = time.monotonic()
    row: dict = {
        "kwargs": body,
        "t_request_start_monotonic_s": t0,
        "replay_bearer": bearer,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            chunks: list[dict] = []
            first_token_at: float | None = None
            for evt in _iter_sse_events(resp):
                chunks.append(evt)
                if first_token_at is None:
                    choices = evt.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("tool_calls"):
                            first_token_at = time.monotonic()
            row["chunks"] = chunks
            if first_token_at is not None:
                row["ttft_s"] = round(first_token_at - t0, 4)
        row["elapsed_s"] = round(time.monotonic() - t0, 4)
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["elapsed_s"] = round(time.monotonic() - t0, 4)
    return row


# ---------------------------------------------------------------------------
# Access-pattern schedule
# ---------------------------------------------------------------------------


def _schedule(
    n_users: int,
    n_turns: int,
    rounds: int,
    pattern: Literal["sequential", "interleaved"],
) -> list[tuple[int, int, int]]:
    """Return the firing order as a list of (user_idx, round_idx, turn_idx).

    sequential: for each user, fire all rounds * turns contiguously
                before moving to next user.
    interleaved: for each (round, turn), fire that (round, turn) for
                 all users before moving to the next (round, turn).
    """
    if pattern == "sequential":
        out = []
        for u in range(n_users):
            for r in range(rounds):
                for t in range(n_turns):
                    out.append((u, r, t))
        return out
    if pattern == "interleaved":
        out = []
        for r in range(rounds):
            for t in range(n_turns):
                for u in range(n_users):
                    out.append((u, r, t))
        return out
    raise ValueError(f"unknown pattern: {pattern!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_replay_scenario(
    scenario: ReplayScenario,
    *,
    base_url: str,
    capture_file: Path,
) -> dict:
    """Execute a replay scenario end-to-end and append rows to
    `capture_file`. Returns a summary dict for meta.json."""
    capture_file.parent.mkdir(parents=True, exist_ok=True)
    conversation = load_conversation(scenario.source_capture_path)
    n_turns = len(conversation)

    schedule = _schedule(
        scenario.n_users, n_turns, scenario.rounds, scenario.pattern
    )

    t_start = time.monotonic()
    ok_count = 0
    err_count = 0
    with capture_file.open("a") as fh:
        for iter_idx, (u, r, t) in enumerate(schedule):
            # Deep-copy each turn before mutating (stream=True in _fire_one
            # would otherwise clobber the shared source body).
            body = copy.deepcopy(conversation[t])
            bearer = f"{scenario.bearer_prefix}-u{u:04d}"
            row = _fire_one(base_url, body, bearer)
            row["replay_user_idx"] = u
            row["replay_round"] = r
            row["replay_turn"] = t
            row["replay_iter"] = iter_idx
            # pressure_* aliases so the existing pressure analyzer
            # (bench.analyze_pressure) aggregates replay rows by
            # user / round without a separate analyzer.
            row["pressure_user_idx"] = u
            row["pressure_round"] = r
            row["pressure_iter_idx"] = iter_idx
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row.get("ok"):
                ok_count += 1
            else:
                err_count += 1

    wall_s = time.monotonic() - t_start
    return {
        "scenario_slug": scenario.slug,
        "kind": "replay",
        "source_capture": str(scenario.source_capture_path),
        "n_users": scenario.n_users,
        "n_turns_per_conversation": n_turns,
        "pattern": scenario.pattern,
        "rounds": scenario.rounds,
        "seed": scenario.seed,
        "arm": "baseline",  # keep analyzer compatibility
        "register_ok": False,
        "n_requests": len(schedule),
        "n_successes": ok_count,
        "n_errors": err_count,
        "wall_clock_s": round(wall_s, 3),
    }
