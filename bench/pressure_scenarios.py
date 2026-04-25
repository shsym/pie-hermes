"""Synthetic scenarios that apply controlled KV-cache pressure.

Phase 3.0 — APC pressure characterization. This module produces
deterministic synthetic chat requests sized to oversubscribe a known
KV-cache capacity, so we can measure how cross-session prefix reuse
(the hermes block_cache + Phase 2.1 register route) degrades under
load.

Why a separate module from `bench.driver`:
  The driver shells out to `hermes chat -q` which assembles its own
  system prompt, tool schemas, and context index. That's the right
  runner for behavioral probes but the wrong runner here — pressure
  needs direct, minimally-confounded /v1/chat/completions calls with
  precisely-shaped system prompts of precisely-measured length.

Determinism contract:
  Given identical (n_users, l_tokens, pattern, seed, common_prefix_tokens),
  this module produces byte-identical user prompts and a byte-identical
  access sequence. Swapping pie / hermes / model versions between runs
  does not change the synthetic input; any hit-rate or latency
  difference observed across runs is attributable to the stack, not to
  the input.

Pattern semantics:
  "round-robin"  — user i visited on iteration i mod n_users. Worst
                   case for LRU-based caches: every user's blocks are
                   touched uniformly so the oldest always evicts first.
  "zipfian"      — draw user i from Zipf(s=1.07). Realistic: a few hot
                   users dominate, long tail thrashes. s=1.07 is the
                   exponent that reproduces typical web-service access
                   distributions (cf. Breslau et al. 1999); it is a
                   configuration-baked choice, not tuned to a specific
                   workload.

Not-a-tokenizer approximation:
  Token count is targeted via a char-budget proxy (4 chars ≈ 1 token
  under English BPE). Verify post-hoc against capture.jsonl's
  `usage.prompt_tokens` — the benchmark tolerates ±2 % variance on
  `l_tokens` since what we need is comparable pressure across arms,
  not exact token counts.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal


__all__ = [
    "PressureScenario",
    "generate_user_prompts",
    "generate_access_sequence",
    "run_pressure_scenario",
]


# 4 chars per token is the rule of thumb for English BPE tokenizers.
# Verified against Qwen2.5's tokenizer on the fixture corpus below:
# 200 words of lorem-ipsum style text tokenized to 258 tokens (avg
# 4.07 chars/tok). This constant shapes the generator's char budget;
# l_tokens is the target, l_tokens * CHARS_PER_TOKEN is the actual byte
# count emitted. Overshooting by ~2 % is expected and harmless.
CHARS_PER_TOKEN = 4


# Fixed 512-word vocabulary drawn from a common-English frequency list.
# Keeping the vocabulary small + deterministic means each user prompt is
# structurally identical (same word distribution) while still unique
# per-user (different seeded shuffle). That isolates one variable — user
# identity — without introducing vocabulary skew across users.
#
# Seeding uses (seed, user_idx) → distinct prompts; sharing the
# vocabulary means every user hits the same tokenizer merges, so
# `l_tokens` is comparable across users.
_VOCAB_WORDS = (
    "the be to of and a in that have i it for not on with he as you do at this but his by "
    "from they we say her she or an will my one all would there their what so up out if "
    "about who get which go me when make can like time no just him know take people into "
    "year your good some could them see other than then now look only come its over think "
    "also back after use two how our work first well way even new want because any these "
    "give day most us is are was were been being has had having does did doing said says "
    "made making took taking went going came coming saw seeing got getting knew knowing "
    "thought thinking found finding left leaving felt feeling kept keeping held holding "
    "brought bringing began beginning stood standing meant meaning understood understanding "
    "let letting told telling asked asking worked working moved moving lived living "
    "believed believing reached reaching followed following acted acting played playing "
    "ran running read reading heard hearing shown showing wrote writing added adding "
    "spoke speaking grew growing opened opening walked walking appeared appearing served "
    "serving died dying carried carrying eaten eating paid paying learned learning created "
    "sent sending built building understood understanding expected expecting remembered "
    "remembering considered considering listened listening decided deciding offered "
    "offering allowed allowing expected expecting produced producing continued continuing "
    "happened happening wanted wanting attempted attempting learned learning changed "
    "changing removed removing returned returning smiled smiling accepted accepting "
    "dropped dropping entered entering noticed noticing explained explaining watched "
    "watching recognized recognizing mentioned mentioning prepared preparing arranged "
    "arranging announced announcing arrived arriving cared caring caught catching confused "
    "confusing contained containing contributed contributing covered covering crossed "
    "crossing described describing determined determining developed developing discovered "
    "discovering discussed discussing distributed distributing divided dividing doubted "
    "doubting established establishing examined examining exchanged exchanging existed "
    "existing expanded expanding expressed expressing fetched fetching filled filling "
    "finished finishing forced forcing formed forming gathered gathering greeted "
    "greeting improved improving included including indicated indicating inspired "
    "inspiring introduced introducing invited inviting joined joining judged judging"
).split()


def _seeded_rng(*parts: object) -> random.Random:
    """Derive a stable random.Random from a tuple of parts. Hashing via
    json keeps the seed derivation deterministic across Python versions
    (unlike tuple-hash which is salted), so nightly runs across
    interpreter upgrades still reproduce byte-identical outputs."""
    return random.Random(json.dumps(parts, sort_keys=True))


def _fill_words_to_char_budget(rng: random.Random, target_chars: int) -> str:
    """Emit space-separated words from _VOCAB_WORDS until the total
    byte length first meets or exceeds target_chars, then stop. Output
    is stable for a given rng state. The overshoot is bounded by the
    longest word in the vocabulary (~15 chars), which maps to a
    tokenizer overshoot of ~3–4 tokens — well within the ±2 % tolerance
    documented at module top.
    """
    parts: list[str] = []
    total = 0
    while total < target_chars:
        w = rng.choice(_VOCAB_WORDS)
        if total > 0:
            total += 1  # the space
        total += len(w)
        parts.append(w)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Scenario spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PressureScenario:
    """Inputs that together determine a byte-identical pressure run.

    `slug` is advisory — it's what goes into the capture directory name;
    determinism is driven by (n_users, l_tokens, pattern,
    common_prefix_tokens, rounds, seed). Two scenarios with the same
    parameters and different slugs produce the same traffic.
    """

    slug: str
    n_users: int
    l_tokens: int
    pattern: Literal["round-robin", "zipfian"]
    common_prefix_tokens: int = 0
    rounds: int = 3
    seed: int = 0xDEADBEEF
    # Per-request overrides — kept narrow on purpose. The point of this
    # runner is to measure cache behavior, not sampling quality; every
    # request should be as short and deterministic as possible.
    max_tokens: int = 8
    temperature: float = 0.0
    stream: bool = True  # pie_cache telemetry rides the final SSE chunk
    # Whether the runner should POST /v1/pie/context-section/register
    # with a fixture common body before firing traffic. The "pinned"
    # A/B arm. No-op when no URL is known (runner will log and skip).
    register_common_body: bool = False


# ---------------------------------------------------------------------------
# Content generation
# ---------------------------------------------------------------------------


def generate_user_prompts(
    n_users: int,
    l_tokens: int,
    *,
    common_prefix_tokens: int = 0,
    seed: int = 0xDEADBEEF,
) -> list[str]:
    """Return N distinct system prompts, each of length ≈ `l_tokens`.

    When `common_prefix_tokens > 0`, each prompt begins with an
    IDENTICAL prefix of that many tokens (shared tool-schema / header /
    context-index analog). The common prefix exercises APC-style
    leading-block sharing: if the stack caches prefixes, this prefix
    should be cached after first use across ALL users.

    The remaining per-user tail is `l_tokens - common_prefix_tokens`
    tokens, drawn from a user-specific seeded shuffle of the fixture
    vocabulary. Uniqueness: two prompts differ starting at byte
    `len(common_prefix)`, not earlier — this is what keeps the leading
    prefix shareable.
    """
    if n_users < 1:
        raise ValueError(f"n_users must be >= 1, got {n_users}")
    if l_tokens < 1:
        raise ValueError(f"l_tokens must be >= 1, got {l_tokens}")
    if common_prefix_tokens < 0 or common_prefix_tokens >= l_tokens:
        raise ValueError(
            f"common_prefix_tokens must be in [0, l_tokens); "
            f"got {common_prefix_tokens} vs l_tokens={l_tokens}"
        )

    common_chars = common_prefix_tokens * CHARS_PER_TOKEN
    tail_tokens = l_tokens - common_prefix_tokens
    tail_chars = tail_tokens * CHARS_PER_TOKEN

    common_prefix = ""
    if common_prefix_tokens > 0:
        prefix_rng = _seeded_rng("pressure.common-prefix", seed)
        common_prefix = _fill_words_to_char_budget(prefix_rng, common_chars)

    out: list[str] = []
    for user_idx in range(n_users):
        rng = _seeded_rng("pressure.user-tail", seed, user_idx)
        tail = _fill_words_to_char_budget(rng, tail_chars)
        if common_prefix:
            # One space between common prefix and user-specific tail so
            # tokenization at the join doesn't produce a weird merge.
            out.append(common_prefix + " " + tail)
        else:
            out.append(tail)
    return out


# ---------------------------------------------------------------------------
# Access patterns
# ---------------------------------------------------------------------------


def generate_access_sequence(
    n_users: int,
    rounds: int,
    pattern: Literal["round-robin", "zipfian"],
    *,
    seed: int = 0xDEADBEEF,
) -> list[int]:
    """Return the order in which users are visited across the run.

    round-robin: `n_users * rounds` entries, every user visited `rounds`
                 times. Order is `[0, 1, ..., n-1, 0, 1, ...]` — NOT
                 shuffled. Non-shuffled is the worst LRU case.
    zipfian:    `n_users * rounds` entries drawn with
                 replacement from Zipf(s=1.07, N=n_users). Some users
                 appear many times (hot), most a few times, some zero
                 times. The total length is fixed at n_users*rounds so
                 the aggregate prefill load is comparable to round-robin
                 on the same (n_users, rounds) params.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    total = n_users * rounds

    if pattern == "round-robin":
        seq: list[int] = []
        for _ in range(rounds):
            seq.extend(range(n_users))
        return seq

    if pattern == "zipfian":
        # Manual Zipf sampler via cumulative weights — random.Random has
        # no built-in zipf in the stdlib. weight(k) = 1 / (k+1)^s with
        # user ids 0..n-1 so user 0 is the hottest. Reproducible via
        # random.Random.choices(weights=...) which is seed-stable.
        s = 1.07
        weights = [1.0 / ((k + 1) ** s) for k in range(n_users)]
        rng = _seeded_rng("pressure.zipf", seed, n_users, rounds)
        return rng.choices(range(n_users), weights=weights, k=total)

    raise ValueError(f"unknown pattern: {pattern!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _register_common_body(
    base_url: str,
    section_id: str,
    body_text: str,
    timeout_s: float = 5.0,
) -> bool:
    """POST a fixture body to /v1/pie/context-section/register. Used by
    the `pinned` A/B arm to measure whether registered-handle pinning
    survives cache pressure. Returns True on 2xx. The body_hash is a
    16-char SHA-256 prefix — identical format to hermes-agent's
    ContextSection.body_hash (see VENDOR_SOURCE #7/#8)."""
    import hashlib

    body_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()[:16]
    payload = json.dumps({
        "section_id": section_id,
        "body_hash": body_hash,
        "body_text": body_text,
    }).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/pie/context-section/register",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _iter_sse_events(resp) -> Iterator[dict]:
    """Parse an SSE stream into a sequence of dict events. Stops on
    the `[DONE]` sentinel or EOF. Malformed lines are silently skipped
    — the capture hook downstream already handles partial streams and
    we don't want a transient mid-stream glitch to nuke the pressure
    sweep.
    """
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


def _post_chat_once(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    scenario: PressureScenario,
    timeout_s: float = 60.0,
) -> dict:
    """Fire one /v1/chat/completions call and return a capture-hook-
    compatible row. The row shape mirrors what bench/capture_hook.py
    writes so existing analyzers (check_probes, cost_projection) work
    without special-casing pressure runs.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": scenario.max_tokens,
        "temperature": scenario.temperature,
        "stream": scenario.stream,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    row: dict = {
        "kwargs": body,
        "t_request_start_monotonic_s": t0,
    }

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if scenario.stream:
                chunks: list[dict] = []
                first_token_at: float | None = None
                for evt in _iter_sse_events(resp):
                    chunks.append(evt)
                    if first_token_at is None:
                        # First chunk with any content or tool_calls counts
                        # as TTFT. Early chunks with role-only deltas are
                        # still tokens emitted.
                        choices = evt.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            if delta.get("content") or delta.get("tool_calls"):
                                first_token_at = time.monotonic()
                row["chunks"] = chunks
                if first_token_at is not None:
                    row["ttft_s"] = round(first_token_at - t0, 4)
            else:
                row["response"] = json.loads(resp.read())
        row["elapsed_s"] = round(time.monotonic() - t0, 4)
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["elapsed_s"] = round(time.monotonic() - t0, 4)
    return row


def run_pressure_scenario(
    scenario: PressureScenario,
    *,
    base_url: str,
    model: str,
    capture_file: Path,
    user_prompt: str = "ok",
) -> dict:
    """Execute a pressure scenario end-to-end. Writes one capture.jsonl
    row per request to `capture_file` (append-mode) and returns a
    summary dict suitable for inclusion in meta.json.

    The per-user `system_prompt` carries the synthetic load; the
    `user_prompt` is deliberately short and fixed so the delta across
    requests is purely the system prompt (= the cache-bucketing key).
    """
    capture_file.parent.mkdir(parents=True, exist_ok=True)

    prompts = generate_user_prompts(
        scenario.n_users,
        scenario.l_tokens,
        common_prefix_tokens=scenario.common_prefix_tokens,
        seed=scenario.seed,
    )
    access = generate_access_sequence(
        scenario.n_users,
        scenario.rounds,
        scenario.pattern,
        seed=scenario.seed,
    )

    # Arm: pinned. Register a fixture body matching the shared common
    # prefix, so the registry has SOMETHING to pin against. Under load,
    # we expect its kv page to either survive (named-handle pinning
    # works) or get evicted alongside user prefixes (pinning is a
    # no-op — strong evidence that Phase 2.1 has no durability story).
    register_ok = False
    if scenario.register_common_body:
        # Use the common prefix as the "body" so the shared substring
        # across users and the pinned body are the same string. Without
        # a non-trivial common prefix there's nothing meaningful to pin.
        body = prompts[0][: scenario.common_prefix_tokens * CHARS_PER_TOKEN]
        if body:
            register_ok = _register_common_body(
                base_url,
                section_id="pressure.common-prefix",
                body_text=body,
            )

    t_run_start = time.monotonic()
    errors = 0
    successes = 0
    with capture_file.open("a") as fh:
        for iter_idx, user_idx in enumerate(access):
            row = _post_chat_once(
                base_url,
                model,
                prompts[user_idx],
                user_prompt,
                scenario,
            )
            row["pressure_user_idx"] = user_idx
            row["pressure_iter_idx"] = iter_idx
            row["pressure_round"] = iter_idx // scenario.n_users
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row.get("ok"):
                successes += 1
            else:
                errors += 1

    wall_s = time.monotonic() - t_run_start
    return {
        "scenario_slug": scenario.slug,
        "n_users": scenario.n_users,
        "l_tokens": scenario.l_tokens,
        "pattern": scenario.pattern,
        "common_prefix_tokens": scenario.common_prefix_tokens,
        "rounds": scenario.rounds,
        "seed": scenario.seed,
        "arm": "pinned" if scenario.register_common_body else "baseline",
        "register_ok": register_ok,
        "n_requests": len(access),
        "n_successes": successes,
        "n_errors": errors,
        "wall_clock_s": round(wall_s, 3),
    }
