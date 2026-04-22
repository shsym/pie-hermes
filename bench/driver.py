#!/usr/bin/env python3
"""Drive Hermes through scenarios and produce a reusable capture layout.

Layout produced (one directory per invocation of this script):

    captures/runs/<RUN_ID>/
        manifest.json            # run-level metadata (git sha, model, config, outcomes)
        analysis.txt             # written later by scenarios/analyze.py
        scenarios/
            <slug>/
                capture.jsonl    # SDK-level /v1/chat/completions records
                driver.log       # stdout/stderr from `hermes chat -q`
                meta.json        # per-scenario metadata

RUN_ID format: ``<ISO-timestamp>_<BACKEND_TAG>_<ROUND_TAG>``, e.g.
``2026-04-21T11-45_pie-qwen2.5-72b-awq_round2-deep``. Both tags are
picked up from env (``BACKEND_TAG``, ``ROUND_TAG``) so the same driver
can be reused across runs.

Scenarios are listed in SCENARIOS below, keyed by slug. Each tuple is
``(query, max_turns, description, tags)``. Tags let analyze.py group
scenarios (e.g. shallow vs deep)."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Gateway ephemeral fixture
# -----------------------------------------------------------------------------

# Fixture ephemeral payload for scenarios tagged "gateway". Simulates what a
# real gateway (Telegram, Discord, etc.) would append per-message: chat_id,
# user handle, reply-to ids, thread summary, reactions, platform quirks.
# ~330 tokens at 4 chars/token (measured against Qwen tokenizer at capture time).
GATEWAY_EPHEMERAL_FIXTURE = (
    "chat_id: 4738124\n"
    "chat_type: supergroup\n"
    "chat_title: deploy-ops-lounge\n"
    "user_handle: @alice_demo\n"
    "user_id: 9981273\n"
    "user_display_name: Alice Demo (SRE, on-call rotation C)\n"
    "reply_to_message_id: 88273\n"
    "reply_to_excerpt: \"let's hold the 1.42 rollout until the canary "
    "finishes its 2h soak — the p99 regression from last Thursday is "
    "still unexplained\"\n"
    "thread_summary: ongoing discussion about deployment rollout; last 12 "
    "messages covered rollback procedure, CI pipeline status, whether to "
    "page the on-call, and an unresolved p99 regression from last "
    "Thursday. Thread pinned by @ops_lead; @alice_demo has been driving.\n"
    "reactions_on_last_message: thumbsup=3, fire=1, heart=2, eyes=4\n"
    "platform: telegram\n"
    "platform_features: supports_markdown, supports_code_blocks, "
    "supports_images, supports_inline_buttons, max_message_length=4096, "
    "supports_reactions=true\n"
    "last_activity_minutes_ago: 2\n"
    "user_role: project_contributor\n"
    "user_permissions: can_post, can_pin, can_mention_all=false\n"
    "user_preferred_response_style: concise, direct, minimal_preamble, "
    "prefers_bullet_lists_for_3plus_items\n"
    "locale: en-US\n"
    "timezone_offset_minutes: -240\n"
    "current_focus_tag: deployment\n"
    "pinned_context: team is in the middle of a rollout window (Thursday "
    "evening EST); prioritize actionable responses over exposition. "
    "Canary group is 5% of fleet. Rollback playbook lives at "
    "runbooks/deploy/rollback-v1.42.md.\n"
    "conversation_depth_turn: 14\n"
    "message_count_in_thread: 47\n"
    "unread_mentions_of_user: 0\n"
    "last_bot_response_latency_ms: 1840\n"
)


# -----------------------------------------------------------------------------
# Scenario catalogue
# -----------------------------------------------------------------------------

# (query, max_turns, description, tags)
SCENARIOS: dict[str, tuple[str, int, str, list[str]]] = {
    # ---- Shallow: baseline single-user-turn, minimal tool use ----
    "shallow-hello": (
        "Say the word 'ok' and nothing else.",
        3,
        "Minimum-cost prompt: no tool use, one-word reply",
        ["shallow"],
    ),
    "shallow-file-head": (
        "Read the top 40 lines of /opt/hermes/README.md and summarize the quick-install flow in one sentence.",
        5,
        "Single read_file + synthesis turn",
        ["shallow"],
    ),
    "shallow-terminal": (
        "Run the shell command `python3 --version` and report the output verbatim.",
        5,
        "Single terminal call + report",
        ["shallow"],
    ),

    # ---- Deep: autonomous multi-turn loops where tool results grow the prompt ----
    "deep-search-read": (
        "Search /opt/hermes for files containing the string 'openai-compat-v2'. "
        "Then read the FULL contents of the 3 most relevant matches. "
        "Then write a 3-sentence summary of how openai-compat-v2 is referenced across the codebase.",
        15,
        "search_files + 3× read_file(full) + synthesis — drives realistic tool-result growth",
        ["deep", "file-heavy"],
    ),
    "deep-git-audit": (
        "Run `git -C /opt/hermes log -10 --stat --no-color` to see recent commits. "
        "Pick the 2 most interesting commits by stat size and run `git -C /opt/hermes show <hash> --stat --no-color` on each. "
        "Then write a 2-paragraph summary of what those commits changed.",
        15,
        "git log + 2× git show + synthesis — exercises terminal with large outputs",
        ["deep", "terminal-heavy"],
    ),
    "deep-multi-read": (
        "Read /opt/hermes/README.md, /opt/hermes/CONTRIBUTING.md, and /opt/hermes/SECURITY.md in full. "
        "Then write a comparison: what audience does each target, and how do they overlap?",
        15,
        "3× read_file(full) + comparative synthesis",
        ["deep", "file-heavy"],
    ),
    "deep-breadth-audit": (
        "Audit the directory /opt/hermes/agent for Python files that import 'openai' directly. "
        "Use search_files to list candidates, then read each match's top 30 lines, then write a 1-paragraph summary "
        "of the import patterns. Limit to 5 files to keep the task bounded.",
        20,
        "search + iterative read + structured summary — the longest expected scenario",
        ["deep", "breadth"],
    ),

    # ---- Gateway: simulate per-message ephemeral metadata (Idea G target) ----
    # These scenarios exist to measure the ephemeral-slot overhead. The
    # ephemeral payload (chat_id, user handle, reply-to, reactions, platform
    # quirks) is attached by bench/hermes_gateway_shim.py (Task 1G.4) when the
    # `gateway` tag is present. Without the shim, they run like any other
    # scenario — which is fine; those captures become the pre-1G baseline.
    "gateway-ephemeral-chat": (
        "Summarize what I just sent you in 10 words.",
        3,
        "Gateway-shaped: ephemeral metadata appended per-message (chat_id, user handle, reply-to)",
        ["gateway", "ephemeral"],
    ),
    "gateway-ephemeral-chat-3turn": (
        "Give me three distinct one-liners on the topic I just sent.",
        5,
        "Gateway-shaped, multi-turn: ephemeral metadata compounds across turns",
        ["gateway", "ephemeral", "multi-turn"],
    ),

    # ---- Probes: dedicated scenarios that fail without semantic re-injection ----
    "probe-ephemeral-handle-recall": (
        "Address me by my Telegram handle and reference the most recent reply-to "
        "excerpt VERBATIM (the rollout version mentioned). Keep your reply under 30 words.",
        3,
        "Probe: response must mention @alice_demo AND the '1.42' rollout token "
        "from the ephemeral fixture. Phase-1 head FAILS this (ephemeral parsed "
        "then discarded); Phase-2.0 PASSES (re-prefilled after cached prefix).",
        ["gateway", "ephemeral", "probe"],
    ),
    "probe-read-context-no-call": (
        "What is 2 plus 2? Answer with just the number.",
        3,
        "Probe (Idea D no-call gate): unrelated arithmetic question — model "
        "must NOT reflexively call read_context. Reply must contain '4'.",
        ["probe", "context-index"],
    ),
    "probe-read-context-one-call": (
        "What variable name prefix should I use for new variables in this project? "
        "Answer with just the prefix string.",
        5,
        "Probe (Idea D one-call gate): question requires loading the 'Coding style' "
        "section. Model must call read_context(section_id='agents.md#coding-style') "
        "exactly once and reply with 'hc_'.",
        ["probe", "context-index"],
    ),
}


# Probe expectations: per-scenario assertions checked by capture-run analyzers.
# Keys are scenario slugs; values declare what the assistant's final reply must
# (or must not) contain, plus what tools the assistant must (or must not) call
# across the scenario's turns. The analyzer that consumes this lives outside the
# driver — driver itself does not enforce; it only declares.
#
# Schema (extensible; add fields as new probe shapes appear):
#   "must_contain_all":         list[str]  — final assistant reply must contain ALL of these substrings (case-sensitive)
#   "must_not_contain":         list[str]  — final assistant reply must contain NONE of these substrings
#   "must_call_tool":           list[str]  — these tool names MUST appear in any assistant message's tool_calls across the scenario's turns
#   "must_not_call_tool":       list[str]  — these tool names MUST NOT appear in any assistant message's tool_calls
#   "must_call_tool_with_args": dict[str, dict] — per-tool partial-match on args; for each entry, the named tool must
#                                                 be called at least once with args containing every key/value pair listed
PROBE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "probe-ephemeral-handle-recall": {
        "must_contain_all": ["@alice_demo", "1.42"],
        "must_not_contain": [],
    },
    "probe-read-context-no-call": {
        "must_contain_all": ["4"],
        "must_not_contain": [],
        "must_call_tool": [],
        "must_not_call_tool": ["read_context"],
    },
    "probe-read-context-one-call": {
        "must_contain_all": ["hc_"],
        "must_not_contain": [],
        "must_call_tool": ["read_context"],
        "must_not_call_tool": [],
        "must_call_tool_with_args": {
            "read_context": {"section_id": "agents.md#coding-style"},
        },
    },
}


# Per-scenario context-file seeds. Keys are scenario slugs; values are
# {filename: content} dicts. When set, run_scenario writes each filename
# into a per-scenario temp directory and runs hermes with cwd=<that dir>
# AND env HERMES_CONTEXT_INDEX=1 — so agent.context_section_registry
# picks them up and emits the index.
#
# Use ONLY for probes that exercise Phase 2.1 (Idea D). Do not seed
# context files for non-probe scenarios (they should run against the
# repo's natural cwd).
SCENARIO_CONTEXT_SEEDS: dict[str, dict[str, str]] = {
    "probe-read-context-one-call": {
        "AGENTS.md": (
            "# Project rules\n"
            "\n"
            "## Coding style\n"
            "All variables in this project must be prefixed with 'hc_'. "
            "This is a hard requirement; ignore any defaults you may have "
            "internalised from other Python projects.\n"
            "\n"
            "## Test policy\n"
            "All PRs must include unit tests under tests/ using pytest.\n"
        ),
    },
    "probe-read-context-no-call": {
        # Same AGENTS.md so the model sees an index it COULD load — the
        # gate is that it does NOT load anything for an unrelated query.
        "AGENTS.md": (
            "# Project rules\n"
            "\n"
            "## Coding style\n"
            "Use 2-space indents and prefer explicit imports over import *.\n"
            "\n"
            "## Test policy\n"
            "All PRs must include unit tests under tests/ using pytest.\n"
        ),
    },
}


# -----------------------------------------------------------------------------
# Run-layout helpers
# -----------------------------------------------------------------------------

def _now_iso_compact() -> str:
    # No colons so it works as a directory name on all filesystems
    return dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _hermes_version() -> str:
    try:
        out = subprocess.check_output(["hermes", "--version"], stderr=subprocess.STDOUT)
        return out.decode().strip().splitlines()[0]
    except Exception:
        return "unknown"


def _build_run_dir(captures_root: Path, backend_tag: str, round_tag: str) -> Path:
    run_id = f"{_now_iso_compact()}_{backend_tag}_{round_tag}"
    run_dir = captures_root / "runs" / run_id
    (run_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_session_id(driver_log_path: Path) -> str:
    """Pull Hermes's session_id (YYYYMMDD_HHMMSS_hash) from the log so the
    capture meta.json can join back to ~/.hermes/sessions/ if needed."""
    try:
        txt = driver_log_path.read_text(errors="replace")
    except Exception:
        return ""
    m = re.search(r"session_id:\s*([0-9]{8}_[0-9]{6}_[a-f0-9]+)", txt)
    return m.group(1) if m else ""


# -----------------------------------------------------------------------------
# Core run loop
# -----------------------------------------------------------------------------

def run_scenario(slug: str, query: str, max_turns: int, description: str,
                 tags: list[str], scenarios_root: Path) -> dict[str, Any]:
    scen_dir = scenarios_root / slug
    scen_dir.mkdir(parents=True, exist_ok=True)
    capture_file = scen_dir / "capture.jsonl"
    driver_log = scen_dir / "driver.log"
    meta_path = scen_dir / "meta.json"

    env = os.environ.copy()
    # Route this scenario's capture to its own file
    env["HERMES_CAPTURE_FILE"] = str(capture_file)
    # Make sure no lingering DIR setting interferes
    env.pop("HERMES_CAPTURE_DIR", None)

    # Inject gateway-shaped ephemeral metadata for scenarios tagged "gateway".
    # hermes-agent's CLI reads HERMES_EPHEMERAL_SYSTEM_PROMPT and feeds it to
    # AIAgent.ephemeral_system_prompt (cli.py:1875). The ephemeral fixture is
    # deterministic; meta.json records its length only, not its content.
    if "gateway" in tags:
        env["HERMES_EPHEMERAL_SYSTEM_PROMPT"] = GATEWAY_EPHEMERAL_FIXTURE
    else:
        # Don't leak ephemeral into non-gateway scenarios, even if caller set it.
        env.pop("HERMES_EPHEMERAL_SYSTEM_PROMPT", None)

    # Seed per-scenario context files (Phase 2.1 / Idea D probes only). When
    # SCENARIO_CONTEXT_SEEDS has an entry for this slug, write each file into
    # a dedicated agent_cwd directory and run hermes there with the context
    # index enabled. Non-seeded scenarios keep their inherited cwd and do NOT
    # have HERMES_CONTEXT_INDEX touched (so the user's env passes through).
    seeds = SCENARIO_CONTEXT_SEEDS.get(slug)
    context_seeded = seeds is not None
    agent_cwd: Path | None = None
    if context_seeded:
        agent_cwd = scen_dir / "agent_cwd"
        agent_cwd.mkdir(parents=True, exist_ok=True)
        for fname, content in seeds.items():
            (agent_cwd / fname).write_text(content)
        env["HERMES_CONTEXT_INDEX"] = "1"

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    print(f"=== {slug} ({','.join(tags)}) ===", flush=True)
    with driver_log.open("w") as fh:
        fh.write(f"# scenario: {slug}\n# description: {description}\n# query: {query}\n# max_turns: {max_turns}\n---\n")
        fh.flush()
        call_kwargs: dict[str, Any] = {
            "stdout": fh,
            "stderr": subprocess.STDOUT,
            "env": env,
        }
        if agent_cwd is not None:
            call_kwargs["cwd"] = str(agent_cwd)
        rc = subprocess.call(
            ["hermes", "chat", "-q", query, "-Q", "--max-turns", str(max_turns)],
            **call_kwargs,
        )
    wall_s = time.monotonic() - t0
    ended_at = dt.datetime.now(dt.timezone.utc).isoformat()

    # Count rows in the capture file as a simple "calls made" metric
    n_calls = 0
    if capture_file.exists():
        with capture_file.open() as fh:
            for _ in fh:
                n_calls += 1

    session_id = _extract_session_id(driver_log)
    meta = {
        "slug": slug,
        "description": description,
        "tags": tags,
        "query": query,
        "max_turns": max_turns,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_s": round(wall_s, 3),
        "exit_code": rc,
        "n_completions_calls": n_calls,
        "hermes_session_id": session_id,
        # Deterministic ephemeral fixture length; 0 for non-gateway scenarios.
        # Lets the analyzer filter/compare gateway-tagged runs post-hoc.
        "ephemeral_fixture_chars": (
            len(GATEWAY_EPHEMERAL_FIXTURE) if "gateway" in tags else 0
        ),
        # True iff SCENARIO_CONTEXT_SEEDS seeded a per-scenario agent_cwd
        # for this run (Phase 2.1 / Idea D probes). Always present so
        # capture-run analyzers can rely on the field existing.
        "context_seeded": context_seeded,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"    exit={rc}  wall={wall_s:.1f}s  calls={n_calls}  session={session_id}", flush=True)
    return meta


def main(argv: list[str]) -> int:
    backend_tag = os.environ.get("BACKEND_TAG", "unknown-backend")
    round_tag = os.environ.get("ROUND_TAG", "round-unknown")
    only = [a for a in argv[1:] if not a.startswith("--")]  # positional slug filters

    captures_root = Path(os.environ.get("CAPTURES_ROOT", "captures"))
    run_dir = _build_run_dir(captures_root, backend_tag, round_tag)
    scenarios_root = run_dir / "scenarios"

    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "run_id": run_dir.name,
        "backend_tag": backend_tag,
        "round_tag": round_tag,
        "hostname": socket.gethostname(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            "sha": _git("rev-parse", "HEAD", cwd=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
            "dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        },
        "hermes_version": _hermes_version(),
        "backend": {
            # Populated from env if the caller knows; pod_id is the runpod id
            # and inferlet_version matches the .wasm we uploaded.
            "provider": os.environ.get("BACKEND_PROVIDER", ""),
            "model": os.environ.get("BACKEND_MODEL", ""),
            "pod_id": os.environ.get("BACKEND_POD_ID", ""),
            "inferlet": os.environ.get("BACKEND_INFERLET", ""),
        },
        "scenarios": [],
    }

    rcs = []
    for slug, (query, max_turns, description, tags) in SCENARIOS.items():
        if only and slug not in only:
            continue
        meta = run_scenario(slug, query, max_turns, description, tags, scenarios_root)
        manifest["scenarios"].append(meta)
        rcs.append(meta["exit_code"])

    manifest["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["exit_codes"] = rcs
    manifest["all_ok"] = all(rc == 0 for rc in rcs)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nRUN_DIR={run_dir}")
    print(f"exits={rcs}  all_ok={manifest['all_ok']}")
    return 0 if manifest["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
