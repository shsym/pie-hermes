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

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    print(f"=== {slug} ({','.join(tags)}) ===", flush=True)
    with driver_log.open("w") as fh:
        fh.write(f"# scenario: {slug}\n# description: {description}\n# query: {query}\n# max_turns: {max_turns}\n---\n")
        fh.flush()
        rc = subprocess.call(
            ["hermes", "chat", "-q", query, "-Q", "--max-turns", str(max_turns)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
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
