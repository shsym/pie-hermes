"""Session-start sanity checks.

Several tests import from the `hermes-agent` and `pie-vllm` submodules
(e.g. `from agent.prompt_optimizer import ...`, or the Rust inferlet
that test_submodules.py validates the on-disk path of). When the repo
is cloned without `--recurse-submodules`, those submodule directories
exist but are empty, and tests fail deep inside unrelated assertions
with cryptic `ModuleNotFoundError: No module named 'agent'`.

Catch it at session start with an actionable message so the fix is
obvious (`git submodule update --init --recursive`) rather than buried
in 11 failing tests' tracebacks.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# (submodule_dir, marker_path_inside, reason)
REQUIRED_SUBMODULES = [
    (
        "hermes-agent",
        "agent/__init__.py",
        "tests import `from agent.prompt_optimizer import ...`",
    ),
    (
        "pie-vllm",
        "pie/sdk/rust/inferlet/Cargo.toml",
        "inferlets/hermes-openai-compat/Cargo.toml resolves `inferlet` via "
        "`../../pie-vllm/pie/sdk/rust/inferlet`",
    ),
]


def pytest_sessionstart(session):
    missing = []
    for sub, marker, reason in REQUIRED_SUBMODULES:
        if not (REPO_ROOT / sub / marker).is_file():
            missing.append((sub, reason))
    if missing:
        lines = ["Required submodule(s) not initialized:"]
        for sub, reason in missing:
            lines.append(f"  - {sub!r}: {reason}")
        lines.append("")
        lines.append("Run: git submodule update --init --recursive")
        pytest.exit("\n".join(lines), returncode=2)
