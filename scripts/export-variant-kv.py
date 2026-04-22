#!/usr/bin/env python3
"""Pre-export variant-guidance KV segments into a pie pod.

Usage:
  python3 scripts/export-variant-kv.py --base-url http://127.0.0.1:9090/v1

Posts one {variant_name, text} record per variant to
POST <base-url>/pie/variant/export. The inferlet (Task 1A.2) prefills
a context with text and exports its KV pages under the handle name
hermes-variant-<variant_name>.

Designed to be idempotent: re-running overwrites the same handles."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Make the hermes-agent submodule importable so we read the same constant
# strings the agent will inject at runtime — this way the exported KV
# prefills to bit-identical tokens that the VariantSwapOptimizer (Task 1A.4)
# later strips from the text prompt.
sys.path.insert(0, str(REPO / "hermes-agent"))

from agent.prompt_builder import (  # noqa: E402 — intentional post-sys.path.insert
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
)

VARIANTS: dict[str, str] = {
    "codex":  TOOL_USE_ENFORCEMENT_GUIDANCE + "\n\n" + OPENAI_MODEL_EXECUTION_GUIDANCE,
    "google": TOOL_USE_ENFORCEMENT_GUIDANCE + "\n\n" + GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    "none":   "",  # base variant: no guidance text, import is a no-op prepend
}


def export_one(base_url: str, name: str, text: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/pie/variant/export",
        method="POST",
        data=json.dumps({"variant_name": name, "text": text}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer variant-exporter",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True,
                    help="Inferlet base URL, e.g. http://127.0.0.1:9090/v1")
    args = ap.parse_args(argv)

    rc = 0
    for name, text in VARIANTS.items():
        try:
            resp = export_one(args.base_url, name, text)
            print(f"{name}: handle={resp.get('handle')} "
                  f"token_count={resp.get('token_count')}")
        except Exception as e:
            print(f"{name}: FAILED — {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
