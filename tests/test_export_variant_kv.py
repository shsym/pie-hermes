"""scripts/export-variant-kv.py issues one POST per variant to the inferlet's
/pie/variant/export route. Tests use a mock HTTP server so they don't need
a live pod."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "export-variant-kv.py"


class _Recorder(BaseHTTPRequestHandler):
    """Record every POST body into the server instance's ``received`` list."""
    server_version = "TestRecorder/0"

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        body = json.loads(raw)
        self.server.received.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "body": body,
        })
        resp = {"handle": f"hermes-variant-{body.get('variant_name', '')}",
                "token_count": len(body.get("text", "")) // 4}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, fmt, *args):  # suppress noisy logs during tests
        pass


@pytest.fixture()
def mock_inferlet():
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    server.received = []
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_script_posts_three_variants(mock_inferlet):
    server, base = mock_inferlet
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-url", base],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, f"script exit {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    names = sorted(rec["body"]["variant_name"] for rec in server.received)
    assert names == ["codex", "google", "none"], names

    # Every POST must target the variant export route.
    for rec in server.received:
        assert rec["path"].endswith("/pie/variant/export"), rec["path"]


def test_script_uses_hermes_agent_constants(mock_inferlet):
    """The text payload must match the live submodule constants so the
    exported KV prefills to identical tokens that the optimizer later strips."""
    server, base = mock_inferlet
    subprocess.run(
        [sys.executable, str(SCRIPT), "--base-url", base],
        check=True, capture_output=True, cwd=REPO,
    )
    sys.path.insert(0, str(REPO / "hermes-agent"))
    from agent.prompt_builder import (
        TOOL_USE_ENFORCEMENT_GUIDANCE,
        OPENAI_MODEL_EXECUTION_GUIDANCE,
        GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    )
    bodies = {rec["body"]["variant_name"]: rec["body"]["text"]
              for rec in server.received}
    assert bodies["codex"] == (
        TOOL_USE_ENFORCEMENT_GUIDANCE + "\n\n" + OPENAI_MODEL_EXECUTION_GUIDANCE
    )
    assert bodies["google"] == (
        TOOL_USE_ENFORCEMENT_GUIDANCE + "\n\n" + GOOGLE_MODEL_OPERATIONAL_GUIDANCE
    )
    assert bodies["none"] == ""
