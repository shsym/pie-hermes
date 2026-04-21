"""Unit tests for the SDK-level capture hook.

Uses a fake OpenAI-shaped client so the tests don't hit the network.
Covers the three paths that matter: no-op when env is unset, non-stream
capture, and stream capture (chunks recorded as they are yielded)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import capture_hook


class _FakeCompletions:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


class _FakeChunk:
    def __init__(self, text):
        self.text = text

    def model_dump(self):
        return {"text": self.text}


class _FakeResponse:
    def model_dump(self):
        return {"id": "resp-1", "choices": [{"finish_reason": "stop"}]}


def _reset_module():
    capture_hook._INSTALLED.clear()


def test_noop_without_env(monkeypatch, tmp_path):
    """Without either env var set, install is a no-op: calling .create
    writes nothing to disk and leaves the instance __dict__ untouched
    (the bound method comes from the class, not an instance attribute)."""
    _reset_module()
    monkeypatch.delenv(capture_hook._ENV_DIR, raising=False)
    monkeypatch.delenv(capture_hook._ENV_FILE, raising=False)
    completions = _FakeCompletions(_FakeResponse())
    capture_hook.install_capture(_FakeClient(completions))
    assert "create" not in completions.__dict__
    # And just to be paranoid: exercising it produces no capture file,
    # even in an unrelated tmp dir.
    completions.create(model="m", messages=[])
    assert not list(tmp_path.glob("session-*.jsonl"))


def test_explicit_file_path_via_env(monkeypatch, tmp_path):
    """HERMES_CAPTURE_FILE writes to that exact file, even inside a
    subdirectory that doesn't exist yet (parents are created)."""
    _reset_module()
    monkeypatch.delenv(capture_hook._ENV_DIR, raising=False)
    target = tmp_path / "runs" / "r1" / "scenarios" / "s1" / "capture.jsonl"
    monkeypatch.setenv(capture_hook._ENV_FILE, str(target))

    completions = _FakeCompletions(_FakeResponse())
    client = _FakeClient(completions)
    capture_hook.install_capture(client)
    client.chat.completions.create(model="m", messages=[])

    assert target.exists()
    rows = [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["kwargs"]["model"] == "m"


def test_file_env_takes_precedence_over_dir(monkeypatch, tmp_path):
    """When both HERMES_CAPTURE_FILE and HERMES_CAPTURE_DIR are set,
    FILE wins: DIR-style session-*.jsonl files must NOT be created."""
    _reset_module()
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv(capture_hook._ENV_FILE, str(target))
    monkeypatch.setenv(capture_hook._ENV_DIR, str(tmp_path / "should_not_be_used"))

    completions = _FakeCompletions(_FakeResponse())
    client = _FakeClient(completions)
    capture_hook.install_capture(client)
    client.chat.completions.create(model="m", messages=[])

    assert target.exists()
    # No session-* file in the DIR, and the DIR itself should not have been created
    assert not (tmp_path / "should_not_be_used").exists() or \
           not list((tmp_path / "should_not_be_used").glob("session-*.jsonl"))


def test_nonstream_records_request_and_response(monkeypatch, tmp_path):
    _reset_module()
    monkeypatch.setenv(capture_hook._ENV, str(tmp_path))
    completions = _FakeCompletions(_FakeResponse())
    client = _FakeClient(completions)
    capture_hook.install_capture(client)

    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t"}}],
        tool_choice="auto",
    )

    rows = _read_jsonl(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["kwargs"]["model"] == "m"
    assert row["kwargs"]["tools"][0]["function"]["name"] == "t"
    assert row["response"]["id"] == "resp-1"
    assert "ts_start" in row and "ts_end" in row


def test_stream_records_chunks_as_they_pass(monkeypatch, tmp_path):
    _reset_module()
    monkeypatch.setenv(capture_hook._ENV, str(tmp_path))
    chunks = [_FakeChunk("a"), _FakeChunk("b"), _FakeChunk("c")]
    completions = _FakeCompletions(iter(chunks))
    client = _FakeClient(completions)
    capture_hook.install_capture(client)

    stream = client.chat.completions.create(model="m", messages=[], stream=True)
    seen = [c.text for c in stream]
    assert seen == ["a", "b", "c"]

    rows = _read_jsonl(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["stream"] is True
    assert [c["text"] for c in row["chunks"]] == ["a", "b", "c"]


def test_error_is_recorded_and_reraised(monkeypatch, tmp_path):
    _reset_module()
    monkeypatch.setenv(capture_hook._ENV, str(tmp_path))

    class _BoomCompletions(_FakeCompletions):
        def create(self, **kwargs):
            raise RuntimeError("boom")

    completions = _BoomCompletions(None)
    client = _FakeClient(completions)
    capture_hook.install_capture(client)

    with pytest.raises(RuntimeError, match="boom"):
        client.chat.completions.create(model="m", messages=[])

    rows = _read_jsonl(tmp_path)
    assert len(rows) == 1
    assert "RuntimeError: boom" in rows[0]["error"]


def test_install_is_idempotent(monkeypatch, tmp_path):
    _reset_module()
    monkeypatch.setenv(capture_hook._ENV, str(tmp_path))
    completions = _FakeCompletions(_FakeResponse())
    client = _FakeClient(completions)
    capture_hook.install_capture(client)
    wrapped = client.chat.completions.create
    capture_hook.install_capture(client)
    assert client.chat.completions.create is wrapped


def _read_jsonl(directory: Path) -> list[dict]:
    files = sorted(directory.glob("session-*.jsonl"))
    assert files, f"no capture files in {directory}"
    lines = files[-1].read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]
