"""SDK-level capture of /v1/chat/completions traffic to JSONL.

Enabled by setting ``HERMES_CAPTURE_DIR``.  When set, every
``chat.completions.create`` call made through a client passed to
``install_capture`` is recorded as one JSONL row containing the request
kwargs and either the full non-stream response or the list of streamed
chunks.  When unset, ``install_capture`` is a no-op.

Captures live alongside Hermes so we can measure real prompt/token
numbers — system prompt size, tool-schema bloat, static-vs-dynamic
share per turn — without running a separate HTTP proxy.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_DIR = "HERMES_CAPTURE_DIR"
_ENV_FILE = "HERMES_CAPTURE_FILE"
_ENV = _ENV_DIR  # preserved for backward-compat with tests that monkeypatch _ENV
_SESSION_ID = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
_INSTALLED: set[int] = set()


def _jsonl_path() -> Path | None:
    """Resolve the JSONL target file.

    Precedence:
      1. HERMES_CAPTURE_FILE — absolute/relative path to a single file; used
         when a driver wants one file per scenario with a stable name
         (captures/runs/<run>/scenarios/<slug>/capture.jsonl).
      2. HERMES_CAPTURE_DIR — directory; file named session-<id>.jsonl
         (useful for ad-hoc single-process captures).

    Returns None when neither is set, which disables capture."""
    file_raw = os.environ.get(_ENV_FILE, "").strip()
    if file_raw:
        path = Path(file_raw).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    dir_raw = os.environ.get(_ENV_DIR, "").strip()
    if dir_raw:
        path = Path(dir_raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path / f"session-{_SESSION_ID}.jsonl"

    return None


def _append(record: dict) -> None:
    path = _jsonl_path()
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_fallback))
            fh.write("\n")
    except Exception as exc:
        logger.warning("capture_hook: failed to append record: %s", exc)


def _json_fallback(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return repr(obj)


def _scrub_kwargs(kwargs: dict) -> dict:
    """Strip fields we don't want in captures, keep pie-hermes bookkeeping.

    extra_headers can contain auth tokens, so we drop it by default — but
    keep a ``hermes_headers`` summary of any ``X-Hermes-*`` keys (plus the
    base64 length of ephemeral payloads). That's the minimum needed to
    verify Phase-1 optimizers actually emitted headers on the wire without
    leaking credentials.
    """
    out = {}
    for k, v in kwargs.items():
        if k == "extra_headers":
            if isinstance(v, dict):
                summary = {
                    name: (len(val) if name.lower() == "x-hermes-ephemeral" and isinstance(val, str) else val)
                    for name, val in v.items()
                    if isinstance(name, str) and name.lower().startswith("x-hermes-")
                }
                if summary:
                    out["hermes_headers"] = summary
            continue
        out[k] = v
    return out


class _StreamProxy:
    """Wrap an OpenAI ``Stream`` so every chunk is teed into ``record``
    before being forwarded to the caller. The original stream is not
    consumed twice — we log each chunk as it passes."""

    def __init__(self, inner, record: dict):
        self._inner = inner
        self._record = record
        self._chunks: list[dict] = []
        self._record["stream"] = True
        self._record["chunks"] = self._chunks

    def __iter__(self):
        try:
            for chunk in self._inner:
                try:
                    self._chunks.append(chunk.model_dump())
                except Exception:
                    self._chunks.append(_json_fallback(chunk))
                yield chunk
        finally:
            self._record["ts_end"] = time.time()
            _append(self._record)

    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        if hasattr(self._inner, "__exit__"):
            return self._inner.__exit__(*exc)
        return False

    def close(self):
        if hasattr(self._inner, "close"):
            self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install_capture(client: Any) -> None:
    """Monkey-patch ``client.chat.completions.create`` to tee every call to
    a JSONL file.

    Target file is picked by ``_jsonl_path``:
      1. ``HERMES_CAPTURE_FILE`` (explicit single file — preferred for
         structured per-scenario capture directories)
      2. ``HERMES_CAPTURE_DIR`` (directory; auto-picks
         ``session-<id>.jsonl``)

    Safe to call multiple times on the same client (second call is a
    no-op).  No-op when neither env var is set."""
    if _jsonl_path() is None:
        return
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None:
        return
    if id(completions) in _INSTALLED:
        return

    original = completions.create

    def _recording_create(*args, **kwargs):
        record = {
            "session_id": _SESSION_ID,
            "ts_start": time.time(),
            "kwargs": _scrub_kwargs(kwargs),
        }
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            record["ts_end"] = time.time()
            record["error"] = f"{type(exc).__name__}: {exc}"
            _append(record)
            raise

        if kwargs.get("stream"):
            return _StreamProxy(result, record)

        record["ts_end"] = time.time()
        try:
            record["response"] = result.model_dump()
        except Exception:
            record["response"] = _json_fallback(result)
        _append(record)
        return result

    completions.create = _recording_create  # type: ignore[method-assign]
    _INSTALLED.add(id(completions))
    path = _jsonl_path()
    if path is not None:
        logger.info("capture_hook: recording chat.completions → %s", path)
