"""Outer-repo install hook: wrap ``openai.OpenAI.__init__`` so every
client auto-registers the capture tee.

Previously Hermes itself was patched with a 5-line call to
``install_capture(client)`` right after each ``OpenAI(...)`` construction
(see ``agent/capture_hook.py`` in the pie-branch of the fork). That
patch is no longer needed — this module achieves the same effect from
outside the submodule by monkey-patching the SDK class itself.

Activation is conditional on ``HERMES_CAPTURE_FILE`` or
``HERMES_CAPTURE_DIR`` being set so the hook is a no-op in normal use."""

from __future__ import annotations

import logging
import os

from . import capture_hook

logger = logging.getLogger(__name__)

_INSTALLED = False


def _capture_enabled() -> bool:
    return bool(
        os.environ.get("HERMES_CAPTURE_FILE") or os.environ.get("HERMES_CAPTURE_DIR")
    )


def install() -> None:
    """Patch ``openai.OpenAI.__init__`` to tee every client's
    ``chat.completions.create`` calls. Idempotent; no-op if capture env
    is unset or ``openai`` can't be imported."""
    global _INSTALLED
    if _INSTALLED or not _capture_enabled():
        return
    try:
        import openai
    except ImportError as exc:
        logger.debug("bench.install: openai SDK not importable: %s", exc)
        return

    orig_init = openai.OpenAI.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            capture_hook.install_capture(self)
        except Exception as exc:
            logger.warning("bench.install: capture_hook failed on new client: %s", exc)

    openai.OpenAI.__init__ = _patched_init  # type: ignore[method-assign]
    _INSTALLED = True
    logger.info("bench.install: openai.OpenAI patched for capture")
