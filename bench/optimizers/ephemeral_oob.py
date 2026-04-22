"""Phase-1 Idea G: route ephemeral metadata out-of-band via HTTP headers
instead of prompt text.

Two surfaces:
- EphemeralOOBOptimizer: the PromptOptimizer used by Composite. Currently
  a pass-through for prompt_parts (ephemeral is not in prompt_parts).
- EphemeralOOBRouter: the EphemeralRouter that drops the ephemeral text
  and emits X-Hermes-Ephemeral: <base64-utf8-json> header for the inferlet
  to consume."""
from __future__ import annotations

import base64


class EphemeralOOBOptimizer:
    """Pass-through for parts; the real work happens in EphemeralOOBRouter."""
    def optimize(self, parts):
        from agent.prompt_optimizer import PromptOptimization
        return PromptOptimization(parts=parts)


class EphemeralOOBRouter:
    """EphemeralRouter impl that drops the ephemeral text entirely and
    emits X-Hermes-Ephemeral with the base64-encoded original body.

    The inferlet-side consumer (Task 1G.3) parses and logs the header;
    actual platform-specific post-processing is deferred to a later phase.
    """
    def route(self, ephemeral_text: str):
        if not ephemeral_text:
            return (None, {})
        b64 = base64.b64encode(ephemeral_text.encode("utf-8")).decode("ascii")
        return (None, {"X-Hermes-Ephemeral": b64})
