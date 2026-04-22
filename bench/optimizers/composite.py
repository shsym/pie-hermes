"""Compose multiple PromptOptimizers by threading the output of one into the next."""
from __future__ import annotations
from typing import Sequence


class Composite:
    """Run optimizers in order; each sees the previous one's reshaped parts.

    extra_headers are merged (later optimizers can overwrite earlier ones
    on key collision — last write wins). Accepts optimizers that return
    either a PromptOptimization or a bare list (Phase-0 form)."""

    def __init__(self, optimizers: Sequence):
        self._optimizers = list(optimizers)

    def optimize(self, parts):
        from agent.prompt_optimizer import PromptOptimization
        cur_parts = list(parts)
        cur_headers: dict[str, str] = {}
        for opt in self._optimizers:
            r = opt.optimize(cur_parts)
            if hasattr(r, "parts"):
                cur_parts = list(r.parts)
                cur_headers.update(r.extra_headers)
            else:
                cur_parts = list(r)
        return PromptOptimization(parts=cur_parts, extra_headers=cur_headers)
