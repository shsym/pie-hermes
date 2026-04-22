"""EphemeralOOBOptimizer: pass-through stub in Phase 1P. Real implementation
lives in 1G.2 (needs EphemeralRouter hook extension in hermes-agent)."""
from __future__ import annotations


class EphemeralOOBOptimizer:
    def optimize(self, parts):
        from agent.prompt_optimizer import PromptOptimization
        return PromptOptimization(parts=parts)
