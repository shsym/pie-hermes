"""VariantSwapOptimizer: strip model-family guidance blocks from prompt_parts
and emit an ``X-Hermes-Variant`` header so the inferlet can inject the
equivalent pre-exported KV segment. Phase 1P stub — model-name-aware
mapping lands in Task 1A.4."""
from __future__ import annotations


class VariantSwapOptimizer:
    def optimize(self, parts):
        from agent.prompt_optimizer import PromptOptimization
        from agent.prompt_builder import (
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
        )
        dropped_variant = "none"
        filtered: list[str] = []
        for p in parts:
            if p == OPENAI_MODEL_EXECUTION_GUIDANCE:
                dropped_variant = "codex"
                continue
            if p == GOOGLE_MODEL_OPERATIONAL_GUIDANCE:
                dropped_variant = "google"
                continue
            if p == TOOL_USE_ENFORCEMENT_GUIDANCE:
                # Base variant handle subsumes the enforcement text.
                continue
            filtered.append(p)
        return PromptOptimization(
            parts=filtered,
            extra_headers={"X-Hermes-Variant": dropped_variant},
        )
