"""VariantSwapOptimizer: drop model-family guidance blocks from prompt_parts
and emit an ``X-Hermes-Variant`` header so the inferlet can inject the
equivalent pre-exported KV segment.

Variant detection is passive — we look at which of the three guidance
constants hermes-agent's _build_system_prompt decided to append to
prompt_parts based on ``self.model``. There is no separate model-name
provider: hermes-agent already did the model-name mapping upstream.

Stripping rule:
- OPENAI_MODEL_EXECUTION_GUIDANCE present → variant='codex',
  strip BOTH OPENAI and TOOL_USE_ENFORCEMENT_GUIDANCE blocks.
- Elif GOOGLE_MODEL_OPERATIONAL_GUIDANCE present → variant='google',
  strip BOTH GOOGLE and TOOL_USE_ENFORCEMENT_GUIDANCE blocks.
- Else → variant='none'; strip nothing (the inferlet treats 'none' as
  a no-op import so there is no pre-exported KV to compensate for
  missing enforcement text).

The codex/google variant KV handles (pre-exported by
``scripts/export-variant-kv.py``) contain TOOL_USE + OPENAI /
TOOL_USE + GOOGLE concatenated — so stripping TOOL_USE is correct
and non-duplicative when those variants are active.

The ``X-Hermes-Variant: none`` header is emitted unconditionally even
when nothing is stripped, for telemetry consistency and to exercise the
header-parsing path during captures."""
from __future__ import annotations


class VariantSwapOptimizer:
    def optimize(self, parts):
        from agent.prompt_optimizer import PromptOptimization
        from agent.prompt_builder import (
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
        )

        has_openai = OPENAI_MODEL_EXECUTION_GUIDANCE in parts
        has_google = GOOGLE_MODEL_OPERATIONAL_GUIDANCE in parts

        if has_openai:
            variant = "codex"
        elif has_google:
            variant = "google"
        else:
            variant = "none"

        if variant == "none":
            # No pre-exported KV to swap in; pass through untouched.
            return PromptOptimization(
                parts=list(parts),
                extra_headers={"X-Hermes-Variant": "none"},
            )

        # Variant active: strip TOOL_USE and the matching branded block so
        # the prompt doesn't duplicate what the imported KV will supply.
        drop = {TOOL_USE_ENFORCEMENT_GUIDANCE,
                OPENAI_MODEL_EXECUTION_GUIDANCE,
                GOOGLE_MODEL_OPERATIONAL_GUIDANCE}
        filtered = [p for p in parts if p not in drop]
        return PromptOptimization(
            parts=filtered,
            extra_headers={"X-Hermes-Variant": variant},
        )
