"""Register pie-hermes PromptOptimizers with the hermes-agent hook.

Called from ``bootstrap/sitecustomize.py`` at Python startup (same path
as the capture hook). Safe no-op when hermes-agent isn't importable
(e.g. on the host venv for local pytest runs that don't install hermes
deps)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    try:
        from agent.prompt_optimizer import register, register_ephemeral_router
    except ImportError:
        logger.debug("install_optimizers: hermes-agent not importable; skipping")
        return

    from bench.optimizers.composite import Composite
    from bench.optimizers.variant_swap import VariantSwapOptimizer
    from bench.optimizers.ephemeral_oob import EphemeralOOBOptimizer, EphemeralOOBRouter

    register(Composite([VariantSwapOptimizer(), EphemeralOOBOptimizer()]))
    register_ephemeral_router(EphemeralOOBRouter())
    logger.info(
        "install_optimizers: registered Composite[VariantSwap, EphemeralOOB] "
        "and EphemeralOOBRouter"
    )
