"""install_optimizers.install() registers the Phase-1 optimizers with the
hermes-agent PromptOptimizer hook."""
import sys
from pathlib import Path

# The hermes-agent submodule isn't on sys.path by default for this repo's
# tests — its own venv + bootstrap handle that at runtime. For testing
# the optimizer contract we push it onto sys.path up-front.
SUBMODULE = Path(__file__).resolve().parent.parent / "hermes-agent"
if str(SUBMODULE) not in sys.path:
    sys.path.insert(0, str(SUBMODULE))


def test_install_registers_composite_optimizer():
    from agent.prompt_optimizer import get, reset_default, IdentityOptimizer
    from bench.install_optimizers import install

    reset_default()
    assert isinstance(get(), IdentityOptimizer)
    install()
    registered = get()
    assert not isinstance(registered, IdentityOptimizer), \
        "install() did not replace the default identity optimizer"
    reset_default()


def test_installed_optimizer_strips_variant_blocks():
    from agent.prompt_optimizer import get, reset_default
    from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
    from bench.install_optimizers import install

    reset_default()
    install()
    try:
        result = get().optimize([
            "identity block",
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            "timestamp",
        ])
    finally:
        reset_default()
    # TOOL_USE_ENFORCEMENT_GUIDANCE should be gone from parts.
    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in result.parts
    # The remaining parts are the non-variant blocks, in order.
    assert result.parts == ["identity block", "timestamp"]
    # Header should request the matching variant — "none" since the Qwen-like
    # current model doesn't trigger GPT or Gemini blocks.
    assert result.extra_headers.get("X-Hermes-Variant") == "none"


def test_composite_ephemeral_is_passthrough_for_now():
    """EphemeralOOBOptimizer is a stub in Phase 1P; real work lives in 1G.2."""
    from agent.prompt_optimizer import reset_default
    from bench.optimizers.composite import Composite
    from bench.optimizers.ephemeral_oob import EphemeralOOBOptimizer

    reset_default()
    comp = Composite([EphemeralOOBOptimizer()])
    result = comp.optimize(["a", "b"])
    assert result.parts == ["a", "b"]
    assert result.extra_headers == {}
