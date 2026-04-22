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


def test_installed_optimizer_leaves_enforcement_alone_when_variant_is_none():
    """No GPT/Gemini block in parts → variant='none', no stripping (inferlet no-op)."""
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
    # Qwen-shape input: TOOL_USE only, no branded blocks.
    # Correct behavior: pass through; emit "none" variant for telemetry.
    assert result.parts == ["identity block", TOOL_USE_ENFORCEMENT_GUIDANCE, "timestamp"]
    assert result.extra_headers.get("X-Hermes-Variant") == "none"


def test_codex_variant_strips_both_openai_and_enforcement_blocks():
    from agent.prompt_optimizer import get, reset_default
    from agent.prompt_builder import (
        TOOL_USE_ENFORCEMENT_GUIDANCE,
        OPENAI_MODEL_EXECUTION_GUIDANCE,
    )
    from bench.install_optimizers import install

    reset_default()
    install()
    try:
        result = get().optimize([
            "identity",
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            "timestamp",
        ])
    finally:
        reset_default()
    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in result.parts
    assert OPENAI_MODEL_EXECUTION_GUIDANCE not in result.parts
    assert result.parts == ["identity", "timestamp"]
    assert result.extra_headers.get("X-Hermes-Variant") == "codex"


def test_google_variant_strips_both_google_and_enforcement_blocks():
    from agent.prompt_optimizer import get, reset_default
    from agent.prompt_builder import (
        TOOL_USE_ENFORCEMENT_GUIDANCE,
        GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    )
    from bench.install_optimizers import install

    reset_default()
    install()
    try:
        result = get().optimize([
            "identity",
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
            "timestamp",
        ])
    finally:
        reset_default()
    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in result.parts
    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in result.parts
    assert result.parts == ["identity", "timestamp"]
    assert result.extra_headers.get("X-Hermes-Variant") == "google"


def test_codex_wins_over_google_when_both_present():
    """Defensive: hermes-agent shouldn't emit both, but if it ever does,
    precedence is defined (openai first)."""
    from agent.prompt_optimizer import get, reset_default
    from agent.prompt_builder import (
        TOOL_USE_ENFORCEMENT_GUIDANCE,
        OPENAI_MODEL_EXECUTION_GUIDANCE,
        GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    )
    from bench.install_optimizers import install

    reset_default()
    install()
    try:
        result = get().optimize([
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
        ])
    finally:
        reset_default()
    # Both should be stripped + header goes to codex.
    assert OPENAI_MODEL_EXECUTION_GUIDANCE not in result.parts
    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in result.parts
    assert result.extra_headers.get("X-Hermes-Variant") == "codex"


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


def test_install_also_registers_ephemeral_router():
    from agent.prompt_optimizer import (
        get_ephemeral_router, reset_ephemeral_router, IdentityEphemeralRouter,
    )
    from bench.install_optimizers import install

    reset_ephemeral_router()
    assert isinstance(get_ephemeral_router(), IdentityEphemeralRouter)
    install()
    try:
        router = get_ephemeral_router()
        assert not isinstance(router, IdentityEphemeralRouter)
        reduced, headers = router.route("chat_id=123\nuser=alice")
        assert reduced is None
        assert "X-Hermes-Ephemeral" in headers
        # Round-trip the base64 to prove it's the original text.
        import base64
        assert (
            base64.b64decode(headers["X-Hermes-Ephemeral"]).decode("utf-8")
            == "chat_id=123\nuser=alice"
        )
    finally:
        reset_ephemeral_router()


def test_ephemeral_router_empty_text_returns_no_headers():
    from agent.prompt_optimizer import reset_ephemeral_router
    from bench.optimizers.ephemeral_oob import EphemeralOOBRouter

    reset_ephemeral_router()
    r = EphemeralOOBRouter()
    reduced, headers = r.route("")
    assert reduced is None
    assert headers == {}
