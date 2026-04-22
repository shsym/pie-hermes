"""The driver's SCENARIOS dict must carry the Phase-1 gateway scenarios so
Task 1V.6 can run them via SCENARIO_FILTER."""
from bench.driver import SCENARIOS


def test_gateway_ephemeral_chat_present():
    assert "gateway-ephemeral-chat" in SCENARIOS
    query, max_turns, description, tags = SCENARIOS["gateway-ephemeral-chat"]
    assert "gateway" in tags
    assert "ephemeral" in tags


def test_gateway_ephemeral_chat_3turn_present():
    assert "gateway-ephemeral-chat-3turn" in SCENARIOS
    _, _, _, tags = SCENARIOS["gateway-ephemeral-chat-3turn"]
    assert "gateway" in tags
    assert "ephemeral" in tags
    assert "multi-turn" in tags


def test_gateway_fixture_is_large_enough_to_measure():
    """The fixture must be ~300+ tokens so Idea G's savings are measurable."""
    from bench.driver import GATEWAY_EPHEMERAL_FIXTURE
    # Rough check: 4 chars per token approximation
    approx_tokens = len(GATEWAY_EPHEMERAL_FIXTURE) / 4
    assert approx_tokens >= 250, (
        f"Fixture too small to demonstrate Idea G's win: ~{approx_tokens:.0f} tokens"
    )
    assert approx_tokens <= 500, (
        f"Fixture too large — would dominate capture; ~{approx_tokens:.0f} tokens"
    )


def test_gateway_fixture_contains_plausible_metadata():
    """Make sure we didn't copy-paste a test stub by mistake."""
    from bench.driver import GATEWAY_EPHEMERAL_FIXTURE
    # Sanity: real-gateway-shaped content, not placeholder lorem ipsum.
    for keyword in ("chat_id", "user_handle", "reply_to", "reactions", "platform"):
        assert keyword in GATEWAY_EPHEMERAL_FIXTURE, (
            f"Fixture missing plausible gateway metadata keyword: {keyword}"
        )
