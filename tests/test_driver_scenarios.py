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
