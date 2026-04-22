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


def test_probe_ephemeral_handle_recall_present():
    assert "probe-ephemeral-handle-recall" in SCENARIOS
    _, _, _, tags = SCENARIOS["probe-ephemeral-handle-recall"]
    assert "gateway" in tags    # picks up GATEWAY_EPHEMERAL_FIXTURE injection
    assert "ephemeral" in tags
    assert "probe" in tags


def test_probe_expectations_dict_present():
    from bench.driver import PROBE_EXPECTATIONS
    assert "probe-ephemeral-handle-recall" in PROBE_EXPECTATIONS
    spec = PROBE_EXPECTATIONS["probe-ephemeral-handle-recall"]
    assert spec["must_contain_all"] == ["@alice_demo", "1.42"]


def test_every_probe_scenario_has_expectations():
    """Every scenario tagged 'probe' must have a PROBE_EXPECTATIONS entry."""
    from bench.driver import PROBE_EXPECTATIONS
    probe_scenarios = {slug for slug, (_, _, _, tags) in SCENARIOS.items() if "probe" in tags}
    missing = probe_scenarios - PROBE_EXPECTATIONS.keys()
    assert not missing, f"Probe scenarios missing PROBE_EXPECTATIONS: {missing}"


def test_probe_must_contain_strings_actually_appear_in_fixture_for_handle_recall():
    """Smoke check: the probe asks the model to surface tokens that should be
    present in the GATEWAY_EPHEMERAL_FIXTURE injected for gateway-tagged
    scenarios. If the fixture changes and these tokens disappear, the probe
    becomes meaningless — flag that here, before pod time."""
    from bench.driver import GATEWAY_EPHEMERAL_FIXTURE, PROBE_EXPECTATIONS
    for substr in PROBE_EXPECTATIONS["probe-ephemeral-handle-recall"]["must_contain_all"]:
        assert substr in GATEWAY_EPHEMERAL_FIXTURE, (
            f"Probe expects model to mention {substr!r}, but it's not in the fixture"
        )
