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


# -----------------------------------------------------------------------------
# Phase 2.1 / Idea D — read_context probe scenarios
# -----------------------------------------------------------------------------

def test_probe_read_context_no_call_present():
    assert "probe-read-context-no-call" in SCENARIOS
    _, _, _, tags = SCENARIOS["probe-read-context-no-call"]
    assert "probe" in tags
    assert "context-index" in tags


def test_probe_read_context_one_call_present():
    assert "probe-read-context-one-call" in SCENARIOS
    _, _, _, tags = SCENARIOS["probe-read-context-one-call"]
    assert "probe" in tags
    assert "context-index" in tags


def test_probe_read_context_no_call_seeded():
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    assert "probe-read-context-no-call" in SCENARIO_CONTEXT_SEEDS
    assert "AGENTS.md" in SCENARIO_CONTEXT_SEEDS["probe-read-context-no-call"]


def test_probe_read_context_one_call_seeded():
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    assert "probe-read-context-one-call" in SCENARIO_CONTEXT_SEEDS
    assert "AGENTS.md" in SCENARIO_CONTEXT_SEEDS["probe-read-context-one-call"]


def test_probe_read_context_no_call_expectations():
    from bench.driver import PROBE_EXPECTATIONS
    spec = PROBE_EXPECTATIONS["probe-read-context-no-call"]
    assert spec["must_not_call_tool"] == ["read_context"]
    assert spec["must_contain_all"] == ["4"]


def test_probe_read_context_one_call_expectations():
    from bench.driver import PROBE_EXPECTATIONS
    spec = PROBE_EXPECTATIONS["probe-read-context-one-call"]
    assert spec["must_call_tool"] == ["read_context"]
    assert spec["must_call_tool_with_args"]["read_context"]["section_id"] == (
        "agents.md#coding-style"
    )
    assert spec["must_contain_all"] == ["hc_"]


def test_one_call_seed_actually_contains_hc_prefix():
    """Without 'hc_' in the AGENTS.md, the answer isn't derivable from the
    seed and the probe loses its diagnostic value. Lock the fixture in."""
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    content = SCENARIO_CONTEXT_SEEDS["probe-read-context-one-call"]["AGENTS.md"]
    assert "hc_" in content


def test_one_call_seed_summary_does_not_leak_the_rule():
    """Capture-run #1 lesson: when the section's first non-blank body line
    contains the actionable rule, agent.context_summary._summary_for puts
    the rule INTO THE INDEX SUMMARY, and the model can answer from the
    summary without ever calling read_context. The probe then incorrectly
    fails its `must_call_tool: read_context` assertion even though the
    Phase-2.1 mechanism is working — false negative.

    This test locks in the fix: the first non-blank line of the
    'Coding style' section MUST be uninformative meta-prose that does NOT
    contain the answer token 'hc_'. The actual rule lives later in the body
    so the body load (via read_context) is required to find it."""
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    from agent.context_summary import summarize_context_file
    content = SCENARIO_CONTEXT_SEEDS["probe-read-context-one-call"]["AGENTS.md"]
    sections = {s.id: s for s in summarize_context_file("AGENTS.md", content)}
    assert "agents.md#coding-style" in sections
    coding_style = sections["agents.md#coding-style"]
    assert "hc_" not in coding_style.summary, (
        f"Summary leaks the answer: {coding_style.summary!r}. "
        f"First body line must be uninformative meta-prose so the model "
        f"is forced to call read_context to find 'hc_'."
    )
    # Body must still contain the answer (otherwise read_context can't help either).
    assert "hc_" in coding_style.body


def test_no_call_seed_does_not_contain_4_so_baseline_can_answer():
    """The no-call seed must NOT contain the answer to '2+2' — otherwise the
    model could plausibly cheat by loading the section. The whole point of
    this probe is that the model answers from its own knowledge without
    touching read_context. Lock that in."""
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    content = SCENARIO_CONTEXT_SEEDS["probe-read-context-no-call"]["AGENTS.md"]
    lower = content.lower()
    assert "2 plus 2" not in lower
    assert "4" not in content
    assert "two" not in lower


def test_every_probe_expectation_has_known_keys():
    """Catch typos in PROBE_EXPECTATIONS keys. Every key in every entry's
    value-dict must be one of the documented schema keys."""
    from bench.driver import PROBE_EXPECTATIONS
    known = {
        "must_contain_all",
        "must_not_contain",
        "must_call_tool",
        "must_not_call_tool",
        "must_call_tool_with_args",
    }
    for slug, spec in PROBE_EXPECTATIONS.items():
        unknown = set(spec.keys()) - known
        assert not unknown, (
            f"Probe {slug!r} has unknown expectation keys: {unknown}"
        )


def test_every_seeded_scenario_is_a_probe():
    """Seeds are an opt-in probe affordance. Non-probe scenarios shouldn't be
    silently seeded — that would change the cwd they run under and break
    comparability with prior captures."""
    from bench.driver import SCENARIO_CONTEXT_SEEDS
    for slug in SCENARIO_CONTEXT_SEEDS:
        assert slug in SCENARIOS, f"Seeded slug {slug!r} not in SCENARIOS"
        _, _, _, tags = SCENARIOS[slug]
        assert "probe" in tags, (
            f"Seeded scenario {slug!r} is missing the 'probe' tag"
        )
