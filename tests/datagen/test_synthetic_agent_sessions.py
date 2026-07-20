from inference_perf.datagen.synthetic_themes import load_theme, Theme, GENERIC_THEME, DEFAULT_SYSTEM_PROMPT  # noqa: F401
from inference_perf.datagen.synthetic_agent_sessions import (
    session_seed,
    child_rng,
    fit_filler,
    FILLER_MARKER,
    TOOL_CALL_MARGIN,
    build_graph_for_session,
)
from inference_perf.config.common import Distribution
from inference_perf.config.datagen.replay import SyntheticAgentSessionsConfig


def test_load_bundled_theme():
    t = load_theme("db2_latency_incident")
    assert isinstance(t, Theme)
    assert t.name == "db2_latency_incident"
    assert t.objective_template  # non-empty
    assert len(t.verbs) >= 3
    assert t.tool_names  # at least one tool


def test_generic_theme_is_valid():
    assert isinstance(GENERIC_THEME, Theme)
    assert GENERIC_THEME.objective_template


def test_load_unknown_theme_raises():
    import pytest

    with pytest.raises(ValueError):
        load_theme("nonexistent_theme_xyz")


def test_config_requires_the_four_required_fields():
    import pytest
    from pydantic import ValidationError
    from inference_perf.config.datagen.replay import SyntheticAgentSessionsConfig

    with pytest.raises(ValidationError):
        SyntheticAgentSessionsConfig()  # missing num_sessions/rounds/fanout/theme_mix


def test_config_valid_minimal():
    from inference_perf.config.common import Distribution
    from inference_perf.config.datagen.replay import SyntheticAgentSessionsConfig
    from inference_perf.config.datagen.replay import BadToolCallHandling

    cfg = SyntheticAgentSessionsConfig(
        num_sessions=10,
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        theme_mix={"db2_latency_incident": 1.0},
        input_tokens_per_turn=Distribution(type="fixed", mean=500),
        output_tokens_per_turn=Distribution(type="fixed", mean=100),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
    )
    assert cfg.seed == 42
    assert cfg.max_depth == 2
    assert cfg.max_events_per_session == 64
    assert cfg.inject_random_session_id is False
    assert cfg.duplicate_sessions_target is None
    assert cfg.override_tool_call_max_tokens is False
    assert cfg.bad_tool_call_handling == BadToolCallHandling.NONE


def test_session_seed_stable_across_calls_and_processes():
    # Must NOT depend on PYTHONHASHSEED or process -- pure function of inputs.
    a = session_seed(42, 17)
    b = session_seed(42, 17)
    assert a == b
    assert session_seed(42, 18) != a  # different index -> different seed


def test_child_rng_path_derived_independent():
    r1 = child_rng(session_seed(42, 0), 1, 2, 3)
    r2 = child_rng(session_seed(42, 0), 1, 2, 3)
    assert r1.integers(0, 1_000_000) == r2.integers(0, 1_000_000)  # reproducible
    r3 = child_rng(session_seed(42, 0), 1, 2, 4)  # different path
    assert r3.integers(0, 1_000_000) != r1.integers(0, 1_000_000)


class _FakeTok:
    # 1 token per whitespace-word, deterministic -- good enough to test budget logic
    def count_tokens(self, text, add_special_tokens=True):
        return len(text.split())

    def get_tokenizer(self):
        raise NotImplementedError


def test_fit_filler_negative_budget_returns_fixed_only_no_marker():
    tok = _FakeTok()
    fixed = "objective line here"  # 3 tokens
    out = fit_filler(tok, target_tokens=2, fixed_content=fixed, rng=None)  # target < fixed
    assert FILLER_MARKER not in out
    assert out == fixed  # floored to fixed content, no crash


def test_tool_call_margin_value():
    assert TOOL_CALL_MARGIN == 64


# --- Task 8: the seeded single-agent walk ---------------------------------


class _WordTok:
    def count_tokens(self, text, add_special_tokens=True):
        return max(1, len(str(text).split()))

    def get_tokenizer(self):
        raise NotImplementedError


def _cfg(**kw):
    base = dict(
        num_sessions=5,
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        theme_mix={"generic": 1.0},
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
    )
    base.update(kw)
    return SyntheticAgentSessionsConfig(**base)


def test_single_agent_graph_structure():
    g = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), session_index=0)
    assert len(g.events) >= 1
    for ev in g.events.values():
        assert ev.call.messages, "every event has non-empty messages (inv #4)"
        # inv #3: #role:tool == #tool_calls in each event's messages
        n_tool_calls = sum(len(m.get("tool_calls", [])) for m in ev.call.messages if m.get("tool_calls"))
        n_tool_msgs = sum(1 for m in ev.call.messages if m.get("role") == "tool")
        assert n_tool_msgs == n_tool_calls
        # inv #2: each tool_definition has a top-level name
        for td in ev.call.tool_definitions or []:
            assert "name" in td
        # inv #1: tool-call arguments are json.dumps-ed strings
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                assert isinstance(tc["function"]["arguments"], str)


def test_determinism_same_index_same_graph():
    g1 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    g2 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    assert list(g1.events.keys()) == list(g2.events.keys())  # same ids, same insertion order


def test_event_budget_caps_rounds():
    cfg = _cfg(rounds_per_session=Distribution(type="fixed", mean=100), max_events_per_session=6)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    assert len(g.events) <= 6


# --- Task 9: recursive fan-out + merge via tool_output --------------------


def test_fanout_produces_subagents_and_valid_merge():
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    # a sub-agent exists (depth >= 1): some event id contains ":sub"
    assert any(":sub" in eid for eid in g.events), "sub-agents spawned"
    # every dispatch_agent tool_call has a matching role:tool result (inv #3, no dangling)
    for ev in g.events.values():
        n_calls = sum(len(m.get("tool_calls", [])) for m in ev.call.messages if m.get("tool_calls"))
        n_tool = sum(1 for m in ev.call.messages if m.get("role") == "tool")
        assert n_tool == n_calls


def test_no_agent_beyond_max_depth():
    import re

    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    # depth encoded in id as ":dN:"; assert none exceeds max_depth
    for eid in g.events:
        m = re.search(r":d(\d+):", eid)
        if m:
            assert int(m.group(1)) <= 1


def test_subagent_first_call_carries_identical_system_head():
    # §4.2/§6 option (b): the invariant system head rides EVERY agent's first
    # call, byte-identical. Verify a sub-agent's first (dispatch) event carries
    # the same {role:"system"} message the root's first call gets.
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        shared_system_prompt_len=32,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)

    def _system_msg(ev):
        for m in ev.call.messages:
            if m.get("role") == "system":
                return m
        return None

    # root first call = the sole root event's principal
    root_id = g.root_event_ids[0]
    root_system = _system_msg(g.events[root_id])
    assert root_system is not None, "root first call carries a system head"

    # a sub-agent's first event: the child's principal (the ':sub' branch's first event)
    sub_firsts = [ev for eid, ev in g.events.items() if ":sub" in eid and ":principal" in eid]
    assert sub_firsts, "at least one sub-agent principal event exists"
    for ev in sub_firsts:
        sm = _system_msg(ev)
        assert sm is not None, "sub-agent first call carries a system head"
        assert sm == root_system, "sub-agent system head is byte-identical to root's"
        # aliasing guard: must be a distinct object (a copy), not the same dict
        assert sm is not root_system, "system head is copied per event, not aliased"


def test_event_budget_cost_is_k_plus_2_per_round():
    # A round emits 1 principal + k tool-turn events (each tool-turn is ONE
    # event packing [tool_call msg, tool result msg]) + 1 answer = k + 2
    # events -- NOT 2*k + 2. With tool_turns_per_loop fixed at k=2, each round
    # costs exactly 4 events. A budget of 8 fits exactly 2 whole rounds: if
    # the cost formula over-counts (e.g. treats a round as 2*k+2 = 6 events),
    # the budget would only fit 1 round, and this assertion would catch it.
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=100),
        max_events_per_session=8,
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    assert len(g.events) == 8  # exactly 2 full rounds of (k + 2) = 4 events


# --- Task 10: the generator class (lazy build + theme weighting) ----------


def _min_api():
    from inference_perf.config import APIConfig, APIType

    return APIConfig(type=APIType.Chat, streaming=False)


def test_generator_builds_session_lazily():
    from inference_perf.config.datagen.config import DataConfig, DataGenType
    from inference_perf.datagen.synthetic_agent_sessions import SyntheticAgentSessionsDataGenerator

    data = DataConfig(type=DataGenType.SyntheticAgentSessions, synthetic_agent_sessions=_cfg(num_sessions=4))
    gen = SyntheticAgentSessionsDataGenerator(
        api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1
    )
    assert gen.get_session_count() == 4
    gen._ensure_session_built(0)
    assert gen.sessions[0] is not None
    # determinism: two generators, same index -> same event ids
    gen2 = SyntheticAgentSessionsDataGenerator(
        api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1
    )
    gen2._ensure_session_built(0)
    assert list(gen.sessions[0].graph.events.keys()) == list(gen2.sessions[0].graph.events.keys())


# --- Task 11: main.py dispatch wiring --------------------------------------


def test_dispatch_resolves_synthetic_generator():
    # Minimal: assert the generator class is importable and the enum value maps.
    from inference_perf.config.datagen.config import DataGenType
    from inference_perf.datagen import SyntheticAgentSessionsDataGenerator

    assert DataGenType.SyntheticAgentSessions.value == "synthetic_agent_sessions"
    assert SyntheticAgentSessionsDataGenerator is not None


# --- Task 12: end-to-end integration guard (no dangling tool_call_ids) -----


def test_generated_fanout_session_has_no_dangling_tool_call_ids():
    """Build a fan-out session and walk every event's messages; assert no
    role:tool message references a tool_call_id absent from a preceding
    assistant tool_call in the SAME event. This is the exact invariant whose
    violation caused the live IndexError/dangling-id class of bug."""
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    assert len(g.events) > 4, "fan-out actually materialized"
    for ev in g.events.values():
        call_ids = {tc["id"] for m in ev.call.messages for tc in (m.get("tool_calls") or [])}
        tool_ids = {m["tool_call_id"] for m in ev.call.messages if m.get("role") == "tool"}
        assert tool_ids <= call_ids, f"dangling tool_call_id in {ev.event_id}"
