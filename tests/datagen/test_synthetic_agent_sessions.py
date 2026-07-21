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
    gen = SyntheticAgentSessionsDataGenerator(api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1)
    assert gen.get_session_count() == 4
    gen._ensure_session_built(0)
    assert gen.sessions[0] is not None
    # determinism: two generators, same index -> same event ids
    gen2 = SyntheticAgentSessionsDataGenerator(api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1)
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


# --- Follow-up: input_tokens_per_turn must actually size input turns --------


def _principal_user_content(g):
    """Return the user-role content string of the sole root principal turn."""
    root_id = g.root_event_ids[0]
    ev = g.events[root_id]
    user_msgs = [m for m in ev.call.messages if m.get("role") == "user"]
    assert user_msgs, "principal turn has a user message"
    return user_msgs[-1]["content"]


def test_input_tokens_per_turn_is_honored():
    # Two graphs identical except input_tokens_per_turn; a larger target must
    # produce a larger (>=) principal user-turn token count. fit_filler is
    # best-candidate/approximate, so tolerate with >= not exact equality.
    tok = _WordTok()
    small = build_graph_for_session(
        _cfg(input_tokens_per_turn=Distribution(type="fixed", mean=20)), GENERIC_THEME, tok, session_index=0
    )
    large = build_graph_for_session(
        _cfg(input_tokens_per_turn=Distribution(type="fixed", mean=300)), GENERIC_THEME, tok, session_index=0
    )
    small_tokens = tok.count_tokens(_principal_user_content(small))
    large_tokens = tok.count_tokens(_principal_user_content(large))
    assert large_tokens > small_tokens, f"input_tokens_per_turn had no effect: small={small_tokens} large={large_tokens}"
    # And the larger one should be in the neighbourhood of its target (not tiny).
    assert large_tokens >= 200, f"large principal turn far below target: {large_tokens}"


def test_input_sizing_preserves_determinism_and_objective_text():
    tok = _WordTok()
    cfg = _cfg(input_tokens_per_turn=Distribution(type="fixed", mean=300))
    g1 = build_graph_for_session(cfg, GENERIC_THEME, tok, session_index=2)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, tok, session_index=2)
    # identical event-id list
    assert list(g1.events.keys()) == list(g2.events.keys())
    # identical principal-turn content (byte-for-byte)
    assert _principal_user_content(g1) == _principal_user_content(g2)
    # objective text is not lost: the rendered objective is still present verbatim
    # (it is the fixed_content prepended before the FILLER_MARKER).
    content = _principal_user_content(g1)
    assert FILLER_MARKER in content, "large target should have padded with a marker"
    objective_prefix = content.split(FILLER_MARKER)[0].strip()
    assert objective_prefix, "objective text preserved before the filler marker"


# --- Follow-up: parallel_tool_calls_per_turn on ordinary tool turns --------


def _find_tool_turn_events(g):
    """Return the ordinary tool-loop turn events (id ends with ':tN').

    These are the ORDINARY tool turns emitted in _build_agent's tool-loop
    (NOT dispatch events, NOT the merge). Their assistant message carries the
    K parallel calls and is followed by K role:tool results.
    """
    import re

    return [ev for eid, ev in g.events.items() if re.search(r":t\d+$", eid)]


def test_parallel_tool_calls_emits_k_calls_and_k_results():
    # parallel_tool_calls_per_turn fixed 3 -> an ordinary tool turn emits 3
    # tool_calls in its assistant message AND 3 role:tool results, ids matching
    # 1:1 in positional order (inv #3).
    cfg = _cfg(
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    turns = _find_tool_turn_events(g)
    assert turns, "at least one ordinary tool-turn event exists"
    ev = turns[0]
    assistant_msgs = [m for m in ev.call.messages if m.get("role") == "assistant"]
    tool_msgs = [m for m in ev.call.messages if m.get("role") == "tool"]
    assert len(assistant_msgs) == 1, "one assistant tool_call message per turn"
    calls = assistant_msgs[0]["tool_calls"]
    assert len(calls) == 3, f"expected 3 parallel calls, got {len(calls)}"
    assert len(tool_msgs) == 3, f"expected 3 role:tool results, got {len(tool_msgs)}"
    # ids match 1:1 in positional order (inv #3 positional)
    call_ids = [c["id"] for c in calls]
    result_ids = [m["tool_call_id"] for m in tool_msgs]
    assert call_ids == result_ids, f"ids not positionally matched: {call_ids} vs {result_ids}"
    assert len(set(call_ids)) == 3, "the 3 call ids are distinct"
    # inv #1: json.dumps args; inv #2: each call name is a top-level tool_def name
    def_names = {td["name"] for td in ev.call.tool_definitions or []}
    for c in calls:
        assert isinstance(c["function"]["arguments"], str)
        assert c["function"]["name"] in def_names, "call name absent from tool_definitions"


def test_parallel_default_is_single_call():
    # parallel_tool_calls_per_turn unset (None -> fallback fixed 1): an ordinary
    # tool turn has exactly 1 call + 1 result (unchanged default behavior).
    cfg = _cfg(
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    assert cfg.parallel_tool_calls_per_turn is None
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    turns = _find_tool_turn_events(g)
    assert turns, "at least one ordinary tool-turn event exists"
    for ev in turns:
        assistant_msgs = [m for m in ev.call.messages if m.get("role") == "assistant"]
        tool_msgs = [m for m in ev.call.messages if m.get("role") == "tool"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 1
        assert len(tool_msgs) == 1
        assert assistant_msgs[0]["tool_calls"][0]["id"] == tool_msgs[0]["tool_call_id"]


def test_dispatch_still_single_call_under_parallel_knob():
    # The knob must NOT leak into sub-agent dispatch turns: with parallel fixed 3
    # AND fanout forced, every dispatch_agent tool-call turn STILL has exactly 1
    # call (the fan-out mechanism depends on single-call dispatch).
    cfg = _cfg(
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    dispatch_events = [ev for eid, ev in g.events.items() if ":disp" in eid]
    assert dispatch_events, "fan-out dispatch events materialized"
    for ev in dispatch_events:
        # dispatch events carry NO stored assistant tool_call (0 calls, 0 results);
        # the single dispatch call is the EXPECTED output. Assert its expected
        # output is a single tool name and no parallel calls leaked in.
        assert ev.call.expected_output_is_tool_call is True
        assert ev.call.expected_output_tool_names == ["dispatch_agent"]
        n_calls = sum(len(m.get("tool_calls", [])) for m in ev.call.messages if m.get("tool_calls"))
        assert n_calls == 0, "dispatch event stores no parallel calls"


def test_parallel_tool_calls_preserves_determinism():
    cfg = _cfg(
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=1)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=1)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


# --- Gap-fix 1: tool_definitions_per_agent=0 is the bare non-agentic baseline --


def test_zero_tool_definitions_is_bare_baseline():
    # §8: tool_definitions_per_agent=0 -> NO tools advertised at all, and a
    # catalog-less agent cannot emit a forced tool call, so it just answers.
    cfg = _cfg(
        tool_definitions_per_agent=Distribution(type="fixed", mean=0),
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    assert g.events, "graph built"
    for ev in g.events.values():
        # every event advertises an EMPTY tool catalog
        assert ev.call.tool_definitions == [], f"{ev.event_id} advertised tools: {ev.call.tool_definitions}"
        # zero assistant tool_calls anywhere
        n_calls = sum(len(m.get("tool_calls", []) or []) for m in ev.call.messages)
        assert n_calls == 0, f"{ev.event_id} emitted a tool_call with an empty catalog"
        assert ev.call.expected_output_is_tool_call is False
    # session is just principal + answer (no tool turns): with fanout 0 and one
    # round, exactly 2 events and no ':tN' tool-turn event exists.
    import re

    assert not any(re.search(r":t\d+$", eid) for eid in g.events), "no tool-loop turn emitted"
    assert len(g.events) == 2, f"expected principal+answer only, got {sorted(g.events)}"


# --- Gap-fix 2: round-to-round context growth (spec §4.1) ------------------


def _principal_events_by_round(g):
    """Map round index -> the root principal event for that round."""
    import re

    out = {}
    for eid, ev in g.events.items():
        m = re.match(r"synthN\d+:r(\d+):principal$", eid)
        if m:
            out[int(m.group(1))] = ev
    return out


def test_interactive_rounds_carry_growing_context():
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    principals = _principal_events_by_round(g)
    assert set(principals) >= {0, 1, 2}, f"expected 3 rounds, got {sorted(principals)}"

    # Round 0 is a fresh single-turn prompt: no input_segments.
    assert principals[0].call.input_segments == [], "round 0 must be a fresh prompt (no segments)"

    # Rounds 1 and 2 carry [shared, output, unique] segments.
    for r in (1, 2):
        segs = principals[r].call.input_segments
        types = [s.type for s in segs]
        assert types == ["shared", "output", "unique"], f"round {r} segment layout: {types}"
        shared, output, unique = segs
        # cursor math: message_counts must sum to len(original_messages)
        assert shared.message_count + output.message_count + unique.message_count == len(principals[r].call.messages), (
            f"round {r} segment counts don't cover the messages"
        )
        assert output.message_count == 1
        assert unique.message_count == 1
        # BOTH substitution sources must ALSO be predecessors (require_async).
        pred_ids = set(principals[r].predecessor_event_ids)
        assert shared.source_event_id in pred_ids, f"round {r} shared source not a predecessor"
        assert output.source_event_id in pred_ids, f"round {r} output source not a predecessor"

    # Growing conversation: round-2 principal materializes MORE messages than round-0.
    assert len(principals[2].call.messages) > len(principals[0].call.messages), "context did not grow"
    assert len(principals[1].call.messages) > len(principals[0].call.messages)
    assert len(principals[2].call.messages) > len(principals[1].call.messages)


def test_round_k_survives_runtime_substitution():
    # Build a 3-round session, then run the round-2 principal event through the
    # ACTUAL runtime substitution (_build_messages_with_substitution) with a
    # registry populated for its predecessors — mirroring the tool_output tests.
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    principals = _principal_events_by_round(g)
    target = principals[2]
    shared_seg = target.call.input_segments[0]
    output_seg = target.call.input_segments[1]

    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    # shared source = round-1 principal: its stored INPUT must BE the growing
    # prefix. At replay that input is the substituted round-1 messages; here we
    # populate it with round-1 principal's own build-time messages (same length).
    round1_principal = principals[1]
    prior_answer_text = "ROUND-1 ANSWER TEXT MARKER"
    registry.record(
        shared_seg.source_event_id,
        "irrelevant",
        messages=list(round1_principal.call.messages),
    )
    # output source = round-1 answer event -> re-injects the prior answer.
    registry.record(
        output_seg.source_event_id,
        prior_answer_text,
        messages=[],
        output_message={"role": "assistant", "content": prior_answer_text},
    )

    ev = SessionChatCompletionAPIData(
        messages=[],
        max_tokens=50,
        event_id=target.event_id,
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=list(target.predecessor_event_ids),
        input_segments=list(target.call.input_segments),
        original_messages=list(target.call.messages),
    )

    result = ev._build_messages_with_substitution()  # must not raise IndexError

    # The reconstructed round-2 input carries the growing transcript: more than
    # one message, and the prior answer text is present.
    assert len(result) > 1, "round-2 reconstructed input collapsed to a single message"
    joined = " ".join(str(m.get("content", "")) for m in result)
    assert prior_answer_text in joined, "prior answer not re-injected into round-2 context"


def test_interactive_rounds_preserve_determinism():
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=4)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=4)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments


# --- inv #2: forced/emitted tool names must appear in tool_definitions -----


def _event_def_names(ev):
    """Top-level tool_definitions names advertised on an event."""
    return {td["name"] for td in (ev.call.tool_definitions or []) if "name" in td}


def _event_tool_call_names(ev):
    """Tool names appearing in this event's stored assistant tool_calls."""
    names = set()
    for m in ev.call.messages:
        for tc in m.get("tool_calls", []) or []:
            names.add(tc["function"]["name"])
    return names


def _assert_inv2_over_graph(g):
    """inv #2, general form: for EVERY event,
    {forced names} ∪ {names in message tool_calls} ⊆ {tool_definitions names}.

    This is the assertion whose absence let the forced-tool-degradation bug
    through: a dispatch event forced 'dispatch_agent' without advertising it.
    """
    for ev in g.events.values():
        advertised = _event_def_names(ev)
        forced = set(ev.call.expected_output_tool_names or [])
        emitted = _event_tool_call_names(ev)
        needed = forced | emitted
        missing = needed - advertised
        assert not missing, (
            f"{ev.event_id}: tool names {sorted(missing)} forced/emitted but not in "
            f"tool_definitions {sorted(advertised)}"
        )


def test_dispatch_agent_is_in_tool_definitions():
    # fanout forced, normal catalog: every event that forces a tool or stores a
    # tool_call must advertise that tool (inv #2). Specifically the dispatch
    # events must both FORCE dispatch_agent and ADVERTISE it, so replay's
    # tool_choice forcing does not silently degrade to "required".
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)

    dispatch_events = [ev for eid, ev in g.events.items() if ":disp" in eid]
    assert dispatch_events, "fan-out dispatch events materialized"
    for ev in dispatch_events:
        assert ev.call.expected_output_tool_names == ["dispatch_agent"], "dispatch event forces dispatch_agent"
        assert "dispatch_agent" in _event_def_names(ev), "dispatch_agent advertised in dispatch event tool_definitions"

    # the merge event emits dispatch_agent calls in its message history -> inv #2 applies there too.
    merge_events = [ev for eid, ev in g.events.items() if eid.endswith(":merge")]
    assert merge_events, "fan-out merge event materialized"
    for ev in merge_events:
        assert "dispatch_agent" in _event_tool_call_names(ev), "merge emits dispatch_agent calls"
        assert "dispatch_agent" in _event_def_names(ev), "dispatch_agent advertised in merge tool_definitions"


def test_dispatch_agent_present_even_with_empty_theme_catalog():
    # tool_definitions_per_agent=0 + fanout: theme catalog is empty, but the
    # dispatch tool is STRUCTURAL, so dispatch events must advertise exactly
    # [dispatch_agent] (not []).
    cfg = _cfg(
        tool_definitions_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)

    dispatch_events = [ev for eid, ev in g.events.items() if ":disp" in eid]
    assert dispatch_events, "fan-out dispatch events materialized even with empty theme catalog"
    for ev in dispatch_events:
        defs = ev.call.tool_definitions or []
        names = [td["name"] for td in defs if "name" in td]
        assert names == ["dispatch_agent"], f"expected exactly [dispatch_agent], got {names}"


def test_no_dispatch_agent_when_no_fanout():
    # fanout_probability=0.0: single-agent catalogs stay clean -- no
    # dispatch_agent advertised anywhere.
    cfg = _cfg(
        fanout_probability=0.0,
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)
    for ev in g.events.values():
        assert "dispatch_agent" not in _event_def_names(ev), (
            f"{ev.event_id} advertised dispatch_agent without fan-out"
        )


def test_inv2_holds_across_fanout_graph():
    # GENERAL inv #2 regression across a deeper fan-out graph.
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(g.events) > 4, "fan-out actually materialized"
    _assert_inv2_over_graph(g)


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
