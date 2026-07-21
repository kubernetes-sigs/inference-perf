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


# --- Large-target scaling (real tokenizer) --------------------------------
#
# These guard two bugs that only surface past the tokenizer's truncation
# ceiling (SmolLM2 model_max_length=8192): (A) fit_filler silently capped at
# ~8192 tokens because count_tokens truncates, so the loop couldn't measure
# beyond it; (B) the re-tokenizing loop was O(target) slow (tens of seconds
# per turn). A word-count proxy tokenizer would HIDE bug A (it never
# truncates), so at least one test must exercise the REAL tokenizer.

_REAL_TOKENIZER_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _real_tokenizer():
    """Load the real CustomTokenizer, or skip if it can't be loaded offline."""
    import pytest

    try:
        from inference_perf.config import CustomTokenizerConfig
        from inference_perf.utils.custom_tokenizer import CustomTokenizer

        return CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=_REAL_TOKENIZER_MODEL))
    except Exception as e:  # network down / model unavailable in CI
        pytest.skip(f"real tokenizer {_REAL_TOKENIZER_MODEL} unavailable: {e}")


def _untruncated_token_count(ct, text: str) -> int:
    """Length of `text` in tokens WITHOUT the model_max_length truncation.

    count_tokens truncates at model_max_length (8192 here), so it cannot
    measure a 100K-token string. The underlying HF tokenizer called with
    truncation=False gives the true length.
    """
    return len(ct.get_tokenizer()(text, truncation=False, add_special_tokens=False)["input_ids"])


def test_fit_filler_reaches_large_target():
    # Bug A regression: a 50K-token target must NOT be silently capped at
    # ~8192. Measure UNTRUNCATED so we see past the tokenizer's ceiling.
    ct = _real_tokenizer()
    out = fit_filler(ct, target_tokens=50000, fixed_content="Objective: investigate the incident.", rng=None)
    n = _untruncated_token_count(ct, out)
    assert n >= 40000, f"fit_filler capped below target (bug A): got {n} tokens for target 50000"
    assert FILLER_MARKER in out, "filler was added, so the marker must be present"


def test_fit_filler_large_target_is_fast():
    # Bug B regression: sizing must be analytic, not an O(target) re-tokenizing
    # loop. 100K tokens must build in well under 5 seconds.
    import time

    ct = _real_tokenizer()
    start = time.time()
    out = fit_filler(ct, target_tokens=100000, fixed_content="Objective: investigate the incident.", rng=None)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"fit_filler too slow (bug B): {elapsed:.2f}s for target 100000"
    n = _untruncated_token_count(ct, out)
    assert n >= 80000, f"fit_filler capped below target (bug A): got {n} tokens for target 100000"


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


def test_event_budget_cost_is_k_plus_1_per_round():
    # Under the corrected event model a round emits 1 principal + k tool-turn
    # events, where the LAST tool turn's OUTPUT is the answer (no separate
    # answer event) = k + 1 events. With tool_turns_per_loop fixed at k=2 each
    # round costs exactly 3 events. A budget of 9 fits exactly 3 whole rounds
    # (3 * 3 = 9); a budget of 8 fits only 2 whole rounds (the 3rd would need 3
    # more, overflowing) and STOPS -- confirming the per-round cost is k+1, not
    # the old k+2.
    cfg9 = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=100),
        max_events_per_session=9,
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
    )
    g9 = build_graph_for_session(cfg9, GENERIC_THEME, _WordTok(), 0)
    assert len(g9.events) == 9, f"expected 3 rounds of (k+1)=3 events, got {len(g9.events)}"

    cfg6 = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=100),
        max_events_per_session=6,
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
    )
    g6 = build_graph_for_session(cfg6, GENERIC_THEME, _WordTok(), 0)
    # exactly 2 full rounds (6 events); the 3rd round can't even start its
    # principal (6 + 1 > 6), so it never begins. Result: exactly 6 events.
    assert len(g6.events) == 6, f"expected 2 full rounds of (k+1)=3 events, got {len(g6.events)}"


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


def _last_tool_call_group(ev):
    """Return (assistant_tool_calls, trailing_tool_results) for the LAST
    tool-call group in an event's transcript.

    Under the corrected event model a ':tN' event's input is the growing
    transcript ending in [<prior turns>, assistant(K calls), tool×K]. The K
    calls of THIS turn are the last assistant tool_call message; its results
    are the trailing role:tool messages. Prior turns may add earlier
    assistant/tool messages, so we look at the final group only."""
    calls = None
    for m in ev.call.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
    tool_msgs = [m for m in ev.call.messages if m.get("role") == "tool"]
    return calls, tool_msgs


def test_parallel_tool_calls_emits_k_calls_and_k_results():
    # parallel_tool_calls_per_turn fixed 3 -> the tool-turn event that carries a
    # turn's result reconstructs an assistant message with 3 tool_calls AND 3
    # role:tool results, ids matching 1:1 in positional order (inv #3).
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    turns = _find_tool_turn_events(g)
    assert turns, "at least one ordinary tool-turn event exists"
    ev = turns[0]
    calls, tool_msgs = _last_tool_call_group(ev)
    assert calls is not None, "the tool-turn event reconstructs an assistant tool_call message"
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
        calls, tool_msgs = _last_tool_call_group(ev)
        # the LAST tool-call group (this turn) has exactly 1 call + 1 result
        assert calls is not None and len(calls) == 1
        assert len(tool_msgs) == 1
        assert calls[0]["id"] == tool_msgs[0]["tool_call_id"]


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
    # session is just the principal (no tool turns): with fanout 0 and one
    # round, exactly 1 event and no ':tN' tool-turn event exists. The principal
    # IS the terminal call -- its answer is the OUTPUT, not a separate event.
    import re

    assert not any(re.search(r":t\d+$", eid) for eid in g.events), "no tool-loop turn emitted"
    assert len(g.events) == 1, f"expected principal only (answer is its output), got {sorted(g.events)}"


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
            f"{ev.event_id}: tool names {sorted(missing)} forced/emitted but not in tool_definitions {sorted(advertised)}"
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
        assert "dispatch_agent" not in _event_def_names(ev), f"{ev.event_id} advertised dispatch_agent without fan-out"


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


# --- Result-content fidelity: per-tool templates, no placeholder leakage ---


def _find_ordinary_tool_result_msgs(g):
    """Return all role:tool result messages emitted by ORDINARY tool-loop turns
    (id ends with ':tN'), paired with the call name that produced them."""
    import re

    out = []
    for eid, ev in g.events.items():
        if not re.search(r":t\d+$", eid):
            continue
        call_name_by_id = {}
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                call_name_by_id[tc["id"]] = tc["function"]["name"]
        for m in ev.call.messages:
            if m.get("role") == "tool":
                out.append((call_name_by_id.get(m["tool_call_id"]), m["content"]))
    return out


def test_tool_result_uses_per_tool_template():
    # db2 theme's get_bp_stats template is rich ("| time | bp | hit_ratio |"
    # table markers) and distinct from the generic 'default' template. Force
    # a small catalog (tool_definitions_per_agent=1) so the single advertised
    # tool is theme.tool_names[0] == "get_bp_stats" (per _tool_definitions'
    # cycling), guaranteeing every ordinary tool-turn call is get_bp_stats.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_definitions_per_agent=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    results = _find_ordinary_tool_result_msgs(g)
    assert results, "at least one ordinary tool-turn result exists"
    get_bp_stats_results = [content for name, content in results if name == "get_bp_stats"]
    assert get_bp_stats_results, "get_bp_stats was called at least once"
    for content in get_bp_stats_results:
        # Shape of the per-tool template, not the generic default.
        assert "| time | bp | hit_ratio |" in content, f"expected get_bp_stats table shape, got: {content!r}"
        assert not content.startswith("result for "), f"fell back to the generic default template: {content!r}"


def test_tool_result_no_literal_placeholders():
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_definitions_per_agent=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    results = _find_ordinary_tool_result_msgs(g)
    assert results, "at least one ordinary tool-turn result exists"
    import re

    for _, content in results:
        assert "{" not in content and "}" not in content, f"unfilled placeholder leaked: {content!r}"
        assert " x " not in content, f"literal entity stand-in leaked: {content!r}"
        assert "at t0" not in content, f"literal t0 stand-in leaked: {content!r}"
        # time-ish fields (t0, t1, ...) look like HH:MM:SS
        for m in re.findall(r"\b\d{1,2}:\d{2}:\d{2}\b", content):
            hh, mm, ss = (int(x) for x in m.split(":"))
            assert 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59, f"implausible timestamp {m!r} in {content!r}"


def test_tool_result_content_is_deterministic():
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_definitions_per_agent=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, theme, _WordTok(), session_index=7)
    g2 = build_graph_for_session(cfg, theme, _WordTok(), session_index=7)
    r1 = _find_ordinary_tool_result_msgs(g1)
    r2 = _find_ordinary_tool_result_msgs(g2)
    assert r1 == r2, "tool-result contents are not deterministic for the same (config, index)"


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


# --- Config validation: theme_mix and max_model_len fail-fast --------------


def test_theme_mix_empty_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix={})


def test_theme_mix_all_zero_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix={"generic": 0.0})


def test_theme_mix_negative_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix={"generic": -1.0})


def test_theme_mix_valid_accepted():
    # Regression guard: a normal, non-empty, positive-weight mix must still
    # construct without raising.
    cfg = _cfg(theme_mix={"generic": 0.5, "db2_latency_incident": 0.5})
    assert cfg.theme_mix == {"generic": 0.5, "db2_latency_incident": 0.5}


def test_max_model_len_overrun_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(max_model_len=1000, shared_system_prompt_len=2000)


def test_max_model_len_none_accepted():
    # No ceiling configured -> no fail-fast check performed.
    cfg = _cfg(max_model_len=None, shared_system_prompt_len=2000)
    assert cfg.max_model_len is None


def test_max_model_len_comfortable_fit_accepted():
    # shared_system_prompt_len + input_tokens_per_turn.mean well under the cap.
    cfg = _cfg(
        max_model_len=100_000,
        shared_system_prompt_len=100,
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
    )
    assert cfg.max_model_len == 100_000


# --- Event-model fix: each call carries the cumulative transcript ----------
#
# Every event is exactly ONE LLM call whose INPUT is the growing conversation
# transcript ending in a user or tool message; the assistant reply is the
# event's OUTPUT (expected_output), NOT a separate lone-assistant event.


def _last_role(ev):
    """Role of the last message in an event's input transcript."""
    return ev.call.messages[-1].get("role") if ev.call.messages else None


def _is_lone_assistant(ev):
    """True iff the event's input is a single assistant message (the bogus
    lone-assistant 'answer' call the old model emitted)."""
    msgs = ev.call.messages
    return len(msgs) == 1 and msgs[0].get("role") == "assistant"


def test_no_lone_assistant_input():
    # THE core assertion. Across every shape (bare, tool-loop, interactive,
    # fan-out), NO event's input is a lone assistant message, and EVERY event's
    # input ends in role 'user' or 'tool' -- never 'assistant'.
    shapes = {
        "bare": _cfg(
            tool_definitions_per_agent=Distribution(type="fixed", mean=0),
            tool_turns_per_loop=Distribution(type="fixed", mean=2),
            fanout_probability=0.0,
        ),
        "tool_loop": _cfg(
            tool_turns_per_loop=Distribution(type="fixed", mean=3),
            fanout_probability=0.0,
        ),
        "interactive": _cfg(
            rounds_per_session=Distribution(type="fixed", mean=3),
            tool_turns_per_loop=Distribution(type="fixed", mean=1),
            fanout_probability=0.0,
            max_events_per_session=2048,
        ),
        "fanout": _cfg(
            fanout_probability=1.0,
            max_depth=2,
            sub_agents_per_spawn=Distribution(type="fixed", mean=2),
            max_events_per_session=2048,
            tool_turns_per_loop=Distribution(type="fixed", mean=1),
        ),
    }
    for name, cfg in shapes.items():
        for idx in range(3):
            g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
            assert g.events, f"{name}[{idx}] built no events"
            for ev in g.events.values():
                assert not _is_lone_assistant(ev), f"{name}[{idx}] {ev.event_id}: lone-assistant input"
                assert _last_role(ev) in ("user", "tool"), (
                    f"{name}[{idx}] {ev.event_id}: input ends in {_last_role(ev)!r}, not user/tool"
                )


def test_bare_single_round_is_one_event():
    # rounds=1, k=0 (empty catalog), fanout 0 -> EXACTLY 1 event. Its input is
    # [user] (+ system if configured); its expected_output is the (non-empty)
    # answer text; it is NOT a tool call.
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_definitions_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(g.events) == 1, f"expected exactly 1 event, got {sorted(g.events)}"
    ev = next(iter(g.events.values()))
    roles = [m.get("role") for m in ev.call.messages]
    assert roles == ["user"], f"bare principal input should be [user], got {roles}"
    assert ev.call.expected_output_is_tool_call is False
    assert ev.call.expected_output, "terminal answer text must be non-empty"

    # With a system prompt, the input is [system, user].
    cfg_sys = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_definitions_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=0.0,
        shared_system_prompt_len=16,
    )
    gs = build_graph_for_session(cfg_sys, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(gs.events) == 1
    evs = next(iter(gs.events.values()))
    assert [m.get("role") for m in evs.call.messages] == ["system", "user"]


def test_tool_loop_context_grows():
    # single-agent k=3, fanout 0 -> the agent's events' input message counts
    # grow like the OTel reference / real Exgentic (1, 3, 5, 7 for k=3, ignoring
    # any system head). principal + t0 + t1 + t2 = 4 events.
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    # k+1 = 4 events for one round.
    assert len(g.events) == 4, f"expected principal + 3 tool turns = 4 events, got {sorted(g.events)}"
    # Order by id suffix: principal, t0, t1, t2.
    ordered = sorted(g.events.values(), key=lambda e: (0 if e.event_id.endswith(":principal") else 1, e.event_id))
    lengths = [len(e.call.messages) for e in ordered]
    # strictly monotonically increasing, growing by 2 per turn (1,3,5,7).
    assert lengths == sorted(lengths), f"input lengths not monotonic: {lengths}"
    assert lengths[0] == 1, f"principal input should be [user] (1 msg), got {lengths[0]}"
    for a, b in zip(lengths, lengths[1:], strict=False):
        assert b - a == 2, f"tool loop should grow by 2 per turn (assistant+tool), got {lengths}"
    assert lengths == [1, 3, 5, 7], f"expected 1,3,5,7 growth, got {lengths}"


def _drive_substitution(target_ev, prior_by_source):
    """Drive target_ev through the REAL _build_messages_with_substitution.

    prior_by_source maps a source_event_id -> (input_messages, output_message)
    to populate the registry for the target's predecessors. Returns the
    reconstructed message list (raises if substitution mis-slices)."""
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    for src, (in_msgs, out_msg) in prior_by_source.items():
        out_text = out_msg.get("content", "") if out_msg else ""
        registry.record(src, out_text or "x", messages=list(in_msgs), output_message=out_msg)

    ev = SessionChatCompletionAPIData(
        messages=[],
        max_tokens=50,
        event_id=target_ev.event_id,
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=list(target_ev.predecessor_event_ids),
        input_segments=list(target_ev.call.input_segments),
        original_messages=list(target_ev.call.messages),
    )
    return ev._build_messages_with_substitution()


def _ordered_agent_events(g, agent_prefix):
    """Return an agent's events in build order (principal first, then t0, t1...)."""
    import re

    evs = [ev for eid, ev in g.events.items() if eid.startswith(agent_prefix + ":")]

    def _key(ev):
        eid = ev.event_id
        if eid.endswith(":principal"):
            return (0, 0)
        m = re.search(r":t(\d+)$", eid)
        if m:
            return (1, int(m.group(1)))
        return (2, eid)

    return sorted(evs, key=_key)


def test_substitution_survives_all_shapes():
    # Drive tool-loop events AND a fan-out merge through the REAL substitution
    # with a populated registry: no IndexError, transcript reconstructs, prior
    # turns are present.
    # --- tool loop ---
    cfg = _cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    ordered = _ordered_agent_events(g, "synthN0:r0")
    # Walk the chain, simulating live outputs, feeding each event's rebuilt
    # input (its replay `messages`) forward into the registry for the next.
    live_inputs = {}
    for ev in ordered:
        prior_by_source = {}
        for seg in ev.call.input_segments:
            if seg.source_event_id is not None:
                src = seg.source_event_id
                in_msgs = live_inputs.get(src, [])
                # Fabricate the source's live output_message: a tool call if the
                # source's expected output was a tool call, else plain answer.
                src_ev = g.events[src]
                if src_ev.call.expected_output_is_tool_call:
                    # reuse the build-time placeholder tool_calls from THIS event's
                    # output slot so ids line up for the no-dangling post-pass.
                    out_msg = {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": f"live_{src}", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                        ],
                    }
                else:
                    out_msg = {"role": "assistant", "content": f"LIVE-OUT-{src}"}
                prior_by_source[src] = (in_msgs, out_msg)
        result = _drive_substitution(ev, prior_by_source)  # must not raise IndexError
        assert result, f"{ev.event_id}: substitution produced empty input"
        assert result[-1].get("role") in ("user", "tool"), f"{ev.event_id}: rebuilt input ends in assistant"
        live_inputs[ev.event_id] = result
    # the last (terminal) event's rebuilt input carries the whole growing loop
    assert len(live_inputs[ordered[-1].event_id]) == 7, "terminal tool-loop input did not accumulate to 1+2*3=7"

    # --- fan-out merge ---
    fcfg = _cfg(
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
    )
    fg = build_graph_for_session(fcfg, GENERIC_THEME, _WordTok(), session_index=0)
    merges = [ev for eid, ev in fg.events.items() if eid.endswith(":merge")]
    assert merges, "fan-out merge event exists"
    for merge in merges:
        prior_by_source = {}
        for seg in merge.call.input_segments:
            if seg.source_event_id is None:
                continue
            src = seg.source_event_id
            if seg.type == "output":
                # dispatch event -> live dispatch tool call
                out_msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": f"live_{src}", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}
                    ],
                }
                prior_by_source[src] = ([], out_msg)
            elif seg.type == "tool_output":
                out_msg = {"role": "assistant", "content": f"CHILD-ANSWER-{src}"}
                prior_by_source[src] = ([], out_msg)
            elif seg.type == "shared":
                # the pre-spawn transcript source: give it a small balanced input
                prior_by_source[src] = ([{"role": "user", "content": "pre-spawn task"}], None)
        result = _drive_substitution(merge, prior_by_source)  # must not raise
        assert result, "merge substitution produced empty input"
        # a child answer text was injected into a role:tool slot
        joined = " ".join(str(m.get("content", "")) for m in result)
        assert "CHILD-ANSWER-" in joined, "child answer not injected into merge tool slot"
