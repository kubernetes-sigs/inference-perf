import pytest
from inference_perf.datagen.synthetic_themes import load_theme, Theme, GENERIC_THEME, DEFAULT_SYSTEM_PROMPT  # noqa: F401
from inference_perf.datagen.synthetic_agentic import (
    session_seed,
    child_rng,
    fit_filler,
    FILLER_OPEN,
    FILLER_CLOSE,
    TOOL_CALL_MARGIN,
    build_graph_for_session,
    theme_filler_words,
    _tool_definitions,
    _render_intro_doc,
    _render_theme_template,
    _tool_call_max_tokens,
    _accumulated_wire_tokens,
    _FALLBACK_TOOL_PARAMS,
)
from inference_perf.config.common import Distribution
from inference_perf.config.datagen.replay import SyntheticAgenticConfig, ContextCompactionConfig


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
    with pytest.raises(ValueError):
        load_theme("nonexistent_theme_xyz")


def test_config_requires_the_four_required_fields():
    from pydantic import ValidationError
    from inference_perf.config.datagen.replay import SyntheticAgenticConfig

    with pytest.raises(ValidationError):
        SyntheticAgenticConfig()  # missing num_sessions/rounds/fanout/theme_mix


def test_config_valid_minimal():
    from inference_perf.config.common import Distribution
    from inference_perf.config.datagen.replay import SyntheticAgenticConfig
    from inference_perf.config.datagen.replay import BadToolCallHandling

    cfg = SyntheticAgenticConfig(
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


def test_fit_filler_negative_budget_returns_fixed_only_no_wrapper():
    tok = _FakeTok()
    fixed = "objective line here"  # 3 tokens
    out = fit_filler(tok, target_tokens=2, fixed_content=fixed, rng=None)  # target < fixed
    assert FILLER_OPEN not in out and FILLER_CLOSE not in out
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
    # filler was added, so the wrapper block must be present, and the real
    # content must sit AFTER the closing tag (the order-correctness guard).
    assert FILLER_OPEN in out and FILLER_CLOSE in out, "filler was added, so the <context> wrapper must be present"
    fixed = "Objective: investigate the incident."
    assert out.index(fixed) > out.index(FILLER_CLOSE), "real content must follow the </context> block"


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
    return SyntheticAgenticConfig(**base)


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
    from inference_perf.datagen.synthetic_agentic import SyntheticAgenticDataGenerator

    data = DataConfig(type=DataGenType.SyntheticAgentic, synthetic_agentic=_cfg(num_sessions=4))
    gen = SyntheticAgenticDataGenerator(api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1)
    assert gen.get_session_count() == 4
    gen._ensure_session_built(0)
    assert gen.sessions[0] is not None
    # determinism: two generators, same index -> same event ids
    gen2 = SyntheticAgenticDataGenerator(api_config=_min_api(), config=data, tokenizer=_WordTok(), num_workers=1)
    gen2._ensure_session_built(0)
    assert list(gen.sessions[0].graph.events.keys()) == list(gen2.sessions[0].graph.events.keys())


# --- Task 11: main.py dispatch wiring --------------------------------------


def test_dispatch_resolves_synthetic_generator():
    # Minimal: assert the generator class is importable and the enum value maps.
    from inference_perf.config.datagen.config import DataGenType
    from inference_perf.datagen import SyntheticAgenticDataGenerator

    assert DataGenType.SyntheticAgentic.value == "synthetic_agentic"
    assert SyntheticAgenticDataGenerator is not None


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
    # (it is the fixed_content emitted AFTER the </ignore> filler block).
    content = _principal_user_content(g1)
    assert FILLER_OPEN in content and FILLER_CLOSE in content, "large target should have padded with a filler block"
    objective_suffix = content.rsplit(FILLER_CLOSE, 1)[-1].strip()
    assert objective_suffix, "objective text preserved after the filler block"


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


@pytest.mark.parametrize(
    "theme_mix",
    [{}, {"generic": 0.0}, {"generic": -1.0}],
    ids=["empty", "all_zero", "negative"],
)
def test_theme_mix_rejected(theme_mix):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix=theme_mix)


def test_theme_mix_valid_accepted():
    # Regression guard: a normal, non-empty, positive-weight mix must still
    # construct without raising.
    cfg = _cfg(theme_mix={"generic": 0.5, "db2_latency_incident": 0.5})
    assert cfg.theme_mix == {"generic": 0.5, "db2_latency_incident": 0.5}


@pytest.mark.parametrize(
    "kwargs, should_raise",
    [
        # overrun: shared prompt alone exceeds the cap -> fail-fast.
        ({"max_model_len": 1000, "shared_system_prompt_len": 2000}, True),
        # None: no ceiling configured -> no fail-fast check performed.
        ({"max_model_len": None, "shared_system_prompt_len": 2000}, False),
        # comfortable: shared_system_prompt_len + input mean well under the cap.
        (
            {
                "max_model_len": 100_000,
                "shared_system_prompt_len": 100,
                "input_tokens_per_turn": Distribution(type="fixed", mean=20),
            },
            False,
        ),
    ],
    ids=["overrun", "none", "comfortable_fit"],
)
def test_max_model_len_fail_fast(kwargs, should_raise):
    from pydantic import ValidationError

    if should_raise:
        with pytest.raises(ValidationError):
            _cfg(**kwargs)
    else:
        cfg = _cfg(**kwargs)
        assert cfg.max_model_len == kwargs["max_model_len"]


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


# --- Enrichment: tool descriptions, theme filler, intro doc ----------------


def test_both_themes_validate_with_new_fields():
    # Both enriched themes still load/validate and now carry the new fields.
    db2 = load_theme("db2_latency_incident")
    for theme in (GENERIC_THEME, db2):
        assert theme.tool_descriptions, f"{theme.name}: expected per-tool descriptions"
        assert theme.intro_doc_templates, f"{theme.name}: expected an intro doc template"
        assert theme.filler_templates, f"{theme.name}: expected theme filler snippets"
        # every advertised base tool has a description
        for name in theme.tool_names:
            assert name in theme.tool_descriptions, f"{theme.name}: tool {name} missing description"
        # aim for a richer catalog (~6-12 tools)
        assert 6 <= len(theme.tool_names) <= 12, f"{theme.name}: tool count {len(theme.tool_names)} out of range"


def test_tool_definitions_carry_descriptions():
    # Every emitted tool def has a non-empty description at BOTH the top level
    # and nested function level, while KEEPING the top-level name (inv #2).
    defs = _tool_definitions(GENERIC_THEME, 12)
    assert defs
    for td in defs:
        assert "name" in td, "top-level name preserved (inv #2)"
        assert td.get("description"), "top-level description present"
        assert td["function"].get("description"), "nested function.description present"
    # a real theme description is used, not just the generic fallback
    first = defs[0]
    assert first["description"] == GENERIC_THEME.tool_descriptions[GENERIC_THEME.tool_names[0]]


def test_tool_definitions_suffixed_duplicates_reuse_base_description():
    # Request MORE tools than the theme has: suffixed duplicates must be unique
    # and reuse their base tool's description.
    n = len(GENERIC_THEME.tool_names) + 3
    defs = _tool_definitions(GENERIC_THEME, n)
    names = [td["name"] for td in defs]
    assert len(names) == len(set(names)), "suffixed duplicate names must stay unique"
    base0 = GENERIC_THEME.tool_names[0]
    dup = next(td for td in defs if td["name"].startswith(base0 + "_"))
    assert dup["description"] == GENERIC_THEME.tool_descriptions[base0]


def test_theme_filler_words_are_domain_relevant_and_deterministic():
    # db2 filler pool is built from the theme's own snippets (NOT Shakespeare)
    # and is deterministic for a given (seed, path).
    db2 = load_theme("db2_latency_incident")
    seed = session_seed(42, 0)
    pool1 = theme_filler_words(db2, seed, (60,))
    pool2 = theme_filler_words(db2, seed, (60,))
    assert pool1 is not None and pool1 == pool2, "theme filler pool must be deterministic"
    text = " ".join(pool1)
    # a db2-specific token from the filler snippets is present
    assert any(tok in text for tok in ("DSNL027I", "bufferpool", "class2_cpu", "lock-wait")), (
        f"db2 filler pool not domain-relevant: {text[:200]!r}"
    )


def test_theme_without_filler_returns_none():
    # A theme with no filler_templates falls back (None -> corpus in fit_filler).
    bare = Theme(
        name="bare",
        verbs=["Do"],
        entities={"x": ["a"]},
        tool_names=["t"],
        result_templates={"default": "r {n0}"},
        objective_template="{verb}",
    )
    assert theme_filler_words(bare, 1, (60,)) is None


def test_fit_filler_uses_theme_word_pool():
    # With a theme word pool the padding words come FROM the pool, so a pool
    # token appears inside the <context> block and Shakespeare does not drive it.
    tok = _WordTok()
    pool = ["DSNL027I", "bufferpool", "lock-wait", "class2_cpu"]
    out = fit_filler(tok, target_tokens=200, fixed_content="OBJECTIVE-MARKER", rng=None, word_pool=pool)
    assert FILLER_OPEN in out and FILLER_CLOSE in out
    block = out.split(FILLER_CLOSE, 1)[0]
    assert any(w in block for w in pool), "theme pool words not used for filler"
    # real content preserved after the block
    assert out.rsplit(FILLER_CLOSE, 1)[-1].strip().endswith("OBJECTIVE-MARKER")


def test_intro_doc_rides_first_user_turn_and_is_deterministic():
    # The round-0 principal user turn carries the theme's long intro doc; it is
    # deterministic for a given (config, index) and preserved after filler.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=400),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    g2 = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    c1 = _principal_user_content(g1)
    c2 = _principal_user_content(g2)
    assert c1 == c2, "intro-doc-bearing first turn must be deterministic"
    # the real content after the filler block contains an intro-doc marker line
    real = c1.rsplit(FILLER_CLOSE, 1)[-1]
    assert any(marker in real for marker in ("SERVICENOW", "DISPLAY output", "OMEGAMON", "DSNJ031I", "-DIS")), (
        f"intro doc not present on first user turn: {real[:200]!r}"
    )
    # ... and the objective still trails it (intro is a PREFIX, objective last).
    assert "identify root cause" in real, "objective text lost after prepending intro doc"


def test_intro_doc_no_placeholder_leak():
    # Rendered intro docs must fill every placeholder (no {..} leak) for both themes.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        doc = _render_intro_doc(theme, session_seed(42, 3), (0, 61))
        assert doc, f"{theme.name}: intro doc empty"
        assert "{" not in doc and "}" not in doc, f"{theme.name}: unfilled placeholder in intro doc: {doc!r}"


def test_only_round_zero_carries_intro_doc():
    # The intro doc opens the session once; later rounds are terse follow-ups.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=3),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    principals = _principal_events_by_round(g)

    def _user_content(ev):
        return [m for m in ev.call.messages if m.get("role") == "user"][-1]["content"]

    round0 = _user_content(principals[0])
    assert any(m in round0 for m in ("SERVICENOW", "DISPLAY output", "OMEGAMON", "DSNJ031I", "-DIS"))
    # rounds 1 and 2 are follow-ups: no re-pasted intro doc.
    for r in (1, 2):
        content = _user_content(principals[r])
        assert not any(m in content for m in ("SERVICENOW", "DISPLAY output", "OMEGAMON", "-DIS")), (
            f"round {r} unexpectedly re-pasted the intro doc"
        )


# --- Bounded numeric placeholder classes -----------------------------------
#
# Renamed placeholders (`{..._pct}`/`{p99_ms}`/`{status0}`/`{hit_ratio0}`) must
# render values within their semantic bound so the docs read like real
# telemetry (no "273% success rate").


def test_percent_class_field_is_bounded_0_to_100():
    # error_rate_pct in get_service_health is a `_pct` field -> [0, 100].
    tpl = GENERIC_THEME.result_templates["get_service_health"]
    out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, 0), (0, 1))
    import re

    m = re.search(r"error_rate_pct=([0-9]+(?:\.[0-9]+)?)", out)
    assert m, f"error_rate_pct not rendered: {out!r}"
    val = float(m.group(1))
    assert 0.0 <= val <= 100.0, f"percent field out of [0,100]: {val}"


def test_db2_hit_ratio_is_at_most_100():
    # Every hit_ratio in the get_bp_stats table (a ratio) must be <= 100.
    db2 = load_theme("db2_latency_incident")
    tpl = db2.result_templates["get_bp_stats"]
    out = _render_theme_template(db2, tpl, session_seed(42, 5), (0, 4))
    import re

    rows = re.findall(r"\|\s*\d{1,2}:\d{2}:\d{2}\s*\|\s*\d+\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", out)
    assert rows, f"no hit_ratio table rows parsed from: {out!r}"
    for r in rows:
        assert 0.0 <= float(r) <= 100.0, f"hit_ratio out of [0,100]: {r} in {out!r}"


def test_latency_ms_class_field_within_bound():
    # p50_ms / p99_ms in get_service_health are ms-class -> [1, 2000].
    tpl = GENERIC_THEME.result_templates["get_service_health"]
    out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, 0), (0, 1))
    import re

    for key in ("p50_ms", "p99_ms"):
        m = re.search(rf"\b{key}=(\d+)", out)
        assert m, f"{key} not rendered: {out!r}"
        val = int(m.group(1))
        assert 1 <= val <= 2000, f"{key} out of ms bound [1,2000]: {val}"


def test_status_code_class_is_realistic():
    # status0 in run_synthetic_probe must be a plausible HTTP status.
    tpl = GENERIC_THEME.result_templates["run_synthetic_probe"]
    import re

    allowed = {200, 301, 400, 404, 429, 500, 502, 503, 504}
    seen = set()
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m = re.search(r"status=(\d+)", out)
        assert m, f"status not rendered: {out!r}"
        code = int(m.group(1))
        assert code in allowed, f"implausible status code {code}"
        seen.add(code)
    # 200 should dominate the weighted set (sanity: it appears at least once).
    assert 200 in seen, "weighted-common 200 never drawn across 30 seeds"


@pytest.mark.parametrize(
    "template, seed, path",
    [
        # bounded classes (get_service_health)
        (GENERIC_THEME.result_templates["get_service_health"], 3, (0, 1)),
        # in_use <= max (search_logs)
        (GENERIC_THEME.result_templates["search_logs"], 9, (0, 1)),
        # error_pct class (get_service_health, different seed)
        (GENERIC_THEME.result_templates["get_service_health"], 11, (0, 1)),
        # numeric invariants: percentile-sort + heap clamp + in_use/max (literal)
        (
            "p50={p50} p99={p99} heap_used0={heap_used0} heap_max0={heap_max0} in_use0={in_use0} max0={max0}",
            8,
            (0, 1),
        ),
    ],
    ids=["bounded_classes", "in_use_le_max", "error_pct_class", "numeric_invariants"],
)
def test_render_is_deterministic(template, seed, path):
    # Same (theme, seed, path) -> byte-identical render.
    a = _render_theme_template(GENERIC_THEME, template, session_seed(42, seed), path)
    b = _render_theme_template(GENERIC_THEME, template, session_seed(42, seed), path)
    assert a == b, "render not deterministic"


def test_no_bounded_value_exceeds_100_where_percent_signalled():
    # Sweep both themes' templates: wherever the literal text carries a `%` or
    # `hit_ratio`/`_pct` label immediately before a rendered number, that number
    # must be <= 100. Guards the "273% success rate" giveaway across all docs.
    import re

    db2 = load_theme("db2_latency_incident")
    pat = re.compile(r"(?:hit_ratio|_pct|_ratio)\s*[=|]?\s*([0-9]+(?:\.[0-9]+)?)")
    pct_suffix = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
    for theme in (GENERIC_THEME, db2):
        templates = (
            list(theme.result_templates.values()) + list(theme.intro_doc_templates) + list(theme.filler_templates)
        )
        for ti, tpl in enumerate(templates):
            out = _render_theme_template(theme, tpl, session_seed(42, 0), (0, ti))
            for m in pat.finditer(out):
                assert float(m.group(1)) <= 100.0, f"{theme.name} tpl#{ti}: labelled ratio >100: {m.group(0)!r} in {out!r}"
            for m in pct_suffix.finditer(out):
                assert float(m.group(1)) <= 100.0, f"{theme.name} tpl#{ti}: value% >100: {m.group(0)!r} in {out!r}"


# --- Coherence gap 1: intro-doc primary entity == objective primary entity --
#
# The round-0 principal turn = intro_doc + objective. Both must reference the
# SAME primary subject (a live model flagged a doc about `checkout-api` paired
# with a task about `cart-service`). The renderer pins service/db_instance +
# symptom once per round and feeds it to both renders.


def _round0_user_content(g):
    """The round-0 root principal's user-turn content (intro doc + objective)."""
    root_id = g.root_event_ids[0]
    ev = g.events[root_id]
    return [m for m in ev.call.messages if m.get("role") == "user"][-1]["content"]


@pytest.mark.parametrize(
    "theme_name, primary_category",
    [
        # generic: intro doc + objective must name the SAME `service`.
        ("generic", "service"),
        # db2: intro doc + objective must name the SAME `db_instance`.
        ("db2_latency_incident", "db_instance"),
    ],
    ids=["generic_service", "db2_db_instance"],
)
def test_intro_doc_primary_matches_objective(theme_name, primary_category):
    # The round-0 principal turn (intro doc + objective) must name exactly ONE
    # value of the theme's primary category, plus exactly one symptom.
    theme = GENERIC_THEME if theme_name == "generic" else load_theme(theme_name)
    cfg = _cfg(
        theme_mix={theme_name: 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
    )
    primaries = theme.entities[primary_category]
    symptoms = theme.entities["symptom"]
    # Sweep several sessions so we exercise different pinned draws.
    for idx in range(8):
        g = build_graph_for_session(cfg, theme, _WordTok(), session_index=idx)
        content = _round0_user_content(g)
        present = [s for s in primaries if s in content]
        # Exactly ONE primary string appears -> doc + task agree on it.
        assert len(set(present)) == 1, (
            f"idx {idx}: round-0 turn names {sorted(set(present))} {primary_category}s, not one: {content!r}"
        )
        present_sym = [s for s in symptoms if s in content]
        assert len(set(present_sym)) == 1, f"idx {idx}: round-0 turn names {sorted(set(present_sym))} symptoms, not one"


def test_pinned_entity_coherence_is_deterministic():
    # Same (config, index) -> byte-identical round-0 turn (pinning is seeded).
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    assert _round0_user_content(g1) == _round0_user_content(g2)


def test_render_theme_template_honors_pinned_entity():
    # A pinned service value overrides the per-field draw for that category.
    tpl = "service={service} symptom={symptom} dep={dep}"
    out = _render_theme_template(
        GENERIC_THEME, tpl, session_seed(42, 0), (0, 1), pinned={"service": "cart-service", "symptom": "request timeouts"}
    )
    assert "service=cart-service" in out
    assert "symptom=request timeouts" in out
    # a non-pinned category still draws normally (from its own pool)
    assert any(f"dep={d}" in out for d in GENERIC_THEME.entities["dep"])


# --- Coherence gap 2: in_use <= max in rendered pool templates --------------


def test_in_use_never_exceeds_max():
    # search_logs renders "in_use={in_use0}/{max0}"; in_use must be <= max.
    import re

    tpl = GENERIC_THEME.result_templates["search_logs"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m = re.search(r"in_use=(\d+)/(\d+)", out)
        assert m, f"in_use/max pair not rendered: {out!r}"
        in_use, mx = int(m.group(1)), int(m.group(2))
        assert in_use <= mx, f"in_use {in_use} > max {mx} in {out!r}"


def test_in_use_le_max_in_intro_doc_and_filler():
    # The same rule holds in the generic intro docs and the pool-acquire filler.
    import re

    templates = list(GENERIC_THEME.intro_doc_templates) + list(GENERIC_THEME.filler_templates)
    for ti, tpl in enumerate(templates):
        if "in_use" not in tpl or "max" not in tpl:
            continue
        for idx in range(15):
            out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, ti))
            for m in re.finditer(r"in_use[ =](\d+)[ /]+(?:idle=\d+ )?max=?(\d+)|in_use[ =](\d+)/(\d+)", out):
                groups = [x for x in m.groups() if x is not None]
                in_use, mx = int(groups[0]), int(groups[1])
                assert in_use <= mx, f"tpl#{ti} idx{idx}: in_use {in_use} > max {mx} in {out!r}"


# --- Coherence gap 3: error-rate fields render LOW --------------------------


def test_error_rate_pct_reads_low():
    # error_rate_pct is an error-rate percent -> low ([0, 15]), not [80, 100].
    import re

    tpl = GENERIC_THEME.result_templates["get_service_health"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m = re.search(r"error_rate_pct=([0-9]+(?:\.[0-9]+)?)", out)
        assert m, f"error_rate_pct not rendered: {out!r}"
        val = float(m.group(1))
        assert 0.0 <= val <= 15.0, f"error rate not low: {val} in {out!r}"


def test_err_rate_and_err_pct_read_low():
    # `err_rate` (generic filler) and `err_pct` (generic intro doc slack thread)
    # are error percentages -> low. A raw `errors`/`err` COUNT is NOT affected.
    import re

    # err_rate in the metric filler line
    metric_filler = next(t for t in GENERIC_THEME.filler_templates if "err_rate=" in t)
    for idx in range(20):
        out = _render_theme_template(GENERIC_THEME, metric_filler, session_seed(42, idx), (0, 2))
        m = re.search(r"err_rate=([0-9]+(?:\.[0-9]+)?)", out)
        assert m and float(m.group(1)) <= 15.0, f"err_rate not low: {out!r}"
    # err_pct in the slack-thread intro doc (value={err_pct}%)
    slack_doc = next(t for t in GENERIC_THEME.intro_doc_templates if "{err_pct}" in t)
    for idx in range(20):
        out = _render_theme_template(GENERIC_THEME, slack_doc, session_seed(42, idx), (0, 3))
        m = re.search(r"value=([0-9]+(?:\.[0-9]+)?)%", out)
        assert m and float(m.group(1)) <= 15.0, f"err_pct not low: {out!r}"


# --- Forced tool-call turns must carry a real max_tokens budget --------------
#
# Regression guard for the RITS 400 bug: forced tool-call events used to ship
# with expected_output_tokens=0, so a real model truncated its tool call mid-JSON
# and leaked chat-template control tokens into `arguments`, which 400s on replay.
# Each forced tool-call event must be sized to tokens(json.dumps(its calls)) +
# TOOL_CALL_MARGIN; plain-answer events keep their sampled output size.


def _forced_and_answer_events(g):
    """Split a graph's events into (forced-tool-call events, plain-answer events)."""
    forced = [ev for ev in g.events.values() if ev.call.expected_output_is_tool_call]
    answers = [ev for ev in g.events.values() if not ev.call.expected_output_is_tool_call]
    return forced, answers


def test_tool_call_max_tokens_helper():
    import json as _json

    tok = _WordTok()
    assert _tool_call_max_tokens(tok, []) == TOOL_CALL_MARGIN  # no calls -> margin floor
    calls = [{"id": "c0", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}]
    expected = tok.count_tokens(_json.dumps(calls)) + TOOL_CALL_MARGIN
    assert _tool_call_max_tokens(tok, calls) == expected
    assert _tool_call_max_tokens(tok, calls) > TOOL_CALL_MARGIN  # calls add on top of margin


def test_forced_tool_events_are_sized_not_zero():
    # Across every shape that forces tool calls, each forced event's
    # expected_output_tokens is >= TOOL_CALL_MARGIN (never the old 0), so the
    # replay model has room to emit the whole tool call.
    shapes = {
        "tool_loop": _cfg(tool_turns_per_loop=Distribution(type="fixed", mean=3), fanout_probability=0.0),
        "parallel": _cfg(
            tool_turns_per_loop=Distribution(type="fixed", mean=2),
            parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
            fanout_probability=0.0,
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
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
        forced, _ = _forced_and_answer_events(g)
        assert forced, f"{name}: expected at least one forced tool-call event"
        for ev in forced:
            assert ev.call.expected_output_tokens >= TOOL_CALL_MARGIN, (
                f"{name} {ev.event_id}: forced tool-call sized {ev.call.expected_output_tokens} < margin {TOOL_CALL_MARGIN}"
            )


def test_forced_tool_events_sized_from_their_own_calls():
    # The budget equals tokens(json.dumps(the calls this event outputs)) + margin.
    # Reconstruct each forced event's calls from its OWN or its successor's stored
    # tool_calls and check the size matches.
    import json as _json

    tok = _WordTok()
    cfg = _cfg(tool_turns_per_loop=Distribution(type="fixed", mean=3), fanout_probability=0.0)
    g = build_graph_for_session(cfg, GENERIC_THEME, tok, session_index=0)
    # The tool_calls an event OUTPUTS appear in the SUCCESSOR event's messages
    # (as the reconstructed assistant tool_call). Walk the linear chain by id.
    ordered = sorted(
        g.events.values(),
        key=lambda e: (0 if e.event_id.endswith(":principal") else 1, e.event_id),
    )
    for i, ev in enumerate(ordered):
        if not ev.call.expected_output_is_tool_call:
            continue
        # find the calls this event outputs = the LAST assistant tool_calls group
        # in the NEXT event's messages
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            calls = None
            for m in nxt.call.messages:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    calls = m["tool_calls"]
            if calls:
                expected = tok.count_tokens(_json.dumps(calls)) + TOOL_CALL_MARGIN
                assert ev.call.expected_output_tokens == expected, (
                    f"{ev.event_id}: sized {ev.call.expected_output_tokens} != {expected} (json+{TOOL_CALL_MARGIN})"
                )


def test_answer_events_keep_output_tokens_not_tool_budget():
    # Plain-answer terminal events keep the sampled output_tokens_per_turn, NOT
    # the tool-call sizing. With output_tokens_per_turn fixed at 40, the terminal
    # answer event of a tool loop is 40 (its output IS the plain answer).
    cfg = _cfg(
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
        output_tokens_per_turn=Distribution(type="fixed", mean=40),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _, answers = _forced_and_answer_events(g)
    assert answers, "expected a plain-answer terminal event"
    for ev in answers:
        assert ev.call.expected_output_tokens == 40, (
            f"{ev.event_id}: answer sized {ev.call.expected_output_tokens} != sampled 40"
        )


def test_forced_tool_sizing_is_deterministic():
    cfg = _cfg(
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=5)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=5)
    for eid in g1.events:
        assert g1.events[eid].call.expected_output_tokens == g2.events[eid].call.expected_output_tokens


# --- Per-tool parameter schemas + schema-conforming emitted arguments -------
#
# Every advertised tool must carry a REAL (non-empty) parameter schema with
# required fields; every FORCED tool call must emit arguments that parse as JSON
# and contain every required property of the called tool's advertised schema. A
# parameterless forced tool_choice makes some models emit empty `{}` args and
# then fail to stop, leaking chat-template tokens -> the tool call 400s on
# replay; these tests guard against that class of regression.

import json as _json  # noqa: E402


def _defs_by_name(ev):
    """Map advertised tool name -> its tool_definition dict for an event."""
    return {td["name"]: td for td in (ev.call.tool_definitions or []) if "name" in td}


def _emitted_tool_calls(ev):
    """Yield (call_name, parsed_args_dict) for every stored assistant tool_call."""
    for m in ev.call.messages:
        for tc in m.get("tool_calls", []) or []:
            yield tc["function"]["name"], _json.loads(tc["function"]["arguments"])


def _cfg_all_tools(theme, **kw):
    """A config that advertises MORE tool defs than the theme has base tools, so
    every base tool AND at least one suffixed duplicate appears in the catalog.
    Uses a multi-turn, multi-parallel tool loop so many distinct tools are
    actually called."""
    n = len(theme.tool_names) + 2  # forces >=1 suffixed duplicate
    base = dict(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_definitions_per_agent=Distribution(type="fixed", mean=n),
        tool_turns_per_loop=Distribution(type="fixed", mean=n),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    base.update(kw)
    return _cfg(**base)


def test_every_advertised_tool_def_has_nonempty_params_and_required():
    # Build a session for each theme with a catalog LARGER than its tool list
    # (so all base tools + a suffixed duplicate appear); every advertised tool
    # def must have non-empty properties AND a non-empty required list.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        cfg = _cfg_all_tools(theme, theme_mix={theme.name: 1.0})
        g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
        seen_names = set()
        for ev in g.events.values():
            for td in ev.call.tool_definitions or []:
                params = td["function"]["parameters"]
                assert params.get("type") == "object", f"{theme.name} {td['name']}: params not an object"
                assert params.get("properties"), f"{theme.name} {td['name']}: empty properties"
                assert params.get("required"), f"{theme.name} {td['name']}: empty required list"
                # every required name must exist in properties
                for req in params["required"]:
                    assert req in params["properties"], f"{theme.name} {td['name']}: required {req} not in properties"
                seen_names.add(td["name"])
        # a suffixed duplicate was advertised (catalog > base tool count)
        assert any("_" in n and n.rsplit("_", 1)[-1].isdigit() for n in seen_names), (
            f"{theme.name}: no suffixed-duplicate tool advertised; seen {sorted(seen_names)}"
        )


def test_suffixed_duplicate_reuses_base_param_schema():
    # A synthetic suffixed duplicate (get_bp_stats_10) must reuse its base
    # tool's parameter schema, not the generic fallback.
    theme = load_theme("db2_latency_incident")
    n = len(theme.tool_names) + 2
    defs = _tool_definitions(theme, n)
    base0 = theme.tool_names[0]
    base_params = theme.tool_parameters[base0]
    dup = next(td for td in defs if td["name"].startswith(base0 + "_"))
    assert dup["function"]["parameters"] == base_params, "suffixed duplicate did not reuse base param schema"


def test_emitted_tool_call_args_conform_to_advertised_schema():
    # Every emitted tool-call `arguments` string parses as JSON and contains
    # EVERY required property of the called tool's advertised schema. Cross-
    # reference the call name to its def in the SAME event.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        cfg = _cfg_all_tools(theme, theme_mix={theme.name: 1.0})
        g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
        checked = 0
        for ev in g.events.values():
            defs = _defs_by_name(ev)
            for call_name, args in _emitted_tool_calls(ev):
                assert call_name in defs, f"{theme.name}: call {call_name} not advertised in its event"
                required = defs[call_name]["function"]["parameters"]["required"]
                assert isinstance(args, dict), f"{theme.name} {call_name}: args not a JSON object: {args!r}"
                for req in required:
                    assert req in args, f"{theme.name} {call_name}: emitted args missing required {req!r}: {args}"
                checked += 1
        assert checked > 0, f"{theme.name}: no emitted tool calls were checked"


def test_multi_required_param_tool_emits_all_required_fields():
    # Force a catalog whose FIRST tool is a KNOWN multi-required-param tool and
    # confirm an emitted call carries all of its required fields.
    # db2 get_bp_stats requires (db_instance, bufferpool); generic
    # get_service_health has 1 req, but query_metrics (3 req) is reachable via a
    # larger parallel/loop catalog. Use db2's get_bp_stats (tool_names[0]).
    theme = load_theme("db2_latency_incident")
    multi = "get_bp_stats"  # theme.tool_names[0]; required = [db_instance, bufferpool]
    required = theme.tool_parameters[multi]["required"]
    assert len(required) >= 2, "test premise: get_bp_stats is multi-required-param"
    cfg = _cfg(
        theme_mix={theme.name: 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_definitions_per_agent=Distribution(type="fixed", mean=1),  # only get_bp_stats
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    hits = 0
    for ev in g.events.values():
        for call_name, args in _emitted_tool_calls(ev):
            if call_name == multi:
                for req in required:
                    assert req in args, f"multi-param call missing {req!r}: {args}"
                hits += 1
    assert hits > 0, "the multi-required-param tool was never called"


def test_generic_multi_param_call_carries_all_required():
    # Generic query_metrics has 3 required (metric, service, window). Advertise
    # enough tools that query_metrics (tool_names[1]) is in the catalog and is
    # called during the loop.
    theme = GENERIC_THEME
    multi = "query_metrics"
    required = theme.tool_parameters[multi]["required"]
    assert len(required) == 3
    cfg = _cfg_all_tools(theme, theme_mix={"generic": 1.0})
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    hits = 0
    for ev in g.events.values():
        for call_name, args in _emitted_tool_calls(ev):
            if call_name == multi:
                for req in required:
                    assert req in args, f"query_metrics missing {req!r}: {args}"
                hits += 1
    assert hits > 0, "query_metrics (multi-param) was never called"


def test_emitted_args_are_deterministic():
    # Same (config, index) -> byte-identical emitted argument strings.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        cfg = _cfg_all_tools(theme, theme_mix={theme.name: 1.0})
        g1 = build_graph_for_session(cfg, theme, _WordTok(), session_index=6)
        g2 = build_graph_for_session(cfg, theme, _WordTok(), session_index=6)

        def _all_arg_strings(g):
            out = []
            for eid in g.events:
                for m in g.events[eid].call.messages:
                    for tc in m.get("tool_calls", []) or []:
                        out.append((eid, tc["function"]["name"], tc["function"]["arguments"]))
            return out

        assert _all_arg_strings(g1) == _all_arg_strings(g2), f"{theme.name}: emitted args not deterministic"


def test_entity_named_param_threads_pinned_subject():
    # A property NAMED like an entity category (`service`/`db_instance`) is
    # filled with a real value from that theme's pool (coherence). Sweep several
    # sessions and confirm the emitted `service` arg is always a real service.
    services = set(GENERIC_THEME.entities["service"])
    cfg = _cfg_all_tools(GENERIC_THEME, theme_mix={"generic": 1.0})
    for idx in range(4):
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        for ev in g.events.values():
            for _, args in _emitted_tool_calls(ev):
                if "service" in args:
                    assert args["service"] in services, f"service arg not a real service: {args['service']!r}"


def test_both_themes_validate_with_tool_parameters():
    # Both themes still load/validate and now carry per-tool parameter schemas,
    # each a well-formed JSON-Schema object with a required list whose names all
    # exist in properties.
    db2 = load_theme("db2_latency_incident")
    for theme in (GENERIC_THEME, db2):
        assert theme.tool_parameters, f"{theme.name}: expected per-tool parameter schemas"
        for base in theme.tool_names:
            spec = theme.tool_parameters.get(base)
            assert spec is not None, f"{theme.name}: tool {base} missing a parameter schema"
            assert spec["type"] == "object"
            assert spec["properties"], f"{theme.name} {base}: empty properties"
            assert spec.get("required"), f"{theme.name} {base}: empty required list"
            for req in spec["required"]:
                assert req in spec["properties"], f"{theme.name} {base}: required {req} not in properties"


def test_fallback_tool_params_applies_for_theme_without_schemas():
    # A theme with tools but NO tool_parameters must emit the generic {query}
    # fallback schema, and forced calls must emit `query`.
    bare = Theme(
        name="bare_no_params",
        verbs=["Do"],
        entities={"widget": ["alpha", "beta"]},
        tool_names=["do_thing", "check_thing"],
        result_templates={"default": "result {n0}"},
        objective_template="{verb} the {widget}.",
    )
    assert bare.tool_parameters == {}
    defs = _tool_definitions(bare, 3)
    for td in defs:
        assert td["function"]["parameters"] == _FALLBACK_TOOL_PARAMS, "expected the {query} fallback schema"
        assert td["function"]["parameters"]["required"] == ["query"]

    cfg = _cfg(
        theme_mix={"generic": 1.0},  # theme_mix is unused; we pass `bare` directly
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_definitions_per_agent=Distribution(type="fixed", mean=2),
        tool_turns_per_loop=Distribution(type="fixed", mean=2),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, bare, _WordTok(), session_index=0)
    hits = 0
    for ev in g.events.values():
        for _, args in _emitted_tool_calls(ev):
            assert "query" in args, f"fallback call missing `query`: {args}"
            hits += 1
    assert hits > 0, "no tool calls emitted for the fallback-schema theme"


# --- ignore_eos must be False for forced tool-call turns (RITS 400 fix) -------
#
# The load default is ignore_eos=True (to make plain-text turns generate exactly
# N tokens). For a FORCED tool call that is wrong: with EOS ignored the model
# emits the call then keeps generating, spilling chat-template control tokens
# into `arguments` until max_tokens -> malformed JSON -> 400 on the replayed
# turn. to_request_body must force ignore_eos=False for every forced tool-call
# turn, regardless of override_tool_call_max_tokens.


def test_forced_tool_call_forces_ignore_eos_false():
    import asyncio
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    tool_defs = [
        {
            "name": "get_service_health",
            "type": "function",
            "function": {
                "name": "get_service_health",
                "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            },
        }
    ]

    def _mk(is_tool_call, override):
        return SessionChatCompletionAPIData(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            tool_definitions=tool_defs,
            event_id="s:e",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            predecessor_event_ids=[],
            input_segments=[],
            original_messages=[{"role": "user", "content": "hi"}],
            expected_output_is_tool_call=is_tool_call,
            expected_output_tool_names=["get_service_health"],
            override_tool_call_max_tokens=override,
        )

    # Caller passes ignore_eos=True (the load default); a FORCED tool call must
    # override it to False even when override_tool_call_max_tokens is False.
    forced = _mk(True, False)
    payload = asyncio.run(
        forced.to_request_body(effective_model_name="m", max_tokens=64, ignore_eos=True, streaming=False)
    )
    assert payload["ignore_eos"] is False, "forced tool call must send ignore_eos=False"

    # A plain-text turn that STILL advertises tools (this _mk passes tool_defs)
    # is the terminal/answer turn of a tool loop: it also gets ignore_eos=False
    # + tool_choice=none so it can't emit a dangling structured call or spill
    # template tokens. (A plain-text turn with NO tools keeps the caller
    # ignore_eos -- see test_plain_text_turn_without_tools_keeps_defaults.)
    plain = _mk(False, False)
    p2 = asyncio.run(
        plain.to_request_body(effective_model_name="m", max_tokens=64, ignore_eos=True, streaming=False)
    )
    assert p2["ignore_eos"] is False, "plain-text-with-tools turn must stop cleanly (ignore_eos=False)"
    assert p2["tool_choice"] == "none", "plain-text-with-tools turn must forbid a structured tool call"


# --- Tool result echoes the call's arguments (coherence) --------------------
#
# A real tool answers about the entity it was called with. So a result template
# placeholder that matches a call-argument key (e.g. `{service}`) must resolve to
# the value THIS call passed, not an independent draw. Regression guard for the
# observed mismatch (call service=session-gateway, result service=inventory-svc).


def test_tool_result_echoes_call_service():
    import json as _json
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        tool_definitions_per_agent=Distribution(type="fixed", mean=8),
        fanout_probability=0.0,
    )
    checked = 0
    for idx in range(4):
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        for ev in g.events.values():
            # map tool_call_id -> the `service` the call passed
            call_service = {}
            for m in ev.call.messages:
                for tc in m.get("tool_calls", []) or []:
                    args = _json.loads(tc["function"]["arguments"])
                    if "service" in args:
                        call_service[tc["id"]] = args["service"]
            for m in ev.call.messages:
                if m.get("role") == "tool" and m["tool_call_id"] in call_service:
                    want = call_service[m["tool_call_id"]]
                    rm = re.search(r"service=([a-z0-9-]+)", m["content"])
                    if rm:  # only when the result template names a service
                        assert rm.group(1) == want, (
                            f"result service={rm.group(1)!r} != call service={want!r} (must echo the call)"
                        )
                        checked += 1
    assert checked > 0, "no service-bearing tool result found to check"


def test_plain_text_turn_with_tools_forbids_tool_call_and_stops():
    # A plain-text answer turn that still advertises a tool catalog must send
    # tool_choice="none" (no structured tool call -> nothing to dangle into the
    # next round) and ignore_eos=False (stop cleanly, no <|im_end|> spill).
    import asyncio
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    tool_defs = [
        {
            "name": "get_service_health",
            "type": "function",
            "function": {
                "name": "get_service_health",
                "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            },
        }
    ]
    ev = SessionChatCompletionAPIData(
        messages=[{"role": "user", "content": "answer now"}],
        max_tokens=80,
        tool_definitions=tool_defs,
        event_id="s:e",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
        input_segments=[],
        original_messages=[{"role": "user", "content": "answer now"}],
        expected_output_is_tool_call=False,  # plain-text answer turn
        expected_output_tool_names=[],
    )
    payload = asyncio.run(
        ev.to_request_body(effective_model_name="m", max_tokens=80, ignore_eos=True, streaming=False)
    )
    assert payload["tool_choice"] == "none", "plain-text turn with tools must forbid a structured tool call"
    assert payload["ignore_eos"] is False, "plain-text turn with tools must stop cleanly (ignore_eos=False)"


def test_plain_text_turn_without_tools_keeps_defaults():
    # A plain-text turn with NO tool catalog is untouched: no tool_choice, keeps
    # the caller's ignore_eos (so ordinary text turns still generate to length).
    import asyncio
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    ev = SessionChatCompletionAPIData(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=80,
        tool_definitions=None,
        event_id="s:e",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
        input_segments=[],
        original_messages=[{"role": "user", "content": "hi"}],
        expected_output_is_tool_call=False,
        expected_output_tool_names=[],
    )
    payload = asyncio.run(
        ev.to_request_body(effective_model_name="m", max_tokens=80, ignore_eos=True, streaming=False)
    )
    assert payload.get("tool_choice") is None, "plain text turn without tools must not set tool_choice"
    assert payload["ignore_eos"] is True, "plain text turn without tools keeps caller ignore_eos"


def test_fanout_children_pinned_to_parent_entity():
    # Fan-out coherence: every dispatched child's objective names the SAME primary
    # subject entity as the orchestrator (the fan-out is ONE investigation). The
    # verb may differ (children take different angles), but the service/db_instance
    # must match the parent's pinned subject.
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=1,
        tool_turns_per_loop=Distribution(type="fixed", mean=0),
        max_events_per_session=512,
    )

    def _service(text):
        m = re.search(r"the ([a-z]+-[a-z]+) incident", text) or re.search(r"on ([a-z]+-[a-z]+)", text)
        return m.group(1) if m else None

    for idx in range(4):
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        # orchestrator objective (root principal, not a sub)
        orch = None
        for eid, ev in g.events.items():
            if eid.endswith(":principal") and ":sub" not in eid:
                users = [m["content"] for m in ev.call.messages if m["role"] == "user"]
                if users:
                    orch = _service(users[-1])
        # child dispatch objectives
        kids = []
        for eid, ev in g.events.items():
            if ":disp" in eid:
                users = [m["content"] for m in ev.call.messages if m["role"] == "user"]
                if users:
                    kids.append(_service(users[-1]))
        assert orch is not None, f"session {idx}: no orchestrator service parsed"
        assert kids, f"session {idx}: no child dispatch objectives"
        for k in kids:
            assert k == orch, f"session {idx}: child service {k!r} != orchestrator {orch!r} (fan-out not coherent)"


def test_subagent_terminal_ends_with_report_directive():
    # A spawned sub-agent's TERMINAL turn must END with the summarize-report nudge
    # (recency -> the child produces a PROSE report, not tool-call text). The nudge
    # is the LAST message and is a `user` message; cursor math stays exact
    # (sum(seg.message_count) == len(messages)). NON-terminal child turns, the
    # root/orchestrator, and the merge must NOT end with it.
    from inference_perf.datagen.synthetic_agentic import SUBAGENT_REPORT_DIRECTIVE

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=1,
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    frag = SUBAGENT_REPORT_DIRECTIVE

    def ends_with_nudge(ev):
        msgs = ev.call.messages
        return bool(msgs) and msgs[-1].get("role") == "user" and frag in str(msgs[-1].get("content", ""))

    saw_child_terminal = False
    for eid, ev in g.events.items():
        is_child = ":sub" in eid
        is_terminal = not ev.call.expected_output_is_tool_call  # answer turn, not a tool-call turn
        if is_child and is_terminal:
            saw_child_terminal = True
            assert ends_with_nudge(ev), f"child terminal {eid} must END with the report nudge"
            # cursor math must remain exact after the appended message
            if ev.call.input_segments:
                segsum = sum(s.message_count for s in ev.call.input_segments)
                assert segsum == len(ev.call.messages), f"{eid}: segment sum {segsum} != {len(ev.call.messages)}"
        else:
            # everything else (root terminals, non-terminal child tool-turns, merge) must NOT end with it
            assert not ends_with_nudge(ev), f"{eid} should NOT end with the sub-agent report nudge"
    assert saw_child_terminal, "expected >=1 sub-agent terminal turn"


def test_nonroot_merge_ends_with_report_directive():
    # In a recursive (depth-2) tree, a SPAWNING sub-agent's terminal is its MERGE
    # event (it folds in grandchildren, then reports up to its parent). That merge
    # must END with the report nudge (prose report at every non-leaf level); the
    # ROOT merge must NOT (its output is the orchestrator's final answer). Cursor
    # math must stay exact after the appended message.
    from inference_perf.datagen.synthetic_agentic import SUBAGENT_REPORT_DIRECTIVE

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=2,  # children spawn grandchildren -> children's terminal is a merge
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    frag = SUBAGENT_REPORT_DIRECTIVE
    saw_nonroot = saw_root = False
    for eid, ev in g.events.items():
        if not eid.endswith(":merge"):
            continue
        msgs = ev.call.messages
        ends = msgs[-1].get("role") == "user" and frag in str(msgs[-1].get("content", ""))
        # cursor math exact
        segsum = sum(s.message_count for s in ev.call.input_segments)
        assert segsum == len(msgs), f"{eid}: merge segment sum {segsum} != {len(msgs)}"
        # root merge: agent prefix is the bare round root (no ':sub' before ':dN:merge')
        is_root_merge = ":sub" not in eid.rsplit(":d", 1)[0]
        if is_root_merge:
            saw_root = True
            assert not ends, f"root merge {eid} should NOT end with the report nudge"
        else:
            saw_nonroot = True
            assert ends, f"non-root (child) merge {eid} must END with the report nudge"
    assert saw_nonroot and saw_root, "expected both a root merge and >=1 non-root merge"


def test_report_directive_deterministic():
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=1,
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        max_events_per_session=512,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


# --- Realism items 1+3: research_rag theme + richer result shapes -----------
#
# Item 1: a NEW research/retrieval theme (research_rag) that loads, validates,
# and builds coherent sessions with realistic retrieval output shapes.
# Item 3: existing themes patched with stack-trace / JSON-object / not-found
# result shapes so the corpus is not all happy-path tabular payloads.

import json as _json_rr  # noqa: E402

# result-template keys whose RENDERED output is intentionally a JSON blob and so
# legitimately contains literal `{`/`}` (they must PARSE as JSON, not leak
# unfilled placeholders). Every other template must be brace-free after render.
_JSON_RESULT_KEYS = {"generic": {"get_config_snapshot"}, "research_rag": {"search_json"}}


def _assert_template_render_clean(theme, theme_key, seed_idx=0):
    """Render every result/intro/filler template of `theme` and assert no
    single-brace placeholder leaked. JSON-shaped result templates are validated
    by json.loads instead (their braces are literal, doubled in the source)."""
    json_keys = _JSON_RESULT_KEYS.get(theme_key, set())
    for i, (k, tpl) in enumerate(theme.result_templates.items()):
        out = _render_theme_template(theme, tpl, session_seed(42, seed_idx), (0, i))
        if k in json_keys:
            _json_rr.loads(out)  # must be valid JSON (doubled braces resolved)
        else:
            assert "{" not in out and "}" not in out, f"{theme_key} result[{k}] brace leak: {out!r}"
    for i, tpl in enumerate(theme.intro_doc_templates):
        out = _render_theme_template(theme, tpl, session_seed(42, seed_idx), (1, i))
        assert "{" not in out and "}" not in out, f"{theme_key} intro[{i}] brace leak: {out!r}"
    for i, tpl in enumerate(theme.filler_templates):
        out = _render_theme_template(theme, tpl, session_seed(42, seed_idx), (2, i))
        assert "{" not in out and "}" not in out, f"{theme_key} filler[{i}] brace leak: {out!r}"


def test_research_rag_loads_and_validates():
    t = load_theme("research_rag")
    assert isinstance(t, Theme)
    assert t.name == "research_rag"
    assert t.verbs and t.tool_names
    assert "default" in t.result_templates
    # 6-9 retrieval tools, each with a description + a well-formed param schema.
    assert 6 <= len(t.tool_names) <= 9
    for name in t.tool_names:
        assert name in t.tool_descriptions, f"{name} missing description"
        spec = t.tool_parameters[name]
        assert spec["type"] == "object" and spec["properties"] and spec["required"]
        for req in spec["required"]:
            assert req in spec["properties"], f"{name} required {req} not in properties"
    # the expected retrieval toolbox is present
    assert {"web_search", "fetch_url", "retrieve_docs", "read_file", "grep"}.issubset(set(t.tool_names))


def test_research_rag_templates_render_without_leak():
    t = load_theme("research_rag")
    _assert_template_render_clean(t, "research_rag")


def test_research_rag_session_builds_valid_and_deterministic():
    # Build a session with theme_mix {research_rag:1.0}: every emitted tool-call
    # arg parses as JSON, every tool result is brace-clean, deterministic per
    # (config, index).
    t = load_theme("research_rag")
    cfg = _cfg(
        theme_mix={"research_rag": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=2),
        tool_turns_per_loop=Distribution(type="fixed", mean=3),
        tool_definitions_per_agent=Distribution(type="fixed", mean=len(t.tool_names)),
        parallel_tool_calls_per_turn=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    for idx in range(3):
        g = build_graph_for_session(cfg, t, _WordTok(), session_index=idx)
        assert g.events
        for ev in g.events.values():
            for m in ev.call.messages:
                for tc in m.get("tool_calls", []) or []:
                    args = tc["function"]["arguments"]
                    assert isinstance(args, str)
                    _json_rr.loads(args)  # valid JSON
                if m.get("role") == "tool":
                    c = m["content"]
                    assert "{" not in c and "}" not in c, f"research_rag result brace leak: {c!r}"
    # determinism per (config, index)
    g1 = build_graph_for_session(cfg, t, _WordTok(), session_index=1)
    g2 = build_graph_for_session(cfg, t, _WordTok(), session_index=1)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


def test_research_rag_search_hits_and_json_shapes():
    # The web_search shape is a ranked hit list; retrieve_docs carries relevance
    # scores <= 100; the JSON shape parses and carries the expected keys.
    import re

    t = load_theme("research_rag")
    hits = _render_theme_template(t, t.result_templates["web_search"], session_seed(42, 4), (0, 0))
    assert "1. " in hits and "2. " in hits and "https://" in hits, f"web_search not a ranked hit list: {hits!r}"

    passages = _render_theme_template(t, t.result_templates["retrieve_docs"], session_seed(42, 4), (0, 1))
    scores = re.findall(r"score=([0-9]+(?:\.[0-9]+)?)", passages)
    assert scores, f"retrieve_docs carried no relevance score: {passages!r}"
    for s in scores:
        assert float(s) <= 100.0, f"relevance score >100: {s}"

    blob = _render_theme_template(t, t.result_templates["search_json"], session_seed(42, 4), (0, 2))
    obj = _json_rr.loads(blob)
    assert {"query", "results", "total_hits"}.issubset(obj.keys()), f"JSON missing keys: {obj}"

    empty = _render_theme_template(t, t.result_templates["empty_search"], session_seed(42, 4), (0, 3))
    assert "no results" in empty, f"empty_search missing not-found marker: {empty!r}"


def test_generic_stack_trace_and_json_shapes_render():
    # GENERIC's new stack-trace and JSON-object result shapes render with no
    # placeholder leak and carry their expected markers.
    st = _render_theme_template(
        GENERIC_THEME, GENERIC_THEME.result_templates["get_exception_trace"], session_seed(42, 0), (0, 9)
    )
    assert "{" not in st and "}" not in st, f"stack-trace brace leak: {st!r}"
    assert "Traceback" in st and "PoolTimeout" in st, f"stack trace markers absent: {st!r}"

    js = _render_theme_template(
        GENERIC_THEME, GENERIC_THEME.result_templates["get_config_snapshot"], session_seed(42, 0), (0, 10)
    )
    obj = _json_rr.loads(js)  # doubled braces resolved to a valid JSON blob
    assert "service" in obj and "flags" in obj and "limits" in obj, f"config JSON keys absent: {obj}"
    # nested numeric limit stays bounded (named max0/ms0 classes)
    assert isinstance(obj["flags"], dict) and isinstance(obj["limits"], dict)


def test_db2_not_found_error_shape_renders():
    # db2's new get_message_log returns a not-found / connection-error payload.
    t = load_theme("db2_latency_incident")
    out = _render_theme_template(t, t.result_templates["get_message_log"], session_seed(42, 0), (0, 11))
    assert "{" not in out and "}" not in out, f"not-found brace leak: {out!r}"
    assert "ERROR" in out and "no messages" in out, f"not-found markers absent: {out!r}"


def test_new_result_shapes_are_deterministic():
    # Same (theme, seed, path) -> byte-identical render for each new shape.
    cases = [
        (GENERIC_THEME, "get_exception_trace", (0, 9)),
        (GENERIC_THEME, "get_config_snapshot", (0, 10)),
        (load_theme("db2_latency_incident"), "get_message_log", (0, 11)),
        (load_theme("research_rag"), "search_json", (0, 2)),
        (load_theme("research_rag"), "web_search", (0, 0)),
    ]
    for theme, key, path in cases:
        a = _render_theme_template(theme, theme.result_templates[key], session_seed(42, 3), path)
        b = _render_theme_template(theme, theme.result_templates[key], session_seed(42, 3), path)
        assert a == b, f"{theme.name}.{key} not deterministic"


def test_all_three_bundled_themes_still_load_and_validate():
    # Both existing themes plus the new one load, validate, and keep their
    # required invariants (non-empty verbs/tools, a 'default' result template).
    for name in ("db2_latency_incident", "research_rag"):
        t = load_theme(name)
        assert t.verbs and t.tool_names and "default" in t.result_templates
    assert GENERIC_THEME.verbs and GENERIC_THEME.tool_names and "default" in GENERIC_THEME.result_templates
    # db2's rich get_bp_stats table header is preserved (a test elsewhere asserts
    # it live; guard the source template here too).
    db2 = load_theme("db2_latency_incident")
    assert "| time | bp | hit_ratio |" in db2.result_templates["get_bp_stats"]


# --- Item-4 fix 1: numeric invariants in rendered results -------------------
#
# Percentiles obey p50 <= p90 <= p95 <= p99 within a shared suffix (bare, `_ms`,
# and indexed forms), and heap_used <= heap_max, in ADDITION to the existing
# in_use <= max. The renderer draws each field independently, then a paired-field
# pass repairs the ordering deterministically over the drawn values.


def test_percentile_ordering_p50_le_p99_bare_ms_and_indexed():
    import re

    # (a) bare `_ms` siblings in a real template (get_service_health).
    tpl = GENERIC_THEME.result_templates["get_service_health"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        p50 = int(re.search(r"\bp50_ms=(\d+)", out).group(1))
        p99 = int(re.search(r"\bp99_ms=(\d+)", out).group(1))
        assert p50 <= p99, f"p50_ms {p50} > p99_ms {p99} in {out!r}"

    # (b) a synthetic template with all four bare percentiles must come out sorted.
    p4 = "p50={p50} p90={p90} p95={p95} p99={p99}"
    for idx in range(20):
        out = _render_theme_template(GENERIC_THEME, p4, session_seed(42, idx), (0, 1))
        vals = [int(x) for x in re.findall(r"=(\d+)", out)]
        assert vals == sorted(vals), f"p50<=p90<=p95<=p99 violated: {vals} in {out!r}"

    # (c) indexed forms: `p99_0`/`p50_0` (query_metrics rows) AND `p50_ms0`/`p99_ms0`.
    qm = GENERIC_THEME.result_templates["query_metrics"]
    for idx in range(20):
        out = _render_theme_template(GENERIC_THEME, qm, session_seed(42, idx), (0, 1))
        # each row prints "p99=<hi>  p50=<lo>" -> per-row (shared suffix) ordering.
        for p99v, p50v in re.findall(r"p99=(\d+)\s+p50=(\d+)", out):
            assert int(p50v) <= int(p99v), f"p50_N {p50v} > p99_N {p99v} in {out!r}"

    idx_ms = "a={p50_ms0} b={p99_ms0} c={p50_ms1} d={p99_ms1}"
    for idx in range(20):
        out = _render_theme_template(GENERIC_THEME, idx_ms, session_seed(42, idx), (0, 1))
        a, b, c, d = (int(x) for x in re.findall(r"=(\d+)", out))
        assert a <= b, f"p50_ms0 {a} > p99_ms0 {b} in {out!r}"
        assert c <= d, f"p50_ms1 {c} > p99_ms1 {d} in {out!r}"


def test_heap_used_le_heap_max():
    import re

    # heap_used{N} must be clamped to heap_max{N} (same pattern as in_use/max).
    tpl = "gc heap_used_mb={heap_used0} heap_max_mb={heap_max0} note={heap_used1}/{heap_max1}"
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        pairs = re.findall(r"heap_used_mb=(\d+) heap_max_mb=(\d+)", out) + re.findall(r"note=(\d+)/(\d+)", out)
        assert pairs, f"heap pair not rendered: {out!r}"
        for used, mx in pairs:
            assert int(used) <= int(mx), f"heap_used {used} > heap_max {mx} in {out!r}"


def test_percentile_and_heap_render_no_placeholder_leak():
    # The invariant pass never leaves a placeholder unfilled or crashes, even with
    # mixed percentile-shaped names present (`p95word` is a distinct group member).
    tpl = "p50={p50} p99={p99} heap_used0={heap_used0} heap_max0={heap_max0} note={p95word}"
    out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, 0), (0, 1))
    assert "{" not in out and "}" not in out, f"unfilled placeholder leaked: {out!r}"


# --- Item-4 fix 2: connective casing seam in follow-ups ---------------------
#
# "Following up, Are other services..." (capital right after a lowercase
# connective+comma) is an obvious concatenation seam. A common-word first token
# is lowercased; an entity/proper-noun/acronym first word is preserved.


def test_connective_lowercases_common_first_word():
    from inference_perf.datagen.synthetic_agentic import _join_connective_case

    out = _join_connective_case(
        "Following up, ", "Are other services in us-east-1 showing the same 5xx?", GENERIC_THEME
    )
    assert out.startswith("are other services"), f"common-word seam not fixed: {out!r}"
    # full join has no capital-after-lowercase-connective seam.
    joined = "Following up, " + out
    assert "Following up, are" in joined, f"casing seam remains: {joined!r}"


def test_connective_preserves_entity_and_acronym_first_word():
    from inference_perf.datagen.synthetic_agentic import _join_connective_case

    # An entity value (service name) as the first word is a proper noun -> preserved.
    entity = GENERIC_THEME.entities["service"][2]  # "cart-service"
    out = _join_connective_case("Following up, ", f"{entity} is down, why?", GENERIC_THEME)
    assert out.startswith(entity), f"entity first word wrongly lowercased: {out!r}"
    # An all-caps acronym is preserved.
    assert _join_connective_case("Next, ", "DBP1 shows lock waits", GENERIC_THEME).startswith("DBP1")
    # A token containing a digit (e.g. Db2) is preserved.
    assert _join_connective_case("OK, and ", "Db2 latency spiked", GENERIC_THEME).startswith("Db2")
    # An empty connective leaves the text untouched.
    assert _join_connective_case("", "Are other services down?", GENERIC_THEME) == "Are other services down?"


def test_generated_followups_have_no_casing_seam():
    # In a real multi-round session, no follow-up shows a capital letter right
    # after a lowercase connective+comma (unless it's a preserved proper noun).
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=4),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    entity_pool = {v for vals in GENERIC_THEME.entities.values() for v in vals}
    connectives = tuple(c for c in GENERIC_THEME.followup_connectives if c.endswith(" "))
    for idx in range(8):
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        for eid, ev in g.events.items():
            if not re.match(r".*:r([1-9]\d*):principal$", eid):
                continue
            content = [m for m in ev.call.messages if m.get("role") == "user"][-1]["content"]
            real = content.rsplit(FILLER_CLOSE, 1)[-1]
            for conn in connectives:
                pos = real.find(conn)
                if pos < 0:
                    continue
                after = real[pos + len(conn):]
                first_tok = after.split(maxsplit=1)[0] if after.split() else ""
                if not first_tok or not first_tok[0].isupper():
                    continue  # already lowercased -> no seam
                # a leading capital is only OK if it's a preserved proper noun.
                is_entity = first_tok in entity_pool or after.startswith(tuple(entity_pool))
                is_acronym = first_tok.isupper()
                has_digit = any(c.isdigit() for c in first_tok)
                assert is_entity or is_acronym or has_digit, (
                    f"idx{idx} {eid}: casing seam after {conn!r}: {after[:50]!r}"
                )


# --- Item-4 fix 3: region pinned across follow-ups --------------------------


def test_region_is_pinned_across_a_multi_round_session():
    # objective / intro doc / every follow-up must reference the SAME region.
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=4),
        tool_turns_per_loop=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    import re

    regions = GENERIC_THEME.entities["region"]
    for idx in range(10):
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        seen = set()
        for eid, ev in g.events.items():
            if not re.match(r".*:r\d+:principal$", eid):
                continue
            content = [m for m in ev.call.messages if m.get("role") == "user"][-1]["content"]
            for r in regions:
                if r in content:
                    seen.add(r)
        assert len(seen) <= 1, f"idx{idx}: session references {sorted(seen)} regions, not one"


def test_region_in_primary_categories_and_pinned():
    # region is now a pinned primary-subject category, and _pinned_primary_entities
    # only pins categories the theme declares (a theme without `region` is unaffected).
    from inference_perf.datagen.synthetic_agentic import (
        _PRIMARY_ENTITY_CATEGORIES,
        _pinned_primary_entities,
    )

    assert "region" in _PRIMARY_ENTITY_CATEGORIES
    pinned = _pinned_primary_entities(GENERIC_THEME, child_rng(session_seed(42, 0), 62))
    assert pinned.get("region") in GENERIC_THEME.entities["region"]

    # a theme WITHOUT region declares none -> no region key pinned (unaffected).
    bare = Theme(
        name="bare_no_region",
        verbs=["Do"],
        entities={"service": ["svc-a"]},
        tool_names=["t"],
        result_templates={"default": "r {n0}"},
        objective_template="{verb} {service}",
    )
    bare_pinned = _pinned_primary_entities(bare, child_rng(session_seed(42, 0), 62))
    assert "region" not in bare_pinned
    assert bare_pinned.get("service") == "svc-a"


# --- code_change_task theme (READ/RUN subset) --------------------------------
#
# item 2 realism theme: a coding agent that reads/searches code, inspects the
# current diff, and runs tests. The write/edit tools were DROPPED (they need a
# generator arg_templates change to emit realistic payloads), so this asset is
# a read-only tool catalog whose small string args (paths/symbols/patterns) the
# current f"{prop}-NNN" stub renders acceptably.

_CODE_CHANGE_READ_RUN_TOOLS = {
    "list_dir",
    "read_file",
    "grep_code",
    "find_symbol",
    "git_diff",
    "run_tests",
    "run_command",
}
# Write/edit tools: now supported — their big-payload args (content/new_string/patch)
# render as sized code-shaped filler (item 5), so the theme includes them.
_CODE_CHANGE_WRITE_TOOLS = {"edit_file", "write_file", "apply_patch"}


def _code_change_cfg(**kw):
    base = dict(
        num_sessions=5,
        rounds_per_session=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
        theme_mix={"code_change_task": 1.0},
        input_tokens_per_turn=Distribution(type="fixed", mean=40),
        output_tokens_per_turn=Distribution(type="fixed", mean=20),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=6),
        tool_definitions_per_agent=Distribution(type="fixed", mean=7),
    )
    base.update(kw)
    return SyntheticAgenticConfig(**base)


def test_code_change_task_loads_and_validates():
    t = load_theme("code_change_task")
    assert isinstance(t, Theme)
    assert t.name == "code_change_task"
    assert t.verbs  # non-empty
    assert t.tool_names  # non-empty
    assert "default" in t.result_templates
    # read/run subset present, AND the write tools (now supported via sized payload args).
    names = set(t.tool_names)
    assert _CODE_CHANGE_READ_RUN_TOOLS <= names, "read/run tools must all be present"
    assert _CODE_CHANGE_WRITE_TOOLS <= names, "write tools must be present (payload args are now sized)"
    # every tool_parameters entry is a well-formed JSON-Schema object with its
    # required names present in properties.
    for name, spec in t.tool_parameters.items():
        assert spec.get("type") == "object"
        assert isinstance(spec.get("properties"), dict)
        for req in spec.get("required", []):
            assert req in spec["properties"], f"{name}: required {req!r} missing from properties"


def _iter_tool_calls_and_results(g):
    """Yield (kind, call_name, payload) for every emitted tool call and every
    role:tool result content across a graph."""
    for ev in g.events.values():
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                yield "call", tc["function"]["name"], tc["function"]["arguments"]
            if m.get("role") == "tool":
                yield "result", None, str(m.get("content", ""))


def test_code_change_task_session_args_are_valid_json_and_results_have_no_leak():
    import json as _json
    import re

    t = load_theme("code_change_task")
    cfg = _code_change_cfg()
    g = build_graph_for_session(cfg, t, _WordTok(), session_index=0)
    assert g.events, "code_change_task builds a non-empty session"
    placeholder = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
    saw_call = saw_result = False
    for kind, _name, payload in _iter_tool_calls_and_results(g):
        if kind == "call":
            saw_call = True
            obj = _json.loads(payload)  # every tool-call arg is valid JSON
            assert isinstance(obj, dict)
            for v in obj.values():
                if isinstance(v, str):
                    assert "{" not in v and "}" not in v, f"arg value leak: {v!r}"
        else:
            saw_result = True
            # no unrendered placeholder / brace leak in any result
            assert "{" not in payload and "}" not in payload, f"brace leak in result: {payload[:80]!r}"
            assert not placeholder.search(payload), f"placeholder leak in result: {payload[:80]!r}"
    assert saw_call and saw_result, "session emitted both tool calls and results"


def test_code_change_task_deterministic_per_config_and_index():
    t = load_theme("code_change_task")
    cfg = _code_change_cfg()
    g1 = build_graph_for_session(cfg, t, _WordTok(), session_index=3)
    g2 = build_graph_for_session(cfg, t, _WordTok(), session_index=3)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.expected_output == g2.events[eid].call.expected_output


def _find_results_for_tool(g, tool_name):
    """Return the rendered result strings for a given tool name.

    A tool call is emitted in event X (assistant tool_calls) and its result is
    re-injected as a role:tool message in a SUCCESSOR event. We match on the
    tool_call_id prefix (`call_...:tT_J`) which encodes nothing tool-specific,
    so instead we pair calls to results by their shared tool_call_id across the
    whole graph.
    """
    # name -> set of tool_call_ids
    ids_for_name = set()
    for ev in g.events.values():
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                if tc["function"]["name"] == tool_name:
                    ids_for_name.add(tc["id"])
    results = []
    for ev in g.events.values():
        for m in ev.call.messages:
            if m.get("role") == "tool" and m.get("tool_call_id") in ids_for_name:
                results.append(str(m.get("content", "")))
    return results


def test_code_change_task_result_shapes_render_realistically():
    # run_tests -> traceback marker + pass/fail summary; git_diff -> unified diff
    # markers; read_file -> line-number formatting. Rendered directly so the test
    # does not depend on which tools a given session happens to schedule.
    from inference_perf.datagen.synthetic_agentic import _render_tool_result

    t = load_theme("code_change_task")
    seed = 4242

    run_tests = _render_tool_result(t, "run_tests", seed, (1, 2, 3))
    assert ("Traceback" in run_tests) or ("AttributeError" in run_tests) or ("AssertionError" in run_tests), run_tests
    assert ("passed" in run_tests and "failed" in run_tests), run_tests

    git_diff = _render_tool_result(t, "git_diff", seed, (4, 5, 6))
    assert "@@" in git_diff, git_diff
    assert "+++" in git_diff, git_diff
    assert git_diff.count("\n") >= 3, "git_diff should be a multi-line unified diff"

    read_file = _render_tool_result(t, "read_file", seed, (7, 8, 9))
    # numbered source lines "  NN | ..." formatting
    import re

    assert re.search(r"\d+ \| ", read_file), read_file
    assert "def " in read_file, read_file

    # grep_code -> path:lineno: hits (multi-line)
    grep_code = _render_tool_result(t, "grep_code", seed, (10, 11, 12))
    assert re.search(r"\S+:\d+:", grep_code), grep_code

    # none of these leak a placeholder
    for r in (run_tests, git_diff, read_file, grep_code):
        assert "{" not in r and "}" not in r, r[:80]


# --- Item 5 + 6a: payload args, tool rotation, coherent focus threading -----


def test_write_tool_payload_args_are_sized_code_filler():
    # A write tool's big-payload arg (content/new_string/patch) is NOT the tiny
    # `{prop}-NNN` stub: it's a substantial chunk drawn from the theme filler pool.
    import json as _json

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        rounds_per_session=Distribution(type="fixed", mean=1),
        tool_turns_per_loop=Distribution(type="fixed", mean=10),  # reach edit/write/apply (tools 8-10)
        tool_definitions_per_agent=Distribution(type="fixed", mean=10),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, t, _WordTok(), session_index=0)
    seen_payload = 0
    for ev in g.events.values():
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                if tc["function"]["name"] not in _CODE_CHANGE_WRITE_TOOLS:
                    continue
                a = _json.loads(tc["function"]["arguments"])  # valid JSON (inv #1)
                for key in ("content", "new_string", "patch"):
                    if key in a:
                        seen_payload += 1
                        val = a[key]
                        assert len(val.split()) >= 20, f"payload {key} too small (stub?): {val!r}"
                        assert not val.startswith(f"{key}-"), f"payload {key} is still the stub: {val!r}"
    assert seen_payload > 0, "no write-tool payload arg observed (raise k / catalog?)"


def test_non_payload_string_arg_keeps_stub():
    # A non-payload string arg that is not an entity category (e.g. grep pattern)
    # keeps the short `{prop}-NNN` stub — payload sizing is scoped to payload names.
    import json as _json
    import re

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(rounds_per_session=Distribution(type="fixed", mean=1),
                           tool_turns_per_loop=Distribution(type="fixed", mean=8), fanout_probability=0.0)
    g = build_graph_for_session(cfg, t, _WordTok(), session_index=0)
    saw = False
    for ev in g.events.values():
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                if tc["function"]["name"] == "grep_code":
                    a = _json.loads(tc["function"]["arguments"])
                    if "pattern" in a:
                        saw = True
                        assert re.fullmatch(r"pattern-\d+", a["pattern"]), a["pattern"]
    assert saw, "no grep_code pattern arg seen"


def test_tool_loop_varies_tools_across_turns():
    # 6a-i: a multi-turn loop uses >=2 distinct tools (not tool_defs[0] x k).
    import re

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(rounds_per_session=Distribution(type="fixed", mean=1),
                           tool_turns_per_loop=Distribution(type="fixed", mean=6),
                           tool_definitions_per_agent=Distribution(type="fixed", mean=10), fanout_probability=0.0)
    g = build_graph_for_session(cfg, t, _WordTok(), session_index=0)
    names = []
    for eid, ev in g.events.items():
        if re.search(r":t\d+$", eid) or eid.endswith(":principal"):
            for m in ev.call.messages:
                for tc in m.get("tool_calls", []) or []:
                    names.append(tc["function"]["name"])
    # dedup consecutive isn't enough; just assert variety across the loop
    assert len(set(names)) >= 2, f"loop used only one tool: {set(names)}"


def test_focus_entity_threads_across_the_loop():
    # 6a-ii: the file path referenced across a session's tool calls is ONE focus
    # value (coherent chain), and different sessions pin different focuses.
    import json as _json

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(rounds_per_session=Distribution(type="fixed", mean=1),
                           tool_turns_per_loop=Distribution(type="fixed", mean=8),
                           tool_definitions_per_agent=Distribution(type="fixed", mean=10), fanout_probability=0.0)

    def paths_in(idx):
        g = build_graph_for_session(cfg, t, _WordTok(), session_index=idx)
        ps = set()
        for ev in g.events.values():
            for m in ev.call.messages:
                for tc in m.get("tool_calls", []) or []:
                    a = _json.loads(tc["function"]["arguments"])
                    if "path" in a and a["path"] in t.entities["path"]:
                        ps.add(a["path"])
        return ps

    s0 = paths_in(0)
    assert len(s0) == 1, f"session 0 should reference ONE focus path, got {s0}"
    # a different session pins a (usually) different focus — at least not forced identical
    s1 = paths_in(1)
    assert len(s1) == 1
    # determinism: same index -> same focus
    assert paths_in(0) == s0


def test_code_change_focus_and_payload_deterministic():
    t = load_theme("code_change_task")
    cfg = _code_change_cfg(rounds_per_session=Distribution(type="fixed", mean=1),
                           tool_turns_per_loop=Distribution(type="fixed", mean=10),
                           tool_definitions_per_agent=Distribution(type="fixed", mean=10), fanout_probability=0.0)
    g1 = build_graph_for_session(cfg, t, _WordTok(), session_index=2)
    g2 = build_graph_for_session(cfg, t, _WordTok(), session_index=2)
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


# --- Context compaction -----------------------------------------------------
#
# A multi-round session normally GROWS: round r+1's principal re-injects the
# prior transcript via shared+output+unique. With a context_compaction block
# set, once a round's accumulated principal input (content + tool catalog) crosses
# the trigger the NEXT round instead starts FRESH (all-unique, no shared) with a
# seeded summary block replacing the history -> a prefill drop. In _WordTok units
# (1 token == 1 word) the 8-tool generic catalog is ~535 words and the no-compaction
# accumulation climbs ~618, 637, 650, 667, ... per round, so a trigger in that band
# compacts after a couple of grown rounds.


def _compaction_cfg(**kw):
    """A multi-round single-agent config (no tools in the loop, so rounds are the
    only growth) tuned for compaction tests in _WordTok units."""
    base = dict(
        num_sessions=1,
        seed=7,
        rounds_per_session=Distribution(type="fixed", mean=6),
        fanout_probability=0.0,
        theme_mix={"generic": 1.0},
        tool_turns_per_loop=Distribution(type="fixed", mean=0),
        tool_definitions_per_agent=Distribution(type="fixed", mean=8),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
    )
    base.update(kw)
    return SyntheticAgenticConfig(**base)


def _cc(trigger, target):
    """Shorthand for a ContextCompactionConfig with fixed trigger/target token counts."""
    return ContextCompactionConfig(
        trigger_tokens=Distribution(type="fixed", mean=trigger),
        target_tokens=Distribution(type="fixed", mean=target),
    )


def _principal_segments(g):
    """Ordered list of (event_id, [segment types]) for every :principal event.
    A fresh/compacted principal has NO input_segments (None or [])."""
    out = []
    for eid, ev in g.events.items():
        if eid.endswith(":principal"):
            segs = ev.call.input_segments or []
            out.append((eid, [s.type for s in segs]))
    return out


def test_compaction_off_by_default_is_byte_identical():
    # A config WITHOUT the context_compaction block must produce the exact same graph
    # as before the feature existed: the unset block must not shift any seed path.
    # We assert by re-deriving with an explicitly-None block.
    plain = _compaction_cfg()
    withnone = _compaction_cfg(context_compaction=None)
    g1 = build_graph_for_session(plain, GENERIC_THEME, _WordTok(), 0)
    g2 = build_graph_for_session(withnone, GENERIC_THEME, _WordTok(), 0)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments
    # And no compaction => every round r>=1 grows (shared+output+unique).
    for eid, types in _principal_segments(g1):
        if eid == "synthN0:r0:principal":
            assert types == [], "round 0 is always fresh"
        else:
            assert types == ["shared", "output", "unique"], f"{eid} should GROW when compaction off"


def test_compaction_trigger_high_never_compacts():
    # A trigger far above any achievable accumulation must behave exactly like
    # compaction-off: every r>=1 round still grows.
    cfg = _compaction_cfg(context_compaction=_cc(10_000_000, 12))
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    for eid, types in _principal_segments(g):
        if eid != "synthN0:r0:principal":
            assert types == ["shared", "output", "unique"], f"{eid} should GROW under a huge trigger"


def test_compaction_fires_mid_session():
    # A trigger inside the accumulation band compacts at least one mid-session
    # round: that round's principal is FRESH (all-unique, no shared/output),
    # i.e. it does NOT slice into the prior principal -> the transcript is dropped.
    cfg = _compaction_cfg(
        rounds_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    seg_map = dict(_principal_segments(g))
    # some mid-session round (r>=1) reset to fresh
    compacted = [eid for eid, types in seg_map.items() if eid != "synthN0:r0:principal" and types == []]
    assert compacted, f"expected at least one mid-session compaction, got {seg_map}"
    # a compacted round has exactly ONE user message in its build-time input (the
    # summary+objective turn); the system head is added by _build_agent, so the
    # principal input placeholder is the single fresh user turn.
    for eid in compacted:
        msgs = g.events[eid].call.messages
        assert len(msgs) == 1 and msgs[0]["role"] == "user", f"{eid} compacted principal must be one fresh user turn"
        assert (g.events[eid].call.input_segments or []) == [], f"{eid} must have NO shared/output segments"
        # ordering edge to the prior answer is preserved (session stays one chain)
        assert g.events[eid].predecessor_event_ids, f"{eid} should keep an ordering edge to the prior round"


def test_compaction_summary_block_present_and_sized():
    # The compacted round's user turn carries a seeded summary block (plus the
    # objective). With a small target the turn is much smaller than a grown round
    # would be -> the prefill drop.
    cfg = _compaction_cfg(
        rounds_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    seg_map = dict(_principal_segments(g))
    compacted = [eid for eid, t in seg_map.items() if eid != "synthN0:r0:principal" and t == []]
    assert compacted
    content = g.events[compacted[0]].call.messages[0]["content"]
    assert "Summary of prior context:" in content, "compacted turn must carry the summary fixed-content"


def test_compaction_recap_names_real_subject_and_tools():
    # When the theme defines compaction_summary_template, the recap is a real
    # semantic handoff: it names the session's pinned subject and REAL tool names
    # from the catalog (not generic filler), so it reads like a genuine recap.
    cfg = _compaction_cfg(
        rounds_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 40),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    compacted = [eid for eid, t in _principal_segments(g) if eid != "synthN0:r0:principal" and t == []]
    assert compacted
    content = g.events[compacted[0]].call.messages[0]["content"]
    # the generic recap template names a verb, the pinned subject, and 3 real tools
    assert "So far: ran" in content, "recap sentence should be rendered, not the bare marker"
    catalog = {td["name"] for td in _tool_definitions(GENERIC_THEME, 8)}
    named = [name for name in catalog if name in content]
    assert len(named) >= 2, f"recap should name real tools from the catalog, found {named}"


def test_compaction_recap_falls_back_to_bare_marker_without_template():
    # A theme with NO compaction_summary_template still compacts, using the bare
    # "Summary of prior context:" marker (no recap sentence). Build a minimal theme.
    bare = Theme(
        name="bare",
        system_prompt="sys",
        verbs=["Do"],
        entities={"thing": ["x", "y"]},
        tool_names=["t1", "t2"],
        result_templates={"default": "result {thing}"},
        objective_template="{verb} {thing}",
        followup_templates=["more on {thing}?"],
    )
    # bare theme has tiny content, so its accumulation is small -> use a low trigger.
    cfg = _compaction_cfg(
        theme_mix={"bare": 1.0},
        rounds_per_session=Distribution(type="fixed", mean=8),
        tool_definitions_per_agent=Distribution(type="fixed", mean=2),
        context_compaction=_cc(90, 12),
    )
    g = build_graph_for_session(cfg, bare, _WordTok(), 0)
    compacted = [eid for eid, t in _principal_segments(g) if eid != "synthN0:r0:principal" and t == []]
    assert compacted, "compaction fires regardless of whether the theme has a recap template"
    content = g.events[compacted[0]].call.messages[0]["content"]
    assert "Summary of prior context:" in content
    assert "So far: ran" not in content, "no recap sentence when the theme defines no template"


def test_accumulated_wire_tokens_includes_catalog():
    tok = _WordTok()
    defs = _tool_definitions(GENERIC_THEME, 8)
    msgs = [{"role": "user", "content": "one two three four five"}]
    with_cat = _accumulated_wire_tokens(tok, msgs, defs)
    without_cat = _accumulated_wire_tokens(tok, msgs, [])
    import json as _json

    assert with_cat - without_cat == tok.count_tokens(_json.dumps(defs)), "catalog tokens must be added"
    assert without_cat == tok.count_tokens("one two three four five")


def test_compaction_config_requires_both_fields():
    from pydantic import ValidationError

    # The nested block requires BOTH trigger_tokens and target_tokens.
    with pytest.raises(ValidationError):
        ContextCompactionConfig(trigger_tokens=Distribution(type="fixed", mean=655))
    with pytest.raises(ValidationError):
        ContextCompactionConfig(target_tokens=Distribution(type="fixed", mean=12))
    # both set -> accepted, and attaches cleanly to the parent config
    _compaction_cfg(context_compaction=_cc(655, 12))
    # block omitted -> accepted (compaction off)
    _compaction_cfg()


def test_compaction_deterministic():
    cfg = _compaction_cfg(
        rounds_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 2)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 2)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments
