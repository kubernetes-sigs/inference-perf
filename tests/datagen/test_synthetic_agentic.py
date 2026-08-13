import collections
import re
from typing import TYPE_CHECKING, Any, Dict, cast

import pytest
from inference_perf.datagen.replay_graph_types import GraphEvent, ReplayGraph
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
    DISPATCH_AGENT_NAME,
)
from inference_perf.config.common import Distribution
from inference_perf.config.datagen.replay import SyntheticAgenticConfig, ContextCompactionConfig

if TYPE_CHECKING:
    from inference_perf.config import APIConfig
    from inference_perf.utils.custom_tokenizer import CustomTokenizer


def test_load_bundled_theme() -> None:
    t = load_theme("db2_latency_incident")
    assert isinstance(t, Theme)
    assert t.name == "db2_latency_incident"
    assert t.objective_template  # non-empty
    assert len(t.verbs) >= 3
    assert t.tool_names  # at least one tool


def test_generic_theme_is_valid() -> None:
    assert isinstance(GENERIC_THEME, Theme)
    assert GENERIC_THEME.objective_template


def test_load_unknown_theme_raises() -> None:
    with pytest.raises(ValueError):
        load_theme("nonexistent_theme_xyz")


def test_config_requires_the_four_required_fields() -> None:
    from pydantic import ValidationError
    from inference_perf.config.datagen.replay import SyntheticAgenticConfig

    with pytest.raises(ValidationError):
        # Omitting the required fields is the POINT of this test, so mypy's
        # missing-argument complaint is the condition under assertion.
        SyntheticAgenticConfig()  # type: ignore[call-arg]  # missing num_sessions/rounds/fanout/theme_mix


def test_config_valid_minimal() -> None:
    from inference_perf.config.common import Distribution
    from inference_perf.config.datagen.replay import SyntheticAgenticConfig
    from inference_perf.config.datagen.replay import BadToolCallHandling

    cfg = SyntheticAgenticConfig(
        num_sessions=10,
        turns_per_session=Distribution(type="fixed", mean=1),
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


def test_session_seed_stable_across_calls_and_processes() -> None:
    # Must NOT depend on PYTHONHASHSEED or process -- pure function of inputs.
    a = session_seed(42, 17)
    b = session_seed(42, 17)
    assert a == b
    assert session_seed(42, 18) != a  # different index -> different seed


def test_child_rng_path_derived_independent() -> None:
    r1 = child_rng(session_seed(42, 0), 1, 2, 3)
    r2 = child_rng(session_seed(42, 0), 1, 2, 3)
    assert r1.integers(0, 1_000_000) == r2.integers(0, 1_000_000)  # reproducible
    r3 = child_rng(session_seed(42, 0), 1, 2, 4)  # different path
    assert r3.integers(0, 1_000_000) != r1.integers(0, 1_000_000)


class _FakeTok:
    # 1 token per whitespace-word, deterministic -- good enough to test budget logic
    def count_tokens(self, text: Any, add_special_tokens: bool = True) -> int:
        return len(text.split())

    def get_tokenizer(self) -> None:
        raise NotImplementedError


def test_fit_filler_negative_budget_returns_fixed_only_no_wrapper() -> None:
    tok = _FakeTok()
    fixed = "objective line here"  # 3 tokens
    out = fit_filler(tok, target_tokens=2, fixed_content=fixed, rng=None)  # target < fixed
    assert FILLER_OPEN not in out and FILLER_CLOSE not in out
    assert out == fixed  # floored to fixed content, no crash


def test_tool_call_margin_value() -> None:
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


def _real_tokenizer() -> "CustomTokenizer":
    """Load the real CustomTokenizer, or skip if it can't be loaded offline."""
    try:
        from inference_perf.config import CustomTokenizerConfig
        from inference_perf.utils.custom_tokenizer import CustomTokenizer

        return CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=_REAL_TOKENIZER_MODEL))
    except Exception as e:  # network down / model unavailable in CI
        pytest.skip(f"real tokenizer {_REAL_TOKENIZER_MODEL} unavailable: {e}")


def _untruncated_token_count(ct: "CustomTokenizer", text: str) -> int:
    """Length of `text` in tokens WITHOUT the model_max_length truncation.

    count_tokens truncates at model_max_length (8192 here), so it cannot
    measure a 100K-token string. The underlying HF tokenizer called with
    truncation=False gives the true length.
    """
    return len(ct.get_tokenizer()(text, truncation=False, add_special_tokens=False)["input_ids"])


def test_fit_filler_reaches_large_target() -> None:
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


def test_fit_filler_large_target_is_fast() -> None:
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


# --- Seeded single-agent walk ------------------------------------------


class _WordTok:
    def count_tokens(self, text: Any, add_special_tokens: bool = True) -> int:
        return max(1, len(str(text).split()))

    def get_tokenizer(self) -> None:
        raise NotImplementedError


def _word_tok() -> "CustomTokenizer":
    """A `_WordTok` typed as a CustomTokenizer for the generator's typed constructor.

    `_WordTok` is a deliberately minimal structural double: the graph builder only ever
    calls `count_tokens` on its tokenizer, so the double implements just that (plus a
    `get_tokenizer` that raises, to prove nothing reaches for the real HF tokenizer).
    It is not a nominal `CustomTokenizer` subclass, so a cast is needed at the typed
    call sites; keeping it in one helper documents why instead of scattering ignores.
    """
    return cast("CustomTokenizer", _WordTok())


def _cfg(**kw: Any) -> SyntheticAgenticConfig:
    base: Dict[str, Any] = dict(
        num_sessions=5,
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        theme_mix={"generic": 1.0},
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=2),
    )
    base.update(kw)
    return SyntheticAgenticConfig(**base)


def test_single_agent_graph_structure() -> None:
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


def test_determinism_same_index_same_graph() -> None:
    g1 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    g2 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    assert list(g1.events.keys()) == list(g2.events.keys())  # same ids, same insertion order


def test_event_budget_caps_rounds() -> None:
    cfg = _cfg(turns_per_session=Distribution(type="fixed", mean=100), max_events_per_session=6)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    assert len(g.events) <= 6


# --- Recursive fan-out + merge via tool_output --------------------------


def test_fanout_produces_subagents_and_valid_merge() -> None:
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


def test_no_agent_beyond_max_depth() -> None:
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


def test_agent_first_call_carries_role_appropriate_system_head() -> None:
    # Every agent's first call carries a {role:"system"} head, drawn from the
    # ROLE-appropriate pool: the root/orchestrator from ROOT_SYSTEM_PROMPTS, a
    # spawned sub-agent from SUBAGENT_SYSTEM_PROMPTS -- so the root and its
    # sub-agents carry DIFFERENT heads (like a real harness), not one identical
    # head. Each head is a distinct dict (no aliasing).
    from inference_perf.datagen.synthetic_themes import ROOT_SYSTEM_PROMPTS, SUBAGENT_SYSTEM_PROMPTS

    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        # above the longest prompt (~540 words in _WordTok units) so each head is
        # the full real prompt + filler, keeping its opening intact for the checks.
        shared_system_prompt_len=800,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)

    def _system_msg(ev: GraphEvent) -> Any:
        for m in ev.call.messages:
            if m.get("role") == "system":
                return m
        return None

    def _opening(head: Any) -> Any:
        # the real prompt precedes any appended "## Operational context" filler
        return head["content"].split("## Operational context")[0].strip()

    root_openings = {p.split("## Operational context")[0].strip() for p in ROOT_SYSTEM_PROMPTS}
    sub_openings = {p.split("## Operational context")[0].strip() for p in SUBAGENT_SYSTEM_PROMPTS}

    # root first call -> a ROOT prompt
    root_id = g.root_event_ids[0]
    root_system = _system_msg(g.events[root_id])
    assert root_system is not None, "root first call carries a system head"
    assert _opening(root_system) in root_openings, "root head comes from ROOT_SYSTEM_PROMPTS"

    # each sub-agent first call -> a SUBAGENT prompt (and NOT the root's head)
    seen_dicts = [root_system]
    sub_firsts = [ev for eid, ev in g.events.items() if ":sub" in eid and ":principal" in eid]
    assert sub_firsts, "at least one sub-agent principal event exists"
    for ev in sub_firsts:
        sm = _system_msg(ev)
        assert sm is not None, "sub-agent first call carries a system head"
        assert _opening(sm) in sub_openings, "sub-agent head comes from SUBAGENT_SYSTEM_PROMPTS"
        assert sm["content"] != root_system["content"], "sub-agent head differs from the root's"
        for prior in seen_dicts:
            assert sm is not prior, "each event's system head is a distinct dict (no aliasing)"
        seen_dicts.append(sm)


def test_event_budget_cost_is_k_plus_1_per_round() -> None:
    # A round emits 1 principal + k tool-turn events, where the LAST tool turn's
    # OUTPUT is the answer (no separate answer event) = k + 1 events. With
    # tool_loop_depth fixed at k=2 each round costs exactly 3 events. A budget of 9
    # fits exactly 3 whole rounds (3 * 3 = 9); a budget of 8 fits only 2 whole rounds
    # (the 3rd would need 3 more, overflowing) and STOPS -- confirming the per-round
    # cost is k+1.
    cfg9 = _cfg(
        turns_per_session=Distribution(type="fixed", mean=100),
        max_events_per_session=9,
        tool_loop_depth=Distribution(type="fixed", mean=2),
    )
    g9 = build_graph_for_session(cfg9, GENERIC_THEME, _WordTok(), 0)
    assert len(g9.events) == 9, f"expected 3 rounds of (k+1)=3 events, got {len(g9.events)}"

    cfg6 = _cfg(
        turns_per_session=Distribution(type="fixed", mean=100),
        max_events_per_session=6,
        tool_loop_depth=Distribution(type="fixed", mean=2),
    )
    g6 = build_graph_for_session(cfg6, GENERIC_THEME, _WordTok(), 0)
    # exactly 2 full rounds (6 events); the 3rd round can't even start its
    # principal (6 + 1 > 6), so it never begins. Result: exactly 6 events.
    assert len(g6.events) == 6, f"expected 2 full rounds of (k+1)=3 events, got {len(g6.events)}"


# --- Generator class (lazy build + theme weighting) --------------------


def _min_api() -> "APIConfig":
    from inference_perf.config import APIConfig, APIType

    return APIConfig(type=APIType.Chat, streaming=False)


def test_generator_builds_session_lazily() -> None:
    from inference_perf.config.datagen.config import DataConfig, DataGenType
    from inference_perf.datagen.synthetic_agentic import SyntheticAgenticDataGenerator

    data = DataConfig(type=DataGenType.SyntheticAgentic, synthetic_agentic=_cfg(num_sessions=4))
    gen = SyntheticAgenticDataGenerator(api_config=_min_api(), config=data, tokenizer=_word_tok(), num_workers=1)
    assert gen.get_session_count() == 4
    gen._ensure_session_built(0)
    assert gen.sessions[0] is not None
    # determinism: two generators, same index -> same event ids
    gen2 = SyntheticAgenticDataGenerator(api_config=_min_api(), config=data, tokenizer=_word_tok(), num_workers=1)
    gen2._ensure_session_built(0)
    s1, s2 = gen.sessions[0], gen2.sessions[0]
    assert s1 is not None and s2 is not None
    assert list(s1.graph.events.keys()) == list(s2.graph.events.keys())


# --- main.py dispatch wiring -------------------------------------------


def test_dispatch_resolves_synthetic_generator() -> None:
    # Minimal: assert the generator class is importable and the enum value maps.
    from inference_perf.config.datagen.config import DataGenType
    from inference_perf.datagen import SyntheticAgenticDataGenerator

    assert DataGenType.SyntheticAgentic.value == "synthetic_agentic"
    assert SyntheticAgenticDataGenerator is not None


# --- End-to-end integration guard (no dangling tool_call_ids) ----------


# --- Follow-up: input_tokens_per_turn must actually size input turns --------


def _principal_user_content(g: ReplayGraph) -> Any:
    """Return the user-role content string of the sole root principal turn."""
    root_id = g.root_event_ids[0]
    ev = g.events[root_id]
    user_msgs = [m for m in ev.call.messages if m.get("role") == "user"]
    assert user_msgs, "principal turn has a user message"
    return user_msgs[-1]["content"]


def test_input_tokens_per_turn_is_honored() -> None:
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


def test_input_sizing_preserves_determinism_and_objective_text() -> None:
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


# --- Follow-up: parallel_tool_calls_per_step on ordinary tool turns --------


def _find_tool_turn_events(g: ReplayGraph) -> Any:
    """Return the ordinary tool-loop turn events (id ends with ':tN').

    These are the ORDINARY tool turns emitted in _build_agent's tool-loop
    (NOT dispatch events, NOT the merge). Their assistant message carries the
    K parallel calls and is followed by K role:tool results.
    """
    import re

    return [ev for eid, ev in g.events.items() if re.search(r":t\d+$", eid)]


def _last_tool_call_group(ev: GraphEvent) -> Any:
    """Return (assistant_tool_calls, trailing_tool_results) for the LAST
    tool-call group in an event's transcript.

    A ':tN' event's input is the growing transcript ending in
    [<prior turns>, assistant(K calls), tool×K]. The K calls of THIS turn are the
    last assistant tool_call message; its results are the trailing role:tool
    messages. Prior turns may add earlier
    assistant/tool messages, so we look at the final group only."""
    calls = None
    for m in ev.call.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
    tool_msgs = [m for m in ev.call.messages if m.get("role") == "tool"]
    return calls, tool_msgs


def test_parallel_tool_calls_emits_k_calls_and_k_results() -> None:
    # parallel_tool_calls_per_step fixed 3 -> the tool-turn event that carries a
    # turn's result reconstructs an assistant message with 3 tool_calls AND 3
    # role:tool results, ids matching 1:1 in positional order (inv #3).
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=1),
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


def test_parallel_default_is_single_call() -> None:
    # parallel_tool_calls_per_step unset (None -> fallback fixed 1): an ordinary
    # tool turn has exactly 1 call + 1 result (unchanged default behavior).
    cfg = _cfg(
        tool_loop_depth=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
    )
    assert cfg.parallel_tool_calls_per_step is None
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    turns = _find_tool_turn_events(g)
    assert turns, "at least one ordinary tool-turn event exists"
    for ev in turns:
        calls, tool_msgs = _last_tool_call_group(ev)
        # the LAST tool-call group (this turn) has exactly 1 call + 1 result
        assert calls is not None and len(calls) == 1
        assert len(tool_msgs) == 1
        assert calls[0]["id"] == tool_msgs[0]["tool_call_id"]


def test_spawn_emits_sub_agents_per_spawn_dispatch_calls_not_parallel_knob() -> None:
    # A spawn event emits exactly sub_agents_per_spawn parallel dispatch_agent calls
    # (one per child) in a SINGLE assistant output -- mirroring how a real harness
    # emits N Agent tool_calls at once. This spawn WIDTH is governed by
    # sub_agents_per_spawn, NOT parallel_tool_calls_per_step: with parallel fixed 3
    # but sub_agents_per_spawn fixed 2, each spawn advertises exactly 2 dispatch
    # calls (the knob does not leak into the spawn width).
    cfg = _cfg(
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=3),
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    spawn_events = [ev for eid, ev in g.events.items() if eid.endswith(":spawn")]
    assert spawn_events, "fan-out spawn events materialized"
    # Matched precisely: the async tail has a `:dispatch_ack` orchestrator turn (the immediate post-dispatch "agents
    # are running" turn), which is NOT a headless per-child dispatch event.
    headless = [eid for eid in g.events if ":disp" in eid and not eid.endswith(":dispatch_ack")]
    assert not headless, f"no headless dispatch events remain, got {headless}"
    for ev in spawn_events:
        # The spawn's EXPECTED output is K parallel dispatch_agent calls (K=2 here).
        assert ev.call.expected_output_is_tool_call is True
        assert ev.call.expected_output_tool_names == ["dispatch_agent", "dispatch_agent"], (
            f"spawn width should equal sub_agents_per_spawn (2), got {ev.call.expected_output_tool_names}"
        )


def test_parallel_tool_calls_preserves_determinism() -> None:
    cfg = _cfg(
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=1)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=1)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


# --- Bare non-agentic baseline (tool_catalog_size_per_agent=0) ---------


def test_zero_tool_definitions_is_bare_baseline() -> None:
    # tool_catalog_size_per_agent=0 -> NO tools advertised at all, and a
    # catalog-less agent cannot emit a forced tool call, so it just answers.
    cfg = _cfg(
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
        tool_loop_depth=Distribution(type="fixed", mean=2),
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


# --- Round-to-round context growth -------------------------------------


def _principal_events_by_round(g: ReplayGraph) -> Any:
    """Map round index -> the root principal event for that round."""
    import re

    out = {}
    for eid, ev in g.events.items():
        m = re.match(r"synthN\d+:r(\d+):principal$", eid)
        if m:
            out[int(m.group(1))] = ev
    return out


def test_interactive_rounds_carry_growing_context() -> None:
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=1),
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


def test_round_k_survives_runtime_substitution() -> None:
    # Build a 3-round session, then run the round-2 principal event through the
    # ACTUAL runtime substitution (_build_messages_with_substitution) with a
    # registry populated for its predecessors — mirroring the tool_output tests.
    from inference_perf.datagen.replay_graph_session_datagen import (
        EventOutputRegistry,
        SessionChatCompletionAPIData,
        WorkerSessionTracker,
    )

    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    principals = _principal_events_by_round(g)
    target = principals[2]
    shared_seg = target.call.input_segments[0]
    output_seg = target.call.input_segments[1]

    # Both the shared and the output segment source the prior round's TERMINAL event:
    # shared re-injects its full input (the whole tool loop), output re-injects its
    # answer. So they reference the SAME event id, registered once with both its input
    # messages (for the shared slice) and its output_message (for the output segment).
    assert shared_seg.source_event_id == output_seg.source_event_id, "shared+output both source the terminal"
    prior_terminal = g.events[shared_seg.source_event_id]

    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    prior_answer_text = "ROUND-1 ANSWER TEXT MARKER"
    registry.record(
        prior_terminal.event_id,
        prior_answer_text,
        messages=list(prior_terminal.call.messages),  # its full input -> the growing prefix
        output_message={"role": "assistant", "content": prior_answer_text},  # its answer
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


def test_interactive_rounds_preserve_determinism() -> None:
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=4)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=4)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments


# --- Forced/emitted tool names must appear in tool_definitions -------------


def _event_def_names(ev: GraphEvent) -> Any:
    """Top-level tool_definitions names advertised on an event."""
    return {td["name"] for td in (ev.call.tool_definitions or []) if "name" in td}


def _event_tool_call_names(ev: GraphEvent) -> Any:
    """Tool names appearing in this event's stored assistant tool_calls."""
    names = set()
    for m in ev.call.messages:
        for tc in m.get("tool_calls", []) or []:
            names.add(tc["function"]["name"])
    return names


def _assert_inv2_over_graph(g: ReplayGraph) -> None:
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


def test_dispatch_agent_is_in_tool_definitions() -> None:
    # fanout forced, normal catalog: every event that forces a tool or stores a
    # tool_call must advertise that tool (inv #2). Specifically the spawn event
    # must both FORCE dispatch_agent (K times) and ADVERTISE it, so replay's
    # tool_choice forcing does not silently degrade to "required".
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_loop_depth=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)

    spawn_events = [ev for eid, ev in g.events.items() if eid.endswith(":spawn")]
    assert spawn_events, "fan-out spawn events materialized"
    for ev in spawn_events:
        # spawn forces K dispatch_agent calls (all named dispatch_agent).
        assert ev.call.expected_output_tool_names == ["dispatch_agent", "dispatch_agent"], (
            "spawn event forces sub_agents_per_spawn dispatch_agent calls"
        )
        assert "dispatch_agent" in _event_def_names(ev), "dispatch_agent advertised in spawn event tool_definitions"

    # each notification event carries the dispatch_agent calls in its message
    # history (the spawn's assistant reply) -> inv #2 applies there too.
    notify_events = [ev for eid, ev in g.events.items() if ":notify" in eid]
    assert notify_events, "fan-out notification events materialized"
    for ev in notify_events:
        assert "dispatch_agent" in _event_tool_call_names(ev), "notification carries the dispatch_agent calls"
        assert "dispatch_agent" in _event_def_names(ev), "dispatch_agent advertised in notification tool_definitions"


def test_dispatch_agent_present_even_with_empty_theme_catalog() -> None:
    # tool_catalog_size_per_agent=0 + fanout: theme catalog is empty, but the
    # dispatch tool is STRUCTURAL, so the spawn event must advertise exactly
    # [dispatch_agent] (not []).
    cfg = _cfg(
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_loop_depth=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)

    spawn_events = [ev for eid, ev in g.events.items() if eid.endswith(":spawn")]
    assert spawn_events, "fan-out spawn events materialized even with empty theme catalog"
    for ev in spawn_events:
        defs = ev.call.tool_definitions or []
        names = [td["name"] for td in defs if "name" in td]
        assert names == ["dispatch_agent"], f"expected exactly [dispatch_agent], got {names}"


def test_no_dispatch_agent_when_no_fanout() -> None:
    # fanout_probability=0.0: single-agent catalogs stay clean -- no
    # dispatch_agent advertised anywhere.
    cfg = _cfg(
        fanout_probability=0.0,
        tool_loop_depth=Distribution(type="fixed", mean=2),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    _assert_inv2_over_graph(g)
    for ev in g.events.values():
        assert "dispatch_agent" not in _event_def_names(ev), f"{ev.event_id} advertised dispatch_agent without fan-out"


def test_inv2_holds_across_fanout_graph() -> None:
    # GENERAL inv #2 regression across a deeper fan-out graph.
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_loop_depth=Distribution(type="fixed", mean=1),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(g.events) > 4, "fan-out actually materialized"
    _assert_inv2_over_graph(g)


# --- Result-content fidelity: per-tool templates, no placeholder leakage ---


def _find_ordinary_tool_result_msgs(g: ReplayGraph) -> Any:
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


def test_tool_result_uses_per_tool_template() -> None:
    # db2 theme's get_bp_stats template is rich ("| time | bp | hit_ratio |"
    # table markers) and distinct from the generic 'default' template. Force
    # a small catalog (tool_catalog_size_per_agent=1) so the single advertised
    # tool is theme.tool_names[0] == "get_bp_stats" (per _tool_definitions'
    # cycling), guaranteeing every ordinary tool-turn call is get_bp_stats.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
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


def test_tool_result_no_literal_placeholders() -> None:
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
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


def test_tool_result_content_is_deterministic() -> None:
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, theme, _WordTok(), session_index=7)
    g2 = build_graph_for_session(cfg, theme, _WordTok(), session_index=7)
    r1 = _find_ordinary_tool_result_msgs(g1)
    r2 = _find_ordinary_tool_result_msgs(g2)
    assert r1 == r2, "tool-result contents are not deterministic for the same (config, index)"


def test_generated_fanout_session_has_no_dangling_tool_call_ids() -> None:
    """Build a fan-out session and walk every event's messages; assert no
    role:tool message references a tool_call_id absent from a preceding
    assistant tool_call in the SAME event. This is the exact invariant whose
    violation caused the live IndexError/dangling-id class of bug."""
    cfg = _cfg(
        fanout_probability=1.0,
        max_depth=2,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_loop_depth=Distribution(type="fixed", mean=1),
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
def test_theme_mix_rejected(theme_mix: Any) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix=theme_mix)


def test_theme_mix_valid_accepted() -> None:
    # Regression guard: a normal, non-empty, positive-weight mix must still
    # construct without raising.
    cfg = _cfg(theme_mix={"generic": 0.5, "db2_latency_incident": 0.5})
    assert cfg.theme_weights() == {"generic": 0.5, "db2_latency_incident": 0.5}


def test_theme_mix_accepts_both_shapes_equivalently() -> None:
    # theme_mix accepts BOTH the bare float {name: W} and the explicit weight
    # block {name: {weight: W}}; the two normalize to the same weights and, for
    # a given seed, pick the same theme per session. A mix of both forms in one
    # config is allowed too.
    bare = _cfg(theme_mix={"generic": 0.25, "db2_latency_incident": 0.75})
    block = _cfg(theme_mix={"generic": {"weight": 0.25}, "db2_latency_incident": {"weight": 0.75}})
    mixed = _cfg(theme_mix={"generic": 0.25, "db2_latency_incident": {"weight": 0.75}})
    assert (
        bare.theme_weights()
        == block.theme_weights()
        == mixed.theme_weights()
        == {
            "generic": 0.25,
            "db2_latency_incident": 0.75,
        }
    )
    # identical weighted draws across sessions (same seed -> same theme choices)
    from inference_perf.config.datagen.config import DataConfig, DataGenType
    from inference_perf.datagen.synthetic_agentic import SyntheticAgenticDataGenerator

    gb = SyntheticAgenticDataGenerator(
        api_config=_min_api(),
        config=DataConfig(type=DataGenType.SyntheticAgentic, synthetic_agentic=bare),
        tokenizer=_word_tok(),
        num_workers=1,
    )
    gk = SyntheticAgenticDataGenerator(
        api_config=_min_api(),
        config=DataConfig(type=DataGenType.SyntheticAgentic, synthetic_agentic=block),
        tokenizer=_word_tok(),
        num_workers=1,
    )
    picks_b = [gb._pick_theme(i).name for i in range(20)]
    picks_k = [gk._pick_theme(i).name for i in range(20)]
    assert picks_b == picks_k, "bare and weight-block forms must pick the same themes per session"


def test_theme_mix_weight_block_rejects_negative() -> None:
    # A negative weight in the explicit block form is rejected at the submodel.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg(theme_mix={"generic": {"weight": -1.0}})


@pytest.mark.parametrize(
    "kwargs, should_raise",
    [
        # overrun: shared prompt head alone exceeds the cap -> fail-fast.
        ({"max_model_len": 1000, "shared_system_prompt_len": 2000}, True),
        # None: no ceiling configured -> no fail-fast check performed, even for
        # a config whose peak clearly would overrun.
        ({"max_model_len": None, "input_tokens_per_turn": Distribution(type="fixed", mean=500_000)}, False),
        # comfortable: small everything, generous ceiling.
        ({"max_model_len": 200_000}, False),
        # CATALOG is counted: a huge catalog alone (2000 tools x ~380 = ~760K)
        # overruns a 131K cap even though every token knob is tiny. The old
        # one-turn check missed this -- it never counted the catalog.
        ({"max_model_len": 131_072, "tool_catalog_size_per_agent": Distribution(type="fixed", mean=2000)}, True),
        # ...same catalog, ample ceiling -> accepted (the catalog isn't rejected
        # per se; only when it makes the PEAK overrun).
        ({"max_model_len": 1_000_000, "tool_catalog_size_per_agent": Distribution(type="fixed", mean=2000)}, False),
        # OUTPUT is counted: tiny input, but a uniform output whose clip ceiling
        # (max) is enormous -> peak includes that output and overruns.
        (
            {
                "max_model_len": 131_072,
                "input_tokens_per_turn": Distribution(type="fixed", mean=100),
                "output_tokens_per_turn": Distribution(type="uniform", min=5, max=200_000),
                "tool_loop_depth": Distribution(type="fixed", mean=0),
            },
            True,
        ),
        # MULTI-TURN accumulation is counted: a per-turn size that fits alone
        # overruns once many turns accumulate (turns x per_turn_loop).
        (
            {
                "max_model_len": 131_072,
                "input_tokens_per_turn": Distribution(type="fixed", mean=30_000),
                "output_tokens_per_turn": Distribution(type="fixed", mean=100),
                "tool_loop_depth": Distribution(type="fixed", mean=0),
                "turns_per_session": Distribution(type="fixed", mean=10),  # 10 x 30k = 300k
            },
            True,
        ),
        # TYPE-AWARE worst case, fixed: a fixed distribution uses its `mean`
        # (its `max` is the stale 1024 default and must be IGNORED). mean 50k
        # overruns a 40k cap.
        (
            {
                "max_model_len": 40_000,
                "input_tokens_per_turn": Distribution(type="fixed", mean=50_000),
                "output_tokens_per_turn": Distribution(type="fixed", mean=100),
                "tool_loop_depth": Distribution(type="fixed", mean=0),
            },
            True,
        ),
        # TYPE-AWARE worst case, uniform: a uniform uses its `max` (its `mean` is
        # the stale 512 default and must be IGNORED). Here max is small (100), so
        # despite a large stale mean the config FITS -- proving we don't read the
        # bogus 512.
        (
            {
                "max_model_len": 50_000,
                "input_tokens_per_turn": Distribution(type="uniform", min=10, max=100),
                "output_tokens_per_turn": Distribution(type="fixed", mean=100),
                "tool_loop_depth": Distribution(type="fixed", mean=0),
            },
            False,
        ),
    ],
    ids=[
        "head_overrun",
        "none_never_checks",
        "comfortable_fit",
        "catalog_counted_reject",
        "catalog_counted_accept",
        "output_counted",
        "multiturn_accumulation_counted",
        "fixed_uses_mean_not_default_max",
        "uniform_uses_max_not_default_mean",
    ],
)
def test_max_model_len_fail_fast(kwargs: Any, should_raise: bool) -> None:
    from pydantic import ValidationError

    if should_raise:
        with pytest.raises(ValidationError):
            _cfg(**kwargs)
    else:
        cfg = _cfg(**kwargs)
        assert cfg.max_model_len == kwargs["max_model_len"]


def test_max_model_len_counts_tool_loop_depth() -> None:
    # The tool loop's transcript grows each iteration, so a DEEP loop is part of
    # the peak request. A config that fits at loop depth 0 must be rejected at a
    # large enough loop depth, all else equal -- proving the loop term is in the
    # projection (and that a single-shot sub-agent's own loop is sized, not just
    # the root's multi-turn accumulation).
    from pydantic import ValidationError

    common = dict(
        max_model_len=131_072,
        input_tokens_per_turn=Distribution(type="fixed", mean=5_000),
        output_tokens_per_turn=Distribution(type="fixed", mean=5_000),
        turns_per_session=Distribution(type="fixed", mean=1),
    )
    # loop depth 0: peak ~ head + catalog + input + output, well under 131K.
    cfg = _cfg(tool_loop_depth=Distribution(type="fixed", mean=0), **common)
    assert cfg.max_model_len == 131_072
    # loop depth 12: per_turn_loop ~ 5000 + 12*(5000+5000) = 125000, + output
    # pushes the peak over 131K -> rejected.
    with pytest.raises(ValidationError):
        _cfg(tool_loop_depth=Distribution(type="fixed", mean=12), **common)


def test_max_model_len_message_has_breakdown() -> None:
    # The rejection message itemises the peak so a user can see which knob to
    # cut (catalog, turns, loop, input, output).
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as ei:
        _cfg(max_model_len=1000, tool_catalog_size_per_agent=Distribution(type="fixed", mean=500))
    msg = str(ei.value)
    assert "peak request" in msg
    assert "tool_catalog" in msg
    assert "per_turn_loop" in msg


# --- Event-model fix: each call carries the cumulative transcript ----------
#
# Every event is exactly ONE LLM call whose INPUT is the growing conversation
# transcript ending in a user or tool message; the assistant reply is the
# event's OUTPUT (expected_output), NOT a separate lone-assistant event.


def _last_role(ev: GraphEvent) -> Any:
    """Role of the last message in an event's input transcript."""
    return ev.call.messages[-1].get("role") if ev.call.messages else None


def _is_lone_assistant(ev: GraphEvent) -> Any:
    """True iff the event's input is a single assistant message. No well-formed
    event should look like this (every event's input ends in a user/tool message)."""
    msgs = ev.call.messages
    return len(msgs) == 1 and msgs[0].get("role") == "assistant"


def test_no_lone_assistant_input() -> None:
    # THE core assertion. Across every shape (bare, tool-loop, interactive,
    # fan-out), NO event's input is a lone assistant message, and EVERY event's
    # input ends in role 'user' or 'tool' -- never 'assistant'.
    shapes = {
        "bare": _cfg(
            tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
            tool_loop_depth=Distribution(type="fixed", mean=2),
            fanout_probability=0.0,
        ),
        "tool_loop": _cfg(
            tool_loop_depth=Distribution(type="fixed", mean=3),
            fanout_probability=0.0,
        ),
        "interactive": _cfg(
            turns_per_session=Distribution(type="fixed", mean=3),
            tool_loop_depth=Distribution(type="fixed", mean=1),
            fanout_probability=0.0,
            max_events_per_session=2048,
        ),
        "fanout": _cfg(
            fanout_probability=1.0,
            max_depth=2,
            sub_agents_per_spawn=Distribution(type="fixed", mean=2),
            max_events_per_session=2048,
            tool_loop_depth=Distribution(type="fixed", mean=1),
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


def test_bare_single_round_is_one_event() -> None:
    # rounds=1, k=0 (empty catalog), fanout 0 -> EXACTLY 1 event. With
    # shared_system_prompt_len=0 (head-less baseline) its input is [user]; its
    # expected_output is the (non-empty) answer text; it is NOT a tool call.
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=0.0,
        shared_system_prompt_len=0,  # explicit head-less baseline (default is now 1000)
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(g.events) == 1, f"expected exactly 1 event, got {sorted(g.events)}"
    ev = next(iter(g.events.values()))
    roles = [m.get("role") for m in ev.call.messages]
    assert roles == ["user"], f"bare principal input should be [user], got {roles}"
    assert ev.call.expected_output_is_tool_call is False
    assert ev.call.expected_output, "terminal answer text must be non-empty"

    # With a system prompt (the default), the input is [system, user].
    cfg_sys = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=0.0,
        shared_system_prompt_len=16,
    )
    gs = build_graph_for_session(cfg_sys, GENERIC_THEME, _WordTok(), session_index=0)
    assert len(gs.events) == 1
    evs = next(iter(gs.events.values()))
    assert [m.get("role") for m in evs.call.messages] == ["system", "user"]


def test_shared_system_prompt_len_defaults_to_nonzero() -> None:
    # Virtually every agentic flow ships a system prompt, so the DEFAULT head is
    # non-zero (1000): a config that does not set shared_system_prompt_len still
    # opens each agent call with a system message.
    cfg = SyntheticAgenticConfig(
        num_sessions=1,
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
    )
    assert cfg.shared_system_prompt_len == 1000
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    principal = g.events["synthN0:r0:principal"]
    assert principal.call.messages[0].get("role") == "system", "default config opens with a system head"


def test_render_system_head_fits_truncates_and_is_deterministic() -> None:
    # _render_system_head keeps a real role-appropriate prompt FIRST and fits the
    # requested length: pads with a labeled block when short, truncates the prompt
    # (no filler header) when the prompt alone exceeds the target, and is
    # deterministic given the rng.
    import numpy as np
    from inference_perf.datagen.synthetic_agentic import _render_system_head
    from inference_perf.datagen.synthetic_themes import ROOT_SYSTEM_PROMPTS, SUBAGENT_SYSTEM_PROMPTS

    tok = _WordTok()
    # _WordTok counts words; the real prompts are ~430-540 words, so a target
    # ABOVE the longest exercises the pad path and one BELOW the shortest
    # exercises the truncate path.

    # large target (> longest prompt): real ROOT prompt + labeled filler block, ~= target
    big = _render_system_head(tok, 800, is_root=True, rng=np.random.default_rng(0))
    assert "## Operational context" in big, "short prompt padded with a labeled filler block"
    assert big.split("## Operational context")[0].strip() in {
        p.split("## Operational context")[0].strip() for p in ROOT_SYSTEM_PROMPTS
    }
    assert abs(tok.count_tokens(big) - 800) <= 5, "fitted head lands near the target length"

    # tiny target < prompt length: truncated, NO filler header, never exceeds target
    tiny = _render_system_head(tok, 8, is_root=False, rng=np.random.default_rng(0))
    assert "## Operational context" not in tiny, "truncated head has no filler block"
    assert tok.count_tokens(tiny) <= 8, "truncated head does not exceed the target"
    assert any(tiny.split()[0] == p.split()[0] for p in SUBAGENT_SYSTEM_PROMPTS), (
        "truncated head keeps the real prompt's opening"
    )

    # determinism: same rng seed -> identical head (pad path)
    a = _render_system_head(tok, 700, is_root=True, rng=np.random.default_rng(3))
    b = _render_system_head(tok, 700, is_root=True, rng=np.random.default_rng(3))
    assert a == b


def test_tool_loop_context_grows() -> None:
    # single-agent k=3, fanout 0 -> the agent's events' input message counts
    # grow like the OTel reference / real Exgentic (1, 3, 5, 7 for k=3, ignoring
    # any system head). principal + t0 + t1 + t2 = 4 events.
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        shared_system_prompt_len=0,  # isolate loop growth from the head (default is now 1000)
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    # k+1 = 4 events for one round.
    assert len(g.events) == 4, f"expected principal + 3 tool turns = 4 events, got {sorted(g.events)}"
    # Order by id suffix: principal, t0, t1, t2.
    ordered = sorted(g.events.values(), key=lambda e: (0 if e.event_id.endswith(":principal") else 1, e.event_id))
    lengths = [len(e.call.messages) for e in ordered]
    # Non-terminal turns grow by 2 (prior assistant tool-call + tool result). The TERMINAL
    # turn (t2) grows by 3: the +2 accumulation PLUS one trailing ROOT_ANSWER_DIRECTIVE
    # message that steers the final turn to prose instead of tool-call text -> 1,3,5,8.
    assert lengths == sorted(lengths), f"input lengths not monotonic: {lengths}"
    assert lengths[0] == 1, f"principal input should be [user] (1 msg), got {lengths[0]}"
    for a, b in zip(lengths[:-1], lengths[1:-1], strict=False):
        assert b - a == 2, f"non-terminal tool loop should grow by 2 per turn, got {lengths}"
    assert lengths == [1, 3, 5, 8], f"expected 1,3,5,8 (terminal +1 for the answer nudge), got {lengths}"


def _drive_substitution(target_ev: GraphEvent, prior_by_source: Any) -> Any:
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


def _ordered_agent_events(g: ReplayGraph, agent_prefix: str) -> Any:
    """Return an agent's events in build order (principal first, then t0, t1...)."""
    import re

    evs = [ev for eid, ev in g.events.items() if eid.startswith(agent_prefix + ":")]

    def _key(ev: GraphEvent) -> Any:
        eid = ev.event_id
        if eid.endswith(":principal"):
            return (0, 0)
        m = re.search(r":t(\d+)$", eid)
        if m:
            return (1, int(m.group(1)))
        return (2, eid)

    return sorted(evs, key=_key)


def test_substitution_survives_all_shapes() -> None:
    # Drive tool-loop events AND a fan-out merge through the REAL substitution
    # with a populated registry: no IndexError, transcript reconstructs, prior
    # turns are present.
    # --- tool loop --- (head-less so accumulation math stays literal)
    cfg = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=1),
        fanout_probability=0.0,
        shared_system_prompt_len=0,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    ordered = _ordered_agent_events(g, "synthN0:r0")
    # Walk the chain, simulating live outputs, feeding each event's rebuilt
    # input (its replay `messages`) forward into the registry for the next.
    live_inputs: Dict[str, Any] = {}
    for ev in ordered:
        prior_by_source: Dict[str, Any] = {}
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
    # the last (terminal) event's rebuilt input carries the whole growing loop (1+2*3=7)
    # PLUS one trailing ROOT_ANSWER_DIRECTIVE message on the terminal turn -> 8.
    assert len(live_inputs[ordered[-1].event_id]) == 8, "terminal tool-loop input did not accumulate to 1+2*3+1=8"

    # --- fan-out async notifications ---
    fcfg = _cfg(
        fanout_probability=1.0,
        max_depth=1,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_events_per_session=2048,
        tool_loop_depth=Distribution(type="fixed", mean=1),
    )
    fg = build_graph_for_session(fcfg, GENERIC_THEME, _WordTok(), session_index=0)
    notifies = [ev for eid, ev in fg.events.items() if ":notify" in eid]
    assert notifies, "fan-out notification events exist"
    for notif in notifies:
        prior_by_source = {}
        for seg in notif.call.input_segments:
            if seg.source_event_id is None:
                continue
            src = seg.source_event_id
            if seg.type == "output":
                # spawn event -> live dispatch tool calls. The notification's stub
                # tool results are matched against these by the id-rewrite post-pass,
                # so supply exactly as many live calls as there are stub results.
                n_stubs = sum(1 for m in notif.call.messages if m.get("role") == "tool")
                out_msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"live_{src}_{i}",
                            "type": "function",
                            "function": {"name": "dispatch_agent", "arguments": "{}"},
                        }
                        for i in range(n_stubs)
                    ],
                }
                prior_by_source[src] = ([], out_msg)
            elif seg.type == "async_report":
                out_msg = {"role": "assistant", "content": f"CHILD-ANSWER-{src}"}
                prior_by_source[src] = ([], out_msg)
            elif seg.type == "shared":
                # the prefix source (the spawn event, or the prior notification in the
                # chain). It must supply EXACTLY seg.message_count messages, else the
                # runtime's length-mismatch guard falls back to the recorded prefix.
                prior_by_source[src] = (
                    [{"role": "user", "content": f"prefix-{i}"} for i in range(seg.message_count)],
                    None,
                )
        result = _drive_substitution(notif, prior_by_source)  # must not raise
        assert result, "notification substitution produced empty input"
        # the child's report text was injected into the user-role notification slot
        joined = " ".join(str(m.get("content", "")) for m in result)
        assert "CHILD-ANSWER-" in joined, "child report not injected into the notification slot"
        # and it landed on a USER message, not a tool message, wrapped in the
        # <task-notification><result> envelope a real async harness delivers
        report_msgs = [m for m in result if "CHILD-ANSWER-" in str(m.get("content", ""))]
        assert report_msgs, "no message carries the child report"
        for m in report_msgs:
            assert m["role"] == "user", "child report must arrive as a user-role notification"
            content = str(m["content"])
            assert content.startswith("<task-notification>"), "report must be wrapped in the notification envelope"
            assert content.endswith("</task-notification>")
            body = content.split("<result>\n", 1)[1].split("\n</result>", 1)[0]
            assert body.startswith("CHILD-ANSWER-"), "the <result> body is the child's report"


# --- Enrichment: tool descriptions, theme filler, intro doc ----------------


def test_both_themes_validate_with_new_fields() -> None:
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


def test_tool_definitions_carry_descriptions() -> None:
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


def test_tool_definitions_suffixed_duplicates_reuse_base_description() -> None:
    # Request MORE tools than the theme has: suffixed duplicates must be unique
    # and reuse their base tool's description.
    n = len(GENERIC_THEME.tool_names) + 3
    defs = _tool_definitions(GENERIC_THEME, n)
    names = [td["name"] for td in defs]
    assert len(names) == len(set(names)), "suffixed duplicate names must stay unique"
    base0 = GENERIC_THEME.tool_names[0]
    dup = next(td for td in defs if td["name"].startswith(base0 + "_"))
    assert dup["description"] == GENERIC_THEME.tool_descriptions[base0]


def test_theme_filler_words_are_domain_relevant_and_deterministic() -> None:
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


def test_theme_without_filler_returns_none() -> None:
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


def test_fit_filler_uses_theme_word_pool() -> None:
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


def test_intro_doc_rides_first_user_turn_and_is_deterministic() -> None:
    # The round-0 principal user turn carries the theme's long intro doc; it is
    # deterministic for a given (config, index) and preserved after filler.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
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


def test_intro_doc_no_placeholder_leak() -> None:
    # Rendered intro docs must fill every placeholder (no {..} leak) for both themes.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        doc = _render_intro_doc(theme, session_seed(42, 3), (0, 61))
        assert doc, f"{theme.name}: intro doc empty"
        assert "{" not in doc and "}" not in doc, f"{theme.name}: unfilled placeholder in intro doc: {doc!r}"


def test_only_round_zero_carries_intro_doc() -> None:
    # The intro doc opens the session once; later rounds are terse follow-ups.
    theme = load_theme("db2_latency_incident")
    cfg = _cfg(
        theme_mix={"db2_latency_incident": 1.0},
        turns_per_session=Distribution(type="fixed", mean=3),
        tool_loop_depth=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    g = build_graph_for_session(cfg, theme, _WordTok(), session_index=0)
    principals = _principal_events_by_round(g)

    def _user_content(ev: GraphEvent) -> Any:
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


def test_db2_hit_ratio_is_at_most_100() -> None:
    # Every hit_ratio in the get_bp_stats table (a ratio) must be <= 100.
    db2 = load_theme("db2_latency_incident")
    tpl = db2.result_templates["get_bp_stats"]
    out = _render_theme_template(db2, tpl, session_seed(42, 5), (0, 4))
    import re

    rows = re.findall(r"\|\s*\d{1,2}:\d{2}:\d{2}\s*\|\s*\d+\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", out)
    assert rows, f"no hit_ratio table rows parsed from: {out!r}"
    for r in rows:
        assert 0.0 <= float(r) <= 100.0, f"hit_ratio out of [0,100]: {r} in {out!r}"


def test_latency_ms_class_field_within_bound() -> None:
    # p50_ms / p99_ms in get_service_health are ms-class -> [1, 2000].
    tpl = GENERIC_THEME.result_templates["get_service_health"]
    out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, 0), (0, 1))
    import re

    for key in ("p50_ms", "p99_ms"):
        m = re.search(rf"\b{key}=(\d+)", out)
        assert m, f"{key} not rendered: {out!r}"
        val = int(m.group(1))
        assert 1 <= val <= 2000, f"{key} out of ms bound [1,2000]: {val}"


def test_status_code_class_is_realistic() -> None:
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
def test_render_is_deterministic(template: Any, seed: Any, path: Any) -> None:
    # Same (theme, seed, path) -> byte-identical render.
    a = _render_theme_template(GENERIC_THEME, template, session_seed(42, seed), path)
    b = _render_theme_template(GENERIC_THEME, template, session_seed(42, seed), path)
    assert a == b, "render not deterministic"


def test_no_bounded_value_exceeds_100_where_percent_signalled() -> None:
    # Sweep both themes' templates: wherever the literal text carries a `%` or
    # `hit_ratio`/`_pct` label immediately before a rendered number, that number
    # must be <= 100. Guards the "273% success rate" giveaway across all docs.
    import re

    db2 = load_theme("db2_latency_incident")
    pat = re.compile(r"(?:hit_ratio|_pct|_ratio)\s*[=|]?\s*([0-9]+(?:\.[0-9]+)?)")
    pct_suffix = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
    for theme in (GENERIC_THEME, db2):
        templates = list(theme.result_templates.values()) + list(theme.intro_doc_templates) + list(theme.filler_templates)
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


def _round0_user_content(g: ReplayGraph) -> Any:
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
def test_intro_doc_primary_matches_objective(theme_name: str, primary_category: str) -> None:
    # The round-0 principal turn (intro doc + objective) must name exactly ONE
    # value of the theme's primary category, plus exactly one symptom.
    theme = GENERIC_THEME if theme_name == "generic" else load_theme(theme_name)
    cfg = _cfg(
        theme_mix={theme_name: 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
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


def test_pinned_entity_coherence_is_deterministic() -> None:
    # Same (config, index) -> byte-identical round-0 turn (pinning is seeded).
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=3)
    assert _round0_user_content(g1) == _round0_user_content(g2)


def test_render_theme_template_honors_pinned_entity() -> None:
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


def test_in_use_never_exceeds_max() -> None:
    # search_logs renders "in_use={in_use0}/{max0}"; in_use must be <= max.
    import re

    tpl = GENERIC_THEME.result_templates["search_logs"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m = re.search(r"in_use=(\d+)/(\d+)", out)
        assert m, f"in_use/max pair not rendered: {out!r}"
        in_use, mx = int(m.group(1)), int(m.group(2))
        assert in_use <= mx, f"in_use {in_use} > max {mx} in {out!r}"


def test_in_use_le_max_in_intro_doc_and_filler() -> None:
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


def test_error_rate_pct_reads_low() -> None:
    # error_rate_pct is an error-rate percent -> low ([0, 15]), not [80, 100].
    import re

    tpl = GENERIC_THEME.result_templates["get_service_health"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m = re.search(r"error_rate_pct=([0-9]+(?:\.[0-9]+)?)", out)
        assert m, f"error_rate_pct not rendered: {out!r}"
        val = float(m.group(1))
        assert 0.0 <= val <= 15.0, f"error rate not low: {val} in {out!r}"


def test_err_rate_and_err_pct_read_low() -> None:
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
# Regression guard for the truncated-tool-call 400: forced tool-call events used to ship
# with expected_output_tokens=0, so a real model truncated its tool call mid-JSON
# and leaked chat-template control tokens into `arguments`, which 400s on replay.
# Each forced tool-call event must be sized to tokens(json.dumps(its calls)) +
# TOOL_CALL_MARGIN; plain-answer events keep their sampled output size.


def _forced_and_answer_events(g: ReplayGraph) -> Any:
    """Split a graph's events into (forced-tool-call events, plain-answer events)."""
    forced = [ev for ev in g.events.values() if ev.call.expected_output_is_tool_call]
    answers = [ev for ev in g.events.values() if not ev.call.expected_output_is_tool_call]
    return forced, answers


def test_tool_call_max_tokens_helper() -> None:
    import json as _json

    tok = _WordTok()
    assert _tool_call_max_tokens(tok, []) == TOOL_CALL_MARGIN  # no calls -> margin floor
    calls = [{"id": "c0", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}]
    expected = tok.count_tokens(_json.dumps(calls)) + TOOL_CALL_MARGIN
    assert _tool_call_max_tokens(tok, calls) == expected
    assert _tool_call_max_tokens(tok, calls) > TOOL_CALL_MARGIN  # calls add on top of margin


def test_forced_tool_events_are_sized_not_zero() -> None:
    # Across every shape that forces tool calls, each forced event's
    # expected_output_tokens is >= TOOL_CALL_MARGIN, so the replay model has room
    # to emit the whole tool call.
    shapes = {
        "tool_loop": _cfg(tool_loop_depth=Distribution(type="fixed", mean=3), fanout_probability=0.0),
        "parallel": _cfg(
            tool_loop_depth=Distribution(type="fixed", mean=2),
            parallel_tool_calls_per_step=Distribution(type="fixed", mean=3),
            fanout_probability=0.0,
        ),
        "fanout": _cfg(
            fanout_probability=1.0,
            max_depth=2,
            sub_agents_per_spawn=Distribution(type="fixed", mean=2),
            max_events_per_session=2048,
            tool_loop_depth=Distribution(type="fixed", mean=1),
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


def test_forced_tool_events_sized_from_their_own_calls() -> None:
    # The budget equals tokens(json.dumps(the calls this event outputs)) + margin.
    # Reconstruct each forced event's calls from its OWN or its successor's stored
    # tool_calls and check the size matches.
    import json as _json

    tok = _WordTok()
    cfg = _cfg(tool_loop_depth=Distribution(type="fixed", mean=3), fanout_probability=0.0)
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


def test_answer_events_keep_output_tokens_not_tool_budget() -> None:
    # Plain-answer terminal events keep the sampled output_tokens_per_turn, NOT
    # the tool-call sizing. With output_tokens_per_turn fixed at 40, the terminal
    # answer event of a tool loop is 40 (its output IS the plain answer).
    cfg = _cfg(
        tool_loop_depth=Distribution(type="fixed", mean=2),
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


def test_forced_tool_sizing_is_deterministic() -> None:
    cfg = _cfg(
        tool_loop_depth=Distribution(type="fixed", mean=3),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=2),
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


def _defs_by_name(ev: GraphEvent) -> Any:
    """Map advertised tool name -> its tool_definition dict for an event."""
    return {td["name"]: td for td in (ev.call.tool_definitions or []) if "name" in td}


def _emitted_tool_calls(ev: GraphEvent) -> Any:
    """Yield (call_name, parsed_args_dict) for every stored assistant tool_call."""
    for m in ev.call.messages:
        for tc in m.get("tool_calls", []) or []:
            yield tc["function"]["name"], _json.loads(tc["function"]["arguments"])


def _cfg_all_tools(theme: Theme, **kw: Any) -> SyntheticAgenticConfig:
    """A config that advertises MORE tool defs than the theme has base tools, so
    every base tool AND at least one suffixed duplicate appears in the catalog.
    Uses a multi-turn, multi-parallel tool loop so many distinct tools are
    actually called."""
    n = len(theme.tool_names) + 2  # forces >=1 suffixed duplicate
    base = dict(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=n),
        tool_loop_depth=Distribution(type="fixed", mean=n),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
        max_events_per_session=2048,
    )
    base.update(kw)
    return _cfg(**base)


def test_every_advertised_tool_def_has_nonempty_params_and_required() -> None:
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


def test_suffixed_duplicate_reuses_base_param_schema() -> None:
    # A synthetic suffixed duplicate (get_bp_stats_10) must reuse its base
    # tool's parameter schema, not the generic fallback.
    theme = load_theme("db2_latency_incident")
    n = len(theme.tool_names) + 2
    defs = _tool_definitions(theme, n)
    base0 = theme.tool_names[0]
    base_params = theme.tool_parameters[base0]
    dup = next(td for td in defs if td["name"].startswith(base0 + "_"))
    assert dup["function"]["parameters"] == base_params, "suffixed duplicate did not reuse base param schema"


def test_emitted_tool_call_args_conform_to_advertised_schema() -> None:
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


@pytest.mark.parametrize(
    "theme, multi, min_required, cfg_builder",
    [
        # db2 get_bp_stats requires (db_instance, bufferpool); force a catalog whose
        # sole tool is get_bp_stats (tool_names[0]).
        (
            load_theme("db2_latency_incident"),
            "get_bp_stats",
            2,
            lambda theme: _cfg(
                theme_mix={theme.name: 1.0},
                turns_per_session=Distribution(type="fixed", mean=1),
                tool_catalog_size_per_agent=Distribution(type="fixed", mean=1),
                tool_loop_depth=Distribution(type="fixed", mean=3),
                parallel_tool_calls_per_step=Distribution(type="fixed", mean=2),
                fanout_probability=0.0,
                max_events_per_session=2048,
            ),
        ),
        # generic query_metrics has 3 required (metric, service, window); advertise
        # the whole catalog so it is reachable and called during the loop.
        (
            GENERIC_THEME,
            "query_metrics",
            3,
            lambda theme: _cfg_all_tools(theme, theme_mix={theme.name: 1.0}),
        ),
    ],
    ids=["db2_get_bp_stats", "generic_query_metrics"],
)
def test_multi_required_param_tool_emits_all_required_fields(
    theme: Any, multi: Any, min_required: Any, cfg_builder: Any
) -> None:
    # A multi-required-param tool, when called, emits ALL of its required fields.
    required = theme.tool_parameters[multi]["required"]
    assert len(required) >= min_required, f"test premise: {multi} is multi-required-param"
    g = build_graph_for_session(cfg_builder(theme), theme, _WordTok(), session_index=0)
    hits = 0
    for ev in g.events.values():
        for call_name, args in _emitted_tool_calls(ev):
            if call_name == multi:
                for req in required:
                    assert req in args, f"{multi} call missing {req!r}: {args}"
                hits += 1
    assert hits > 0, f"the multi-required-param tool {multi} was never called"


def test_emitted_args_are_deterministic() -> None:
    # Same (config, index) -> byte-identical emitted argument strings.
    for theme in (GENERIC_THEME, load_theme("db2_latency_incident")):
        cfg = _cfg_all_tools(theme, theme_mix={theme.name: 1.0})
        g1 = build_graph_for_session(cfg, theme, _WordTok(), session_index=6)
        g2 = build_graph_for_session(cfg, theme, _WordTok(), session_index=6)

        def _all_arg_strings(g: ReplayGraph) -> Any:
            out = []
            for eid in g.events:
                for m in g.events[eid].call.messages:
                    for tc in m.get("tool_calls", []) or []:
                        out.append((eid, tc["function"]["name"], tc["function"]["arguments"]))
            return out

        assert _all_arg_strings(g1) == _all_arg_strings(g2), f"{theme.name}: emitted args not deterministic"


def test_entity_named_param_threads_pinned_subject() -> None:
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


def test_both_themes_validate_with_tool_parameters() -> None:
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


def test_fallback_tool_params_applies_for_theme_without_schemas() -> None:
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
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=2),
        tool_loop_depth=Distribution(type="fixed", mean=2),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=1),
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


# --- ignore_eos must be False for forced tool-call turns (400 on replay fix) --
#
# The load default is ignore_eos=True (to make plain-text turns generate exactly
# N tokens). For a FORCED tool call that is wrong: with EOS ignored the model
# emits the call then keeps generating, spilling chat-template control tokens
# into `arguments` until max_tokens -> malformed JSON -> 400 on the replayed
# turn. to_request_body must force ignore_eos=False for every forced tool-call
# turn, regardless of override_tool_call_max_tokens.


def test_forced_tool_call_forces_ignore_eos_false() -> None:
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

    def _mk(is_tool_call: Any, override: Any) -> Any:
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
    payload = asyncio.run(forced.to_request_body(effective_model_name="m", max_tokens=64, ignore_eos=True, streaming=False))
    assert payload["ignore_eos"] is False, "forced tool call must send ignore_eos=False"

    # A plain-text turn that STILL advertises tools (this _mk passes tool_defs)
    # is the terminal/answer turn of a tool loop: it also gets ignore_eos=False
    # + tool_choice=none so it can't emit a dangling structured call or spill
    # template tokens. (A plain-text turn with NO tools keeps the caller
    # ignore_eos -- see test_plain_text_turn_without_tools_keeps_defaults.)
    plain = _mk(False, False)
    p2 = asyncio.run(plain.to_request_body(effective_model_name="m", max_tokens=64, ignore_eos=True, streaming=False))
    assert p2["ignore_eos"] is False, "plain-text-with-tools turn must stop cleanly (ignore_eos=False)"
    assert p2["tool_choice"] == "none", "plain-text-with-tools turn must forbid a structured tool call"


# --- Tool result echoes the call's arguments (coherence) --------------------
#
# A real tool answers about the entity it was called with. So a result template
# placeholder that matches a call-argument key (e.g. `{service}`) must resolve to
# the value THIS call passed, not an independent draw. Regression guard for the
# observed mismatch (call service=session-gateway, result service=inventory-svc).


def test_tool_result_echoes_call_service() -> None:
    import json as _json
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=8),
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


def test_plain_text_turn_with_tools_forbids_tool_call_and_stops() -> None:
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
    payload = asyncio.run(ev.to_request_body(effective_model_name="m", max_tokens=80, ignore_eos=True, streaming=False))
    assert payload["tool_choice"] == "none", "plain-text turn with tools must forbid a structured tool call"
    assert payload["ignore_eos"] is False, "plain-text turn with tools must stop cleanly (ignore_eos=False)"


def test_plain_text_turn_without_tools_keeps_defaults() -> None:
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
    payload = asyncio.run(ev.to_request_body(effective_model_name="m", max_tokens=80, ignore_eos=True, streaming=False))
    assert payload.get("tool_choice") is None, "plain text turn without tools must not set tool_choice"
    assert payload["ignore_eos"] is True, "plain text turn without tools keeps caller ignore_eos"


def test_fanout_children_pinned_to_parent_entity() -> None:
    # Fan-out coherence: every dispatched child's objective names the SAME primary
    # subject entity as the orchestrator (the fan-out is ONE investigation). The
    # verb may differ (children take different angles), but the service/db_instance
    # must match the parent's pinned subject.
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=1,
        tool_loop_depth=Distribution(type="fixed", mean=0),
        max_events_per_session=512,
    )

    def _service(text: Any) -> Any:
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
        # child objectives now live in each spawned sub-agent's principal user turn
        # (the child task the spawn hands off); read them from the sub principals.
        kids = []
        for eid, ev in g.events.items():
            if ":sub" in eid and eid.endswith(":principal"):
                users = [m["content"] for m in ev.call.messages if m["role"] == "user"]
                # the child objective is the FIRST user message (a trailing report-
                # directive nudge may follow it on a k=0 sub-agent).
                if users:
                    kids.append(_service(users[0]))
        assert orch is not None, f"session {idx}: no orchestrator service parsed"
        assert kids, f"session {idx}: no child objectives"
        for k in kids:
            assert k == orch, f"session {idx}: child service {k!r} != orchestrator {orch!r} (fan-out not coherent)"


def test_spawn_output_is_parallel_dispatch_calls_in_one_message() -> None:
    # Real-world fan-out shape: an agent spawns N sub-agents by emitting N
    # dispatch_agent tool_calls in a SINGLE assistant output (like Claude Code's
    # one call emitting 3 Agent tool_uses), NOT via N separate headless dispatch
    # events. Verify the spawn event's expected output is exactly K dispatch_agent
    # calls (K = sub_agents_per_spawn), and that NO headless dispatch node exists.
    K = 3
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=K),
        max_depth=1,
        tool_loop_depth=Distribution(type="fixed", mean=1),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    # The async tail's `:dispatch_ack` orchestrator turn is not one of those, so exclude it explicitly.
    assert not [eid for eid in g.events if ":disp" in eid and not eid.endswith(":dispatch_ack")]
    spawns = [ev for eid, ev in g.events.items() if eid.endswith(":spawn")]
    assert spawns, "at least one spawn event"
    for ev in spawns:
        assert ev.call.expected_output_is_tool_call is True
        assert ev.call.expected_output_tool_names == [DISPATCH_AGENT_NAME] * K, (
            f"spawn emits {K} parallel dispatch calls, got {ev.call.expected_output_tool_names}"
        )


def test_notifications_reconstruct_single_assistant_with_matched_stub_results() -> None:
    # Every notification event mirrors the real [assistant(N tool_calls), tool xN]
    # block: exactly ONE assistant message carrying N dispatch_agent calls, followed
    # by N role:tool results whose tool_call_ids are exactly those N call ids
    # (inv #3). Those results are now STATIC launch acks -- the child reports arrive
    # separately as user-role notifications -- so each ack carries ASYNC_DISPATCH_STUB
    # rather than a child's report.
    from inference_perf.datagen.synthetic_agentic import ASYNC_DISPATCH_STUB

    K = 3
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=K),
        max_depth=1,
        tool_loop_depth=Distribution(type="fixed", mean=0),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    notifies = [ev for eid, ev in g.events.items() if ":notify" in eid]
    assert len(notifies) == K, f"expected K={K} notification events, got {len(notifies)}"
    for ev in notifies:
        msgs = ev.call.messages
        assistants_with_calls = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
        # exactly ONE assistant carrying the N dispatch calls (a single block).
        assert len(assistants_with_calls) == 1, "notification has one N-call assistant block"
        calls = assistants_with_calls[0]["tool_calls"]
        assert len(calls) == K and all(c["function"]["name"] == DISPATCH_AGENT_NAME for c in calls)
        call_ids = [c["id"] for c in calls]
        tool_results = [m for m in msgs if m.get("role") == "tool"]
        tool_ids = [m.get("tool_call_id") for m in tool_results]
        assert tool_ids == call_ids, f"tool result ids {tool_ids} must match call ids {call_ids} (inv #3)"
        # the results are content-free static acks, NOT child reports
        for m in tool_results:
            assert m["content"] == ASYNC_DISPATCH_STUB, "dispatch result must be the static launch ack"


def test_fanout_graph_is_byte_identical_across_rebuilds() -> None:
    # Determinism under the new fan-out shape: same (config, index) -> byte-identical
    # graph (event ids, order, and every message).
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="uniform", min=2, max=3),
        max_depth=2,
        tool_loop_depth=Distribution(type="uniform", min=0, max=2),
        max_events_per_session=512,
    )
    for idx in range(3):
        g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        assert list(g1.events.keys()) == list(g2.events.keys())
        for eid in g1.events:
            assert g1.events[eid].call.messages == g2.events[eid].call.messages
            assert g1.events[eid].call.expected_output_tool_names == g2.events[eid].call.expected_output_tool_names


def test_subagent_terminal_ends_with_report_directive() -> None:
    # Every SUB-AGENT terminal (a leaf child's answer turn AND a spawning sub-agent's
    # LAST notification) ends with the summarize-report nudge (recency -> a PROSE
    # report, not tool-call text). The nudge is the LAST message and is a `user`
    # message; cursor math stays exact. Non-terminal child tool-turns and non-terminal
    # notifications must NOT end with it. (The ROOT's last notification ends with the
    # ANSWER directive, not this one -- covered separately.)
    from inference_perf.datagen.synthetic_agentic import SUBAGENT_REPORT_DIRECTIVE, ROOT_ANSWER_DIRECTIVE

    # depth 2 so there are BOTH leaf-child terminals AND sub-agent (non-root) merges.
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=2,
        tool_loop_depth=Distribution(type="fixed", mean=2),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    frag = SUBAGENT_REPORT_DIRECTIVE

    def ends_with(ev: GraphEvent, text: Any) -> Any:
        msgs = ev.call.messages
        return bool(msgs) and msgs[-1].get("role") == "user" and text in str(msgs[-1].get("content", ""))

    # A notification event is a terminal ONLY if it is the LAST link in its spawn's
    # chain; the earlier ones are ack turns and carry no directive. Identify the last
    # link per agent prefix by the highest :notifyN index.
    last_notify_ids = set()
    notify_by_prefix: Dict[str, Any] = {}
    for eid in g.events:
        if ":notify" not in eid:
            continue
        prefix, idx = eid.rsplit(":notify", 1)
        notify_by_prefix.setdefault(prefix, []).append(int(idx))
    for prefix, idxs in notify_by_prefix.items():
        last_notify_ids.add(f"{prefix}:notify{max(idxs)}")

    saw_child_terminal = False
    saw_subagent_last_notify = False
    saw_nonterminal_notify = False
    for eid, ev in g.events.items():
        is_child = ":sub" in eid
        is_notify = ":notify" in eid
        is_dispatch_ack = eid.endswith(":dispatch_ack")
        # An answer turn, not a tool-call turn -- and, for the async tail, only the LAST
        # notification link terminates the agent. The `:dispatch_ack` turn (the immediate
        # post-dispatch "agents are running" reply) is never a terminal: the reports have
        # not even arrived yet.
        is_terminal = (
            not ev.call.expected_output_is_tool_call and not is_dispatch_ack and (not is_notify or eid in last_notify_ids)
        )
        if (is_notify or is_dispatch_ack) and not is_terminal:
            saw_nonterminal_notify = True
        if is_child and is_terminal:
            saw_child_terminal = True
            if is_notify:
                saw_subagent_last_notify = True
            assert ends_with(ev, frag), f"sub-agent terminal {eid} must END with the report nudge"
            if ev.call.input_segments:  # cursor math stays exact after the appended message
                segsum = sum(s.message_count for s in ev.call.input_segments)
                assert segsum == len(ev.call.messages), f"{eid}: segment sum {segsum} != {len(ev.call.messages)}"
        elif not is_child and is_terminal and is_notify:
            # the ROOT's last notification ends with the ANSWER directive, not the report one.
            assert ends_with(ev, ROOT_ANSWER_DIRECTIVE), f"root terminal {eid} must end with the answer directive"
            assert not ends_with(ev, frag)
        else:
            # non-terminal child tool-turns AND non-terminal (ack) notifications must
            # NOT end with the report nudge.
            assert not ends_with(ev, frag), f"{eid} should NOT end with the sub-agent report nudge"
    assert saw_child_terminal, "expected >=1 sub-agent terminal turn"
    assert saw_subagent_last_notify, "expected >=1 non-root (sub-agent) notification terminal"
    assert saw_nonterminal_notify, "expected >=1 non-terminal (ack) notification"


def test_root_terminal_ends_with_answer_directive() -> None:
    # A root agent's TERMINAL turn (the turn that answers the USER) ends with the
    # ROOT_ANSWER_DIRECTIVE nudge, so the final message is prose, not tool-call text.
    # Applies to both a k>=1 tool loop's terminal and a k=0 answer-directly turn --
    # as long as the agent has a tool catalog (a no-tools agent can't emit tool-call
    # text, so it gets NO nudge). Non-terminal (tool-call) turns must NOT end with it.
    from inference_perf.datagen.synthetic_agentic import ROOT_ANSWER_DIRECTIVE

    frag = ROOT_ANSWER_DIRECTIVE

    def ends_with_nudge(ev: GraphEvent) -> Any:
        msgs = ev.call.messages
        return bool(msgs) and msgs[-1].get("role") == "user" and frag in str(msgs[-1].get("content", ""))

    # (a) k>=1 tool loop: only the terminal (answer) turn ends with the nudge.
    cfg_loop = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg_loop, GENERIC_THEME, _WordTok(), 0)
    for eid, ev in g.events.items():
        terminal = not ev.call.expected_output_is_tool_call
        assert ends_with_nudge(ev) == terminal, f"{eid}: nudge presence must match terminal={terminal}"
        if ev.call.input_segments:  # cursor math stays exact after the appended message
            segsum = sum(s.message_count for s in ev.call.input_segments)
            assert segsum == len(ev.call.messages), f"{eid}: segment sum {segsum} != {len(ev.call.messages)}"

    # (b) k=0 answer-directly turn WITH a catalog: the single principal ends with the nudge.
    cfg_k0 = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=0),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=8),
        fanout_probability=0.0,
    )
    g0 = build_graph_for_session(cfg_k0, GENERIC_THEME, _WordTok(), 0)
    ev0 = next(iter(g0.events.values()))
    assert ends_with_nudge(ev0), "k=0 root terminal with a catalog must end with the answer nudge"

    # (c) NO-tools root: no nudge (nothing to steer away from; keeps the bare turn clean).
    cfg_bare = _cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=0),
        fanout_probability=0.0,
    )
    gb = build_graph_for_session(cfg_bare, GENERIC_THEME, _WordTok(), 0)
    evb = next(iter(gb.events.values()))
    assert not ends_with_nudge(evb), "no-tools root terminal must NOT carry the answer nudge"

    # (d) deterministic
    g_again = build_graph_for_session(cfg_loop, GENERIC_THEME, _WordTok(), 0)
    for eid in g.events:
        assert g.events[eid].call.messages == g_again.events[eid].call.messages


def test_nonroot_last_notification_ends_with_report_directive() -> None:
    # In a recursive (depth-2) tree, a SPAWNING sub-agent's terminal is the LAST link
    # of its notification chain (it folds in grandchildren, then reports up to its
    # parent). That link must END with the report nudge (prose report at every
    # non-leaf level); the ROOT's last notification must NOT (its output is the
    # orchestrator's final answer). Earlier links in either chain are ack turns and
    # carry NO directive. Cursor math must stay exact after the appended message.
    from inference_perf.datagen.synthetic_agentic import SUBAGENT_REPORT_DIRECTIVE

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=2,  # children spawn grandchildren -> children's terminal is a notification
        tool_loop_depth=Distribution(type="fixed", mean=1),
        max_events_per_session=512,
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
    frag = SUBAGENT_REPORT_DIRECTIVE

    # group notification ids per spawn so we know which link is last
    notify_by_prefix: Dict[str, Any] = {}
    for eid in g.events:
        if ":notify" not in eid:
            continue
        prefix, idx = eid.rsplit(":notify", 1)
        notify_by_prefix.setdefault(prefix, []).append(int(idx))

    saw_nonroot = saw_root = saw_ack = False
    for prefix, idxs in notify_by_prefix.items():
        last_idx = max(idxs)
        # root chain: agent prefix is the bare round root (no ':sub' before ':dN')
        is_root_chain = ":sub" not in prefix.rsplit(":d", 1)[0]
        for idx in idxs:
            eid = f"{prefix}:notify{idx}"
            ev = g.events[eid]
            msgs = ev.call.messages
            ends = msgs[-1].get("role") == "user" and frag in str(msgs[-1].get("content", ""))
            # cursor math exact
            segsum = sum(s.message_count for s in ev.call.input_segments)
            assert segsum == len(msgs), f"{eid}: notification segment sum {segsum} != {len(msgs)}"
            if idx != last_idx:
                saw_ack = True
                assert not ends, f"non-terminal notification {eid} should NOT end with the report nudge"
            elif is_root_chain:
                saw_root = True
                assert not ends, f"root terminal {eid} should NOT end with the report nudge"
            else:
                saw_nonroot = True
                assert ends, f"non-root (child) terminal {eid} must END with the report nudge"
    assert saw_nonroot and saw_root, "expected both a root chain and >=1 non-root chain"
    assert saw_ack, "expected >=1 non-terminal (ack) notification"


def test_report_directive_deterministic() -> None:
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=2),
        max_depth=1,
        tool_loop_depth=Distribution(type="fixed", mean=3),
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


def _assert_template_render_clean(theme: Theme, theme_key: str, seed_idx: int = 0) -> None:
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


def test_research_rag_loads_and_validates() -> None:
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


def test_research_rag_templates_render_without_leak() -> None:
    t = load_theme("research_rag")
    _assert_template_render_clean(t, "research_rag")


def test_research_rag_session_builds_valid_and_deterministic() -> None:
    # Build a session with theme_mix {research_rag:1.0}: every emitted tool-call
    # arg parses as JSON, every tool result is brace-clean, deterministic per
    # (config, index).
    t = load_theme("research_rag")
    cfg = _cfg(
        theme_mix={"research_rag": 1.0},
        turns_per_session=Distribution(type="fixed", mean=2),
        tool_loop_depth=Distribution(type="fixed", mean=3),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=len(t.tool_names)),
        parallel_tool_calls_per_step=Distribution(type="fixed", mean=2),
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


def test_research_rag_search_hits_and_json_shapes() -> None:
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


def test_generic_stack_trace_and_json_shapes_render() -> None:
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


def test_db2_not_found_error_shape_renders() -> None:
    # db2's new get_message_log returns a not-found / connection-error payload.
    t = load_theme("db2_latency_incident")
    out = _render_theme_template(t, t.result_templates["get_message_log"], session_seed(42, 0), (0, 11))
    assert "{" not in out and "}" not in out, f"not-found brace leak: {out!r}"
    assert "ERROR" in out and "no messages" in out, f"not-found markers absent: {out!r}"


def test_new_result_shapes_are_deterministic() -> None:
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


def test_all_three_bundled_themes_still_load_and_validate() -> None:
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


def test_percentile_ordering_p50_le_p99_bare_ms_and_indexed() -> None:
    import re

    # (a) bare `_ms` siblings in a real template (get_service_health).
    tpl = GENERIC_THEME.result_templates["get_service_health"]
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        m50 = re.search(r"\bp50_ms=(\d+)", out)
        m99 = re.search(r"\bp99_ms=(\d+)", out)
        assert m50 is not None and m99 is not None, f"expected p50_ms/p99_ms in {out!r}"
        p50 = int(m50.group(1))
        p99 = int(m99.group(1))
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


def test_heap_used_le_heap_max() -> None:
    import re

    # heap_used{N} must be clamped to heap_max{N} (same pattern as in_use/max).
    tpl = "gc heap_used_mb={heap_used0} heap_max_mb={heap_max0} note={heap_used1}/{heap_max1}"
    for idx in range(30):
        out = _render_theme_template(GENERIC_THEME, tpl, session_seed(42, idx), (0, 1))
        pairs = re.findall(r"heap_used_mb=(\d+) heap_max_mb=(\d+)", out) + re.findall(r"note=(\d+)/(\d+)", out)
        assert pairs, f"heap pair not rendered: {out!r}"
        for used, mx in pairs:
            assert int(used) <= int(mx), f"heap_used {used} > heap_max {mx} in {out!r}"


def test_percentile_and_heap_render_no_placeholder_leak() -> None:
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


def test_connective_lowercases_common_first_word() -> None:
    from inference_perf.datagen.synthetic_agentic import _join_connective_case

    out = _join_connective_case("Following up, ", "Are other services in us-east-1 showing the same 5xx?", GENERIC_THEME)
    assert out.startswith("are other services"), f"common-word seam not fixed: {out!r}"
    # full join has no capital-after-lowercase-connective seam.
    joined = "Following up, " + out
    assert "Following up, are" in joined, f"casing seam remains: {joined!r}"


def test_connective_preserves_entity_and_acronym_first_word() -> None:
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


def test_generated_followups_have_no_casing_seam() -> None:
    # In a real multi-round session, no follow-up shows a capital letter right
    # after a lowercase connective+comma (unless it's a preserved proper noun).
    import re

    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=4),
        tool_loop_depth=Distribution(type="fixed", mean=1),
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
                after = real[pos + len(conn) :]
                first_tok = after.split(maxsplit=1)[0] if after.split() else ""
                if not first_tok or not first_tok[0].isupper():
                    continue  # already lowercased -> no seam
                # a leading capital is only OK if it's a preserved proper noun.
                is_entity = first_tok in entity_pool or after.startswith(tuple(entity_pool))
                is_acronym = first_tok.isupper()
                has_digit = any(c.isdigit() for c in first_tok)
                assert is_entity or is_acronym or has_digit, f"idx{idx} {eid}: casing seam after {conn!r}: {after[:50]!r}"


# --- Item-4 fix 3: region pinned across follow-ups --------------------------


def test_region_is_pinned_across_a_multi_round_session() -> None:
    # objective / intro doc / every follow-up must reference the SAME region.
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=4),
        tool_loop_depth=Distribution(type="fixed", mean=1),
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


def test_region_in_primary_categories_and_pinned() -> None:
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


def _code_change_cfg(**kw: Any) -> SyntheticAgenticConfig:
    base = dict(
        num_sessions=5,
        turns_per_session=Distribution(type="fixed", mean=2),
        fanout_probability=0.0,
        theme_mix={"code_change_task": 1.0},
        input_tokens_per_turn=Distribution(type="fixed", mean=40),
        output_tokens_per_turn=Distribution(type="fixed", mean=20),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=6),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=7),
    )
    base.update(kw)
    return SyntheticAgenticConfig(**base)


def test_code_change_task_loads_and_validates() -> None:
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


def _iter_tool_calls_and_results(g: ReplayGraph) -> Any:
    """Yield (kind, call_name, payload) for every emitted tool call and every
    role:tool result content across a graph."""
    for ev in g.events.values():
        for m in ev.call.messages:
            for tc in m.get("tool_calls", []) or []:
                yield "call", tc["function"]["name"], tc["function"]["arguments"]
            if m.get("role") == "tool":
                yield "result", None, str(m.get("content", ""))


def test_code_change_task_session_args_are_valid_json_and_results_have_no_leak() -> None:
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


def test_code_change_task_deterministic_per_config_and_index() -> None:
    t = load_theme("code_change_task")
    cfg = _code_change_cfg()
    g1 = build_graph_for_session(cfg, t, _WordTok(), session_index=3)
    g2 = build_graph_for_session(cfg, t, _WordTok(), session_index=3)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.expected_output == g2.events[eid].call.expected_output


def test_code_change_task_result_shapes_render_realistically() -> None:
    # run_tests -> traceback marker + pass/fail summary; git_diff -> unified diff
    # markers; read_file -> line-number formatting. Rendered directly so the test
    # does not depend on which tools a given session happens to schedule.
    from inference_perf.datagen.synthetic_agentic import _render_tool_result

    t = load_theme("code_change_task")
    seed = 4242

    run_tests = _render_tool_result(t, "run_tests", seed, (1, 2, 3))
    assert ("Traceback" in run_tests) or ("AttributeError" in run_tests) or ("AssertionError" in run_tests), run_tests
    assert "passed" in run_tests and "failed" in run_tests, run_tests

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


def test_write_tool_payload_args_are_sized_code_filler() -> None:
    # A write tool's big-payload arg (content/new_string/patch) is NOT the tiny
    # `{prop}-NNN` stub: it's a substantial chunk drawn from the theme filler pool.
    import json as _json

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=10),  # reach edit/write/apply (tools 8-10)
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=10),
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


def test_non_payload_string_arg_keeps_stub() -> None:
    # A non-payload string arg that is not an entity category (e.g. grep pattern)
    # keeps the short `{prop}-NNN` stub — payload sizing is scoped to payload names.
    import json as _json
    import re

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=8),
        fanout_probability=0.0,
    )
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


def test_tool_loop_varies_tools_across_turns() -> None:
    # 6a-i: a multi-turn loop uses >=2 distinct tools (not tool_defs[0] x k).
    import re

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=6),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=10),
        fanout_probability=0.0,
    )
    g = build_graph_for_session(cfg, t, _WordTok(), session_index=0)
    names = []
    for eid, ev in g.events.items():
        if re.search(r":t\d+$", eid) or eid.endswith(":principal"):
            for m in ev.call.messages:
                for tc in m.get("tool_calls", []) or []:
                    names.append(tc["function"]["name"])
    # dedup consecutive isn't enough; just assert variety across the loop
    assert len(set(names)) >= 2, f"loop used only one tool: {set(names)}"


def test_focus_entity_threads_across_the_loop() -> None:
    # 6a-ii: the file path referenced across a session's tool calls is ONE focus
    # value (coherent chain), and different sessions pin different focuses.
    import json as _json

    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=8),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=10),
        fanout_probability=0.0,
    )

    def paths_in(idx: Any) -> Any:
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


def test_code_change_focus_and_payload_deterministic() -> None:
    t = load_theme("code_change_task")
    cfg = _code_change_cfg(
        turns_per_session=Distribution(type="fixed", mean=1),
        tool_loop_depth=Distribution(type="fixed", mean=10),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=10),
        fanout_probability=0.0,
    )
    g1 = build_graph_for_session(cfg, t, _WordTok(), session_index=2)
    g2 = build_graph_for_session(cfg, t, _WordTok(), session_index=2)
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages


# --- Per-tool payload sizing (x-payload-tokens) + domain payload pools -------


def test_payload_arg_size_from_schema_hint() -> None:
    # A payload arg's word count = its `x-payload-tokens` schema hint; a payload arg
    # without the hint falls back to _DEFAULT_PAYLOAD_WORDS. (_WordTok: 1 word == 1 token.)
    from inference_perf.datagen.synthetic_agentic import (
        _render_tool_arguments,
        theme_payload_words,
        _DEFAULT_PAYLOAD_WORDS,
    )

    t = load_theme("code_change_task")
    pool = theme_payload_words(t, 7, (68,))
    sized = {"type": "object", "properties": {"content": {"type": "string", "x-payload-tokens": 300}}, "required": ["content"]}
    a = _render_tool_arguments(sized, t, 7, (0, 0, 31, 0), pinned={}, word_pool=None, payload_pool=pool)
    assert len(a["content"].split()) == 300, "payload arg must honor x-payload-tokens"

    default = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    b = _render_tool_arguments(default, t, 7, (0, 0, 31, 0), pinned={}, word_pool=None, payload_pool=pool)
    assert len(b["code"].split()) == _DEFAULT_PAYLOAD_WORDS, "payload arg without hint falls back to the default size"


def test_payload_pool_falls_back_to_filler_when_no_payload_templates() -> None:
    # theme_payload_words returns the payload_templates pool when present, else the
    # filler_templates pool (so themes without payload_templates behave as before).
    from inference_perf.datagen.synthetic_agentic import theme_payload_words, theme_filler_words

    # a theme WITH payload_templates -> its payload pool differs from its filler pool
    coding = load_theme("code_change_task")
    assert coding.payload_templates, "code_change_task should declare payload_templates"
    assert theme_payload_words(coding, 7, (68,)) != theme_filler_words(coding, 7, (68,))

    # a theme WITHOUT payload_templates -> payload pool == filler pool (fallback)
    bare = Theme(
        name="bare",
        system_prompt="sys",
        verbs=["Do"],
        entities={"thing": ["x", "y"]},
        tool_names=["t1"],
        result_templates={"default": "r {thing}"},
        objective_template="{verb} {thing}",
        filler_templates=["log line {thing} n={n0}"],
    )
    assert theme_payload_words(bare, 7, (5,)) == theme_filler_words(bare, 7, (5,))


def test_all_themes_payloads_render_domain_shaped_no_leak() -> None:
    # Every theme with a payload tool renders that payload from its payload pool with
    # NO unresolved {placeholder} leak, and long enough to be a real payload.
    import re
    from inference_perf.datagen.synthetic_agentic import _render_tool_arguments, theme_payload_words

    cases = [
        (load_theme("code_change_task"), "write_file", "content"),
        (load_theme("db2_latency_incident"), "explain_sql", "body"),
        (load_theme("research_rag"), "write_answer", "body"),
        (GENERIC_THEME, "apply_remediation", "body"),
    ]
    for theme, tool, arg in cases:
        assert tool in theme.tool_parameters, f"{theme.name}: missing {tool} schema"
        pool = theme_payload_words(theme, 3, (68,))
        args = _render_tool_arguments(
            theme.tool_parameters[tool], theme, 3, (0, 0, 31, 0), pinned={}, word_pool=None, payload_pool=pool
        )
        body = args[arg]
        leaks = re.findall(r"\{[a-z_]+[0-9]*\}", body)  # unresolved theme placeholders
        assert not leaks, f"{theme.name}.{tool}: placeholder leak {leaks} in payload"
        assert len(body.split()) >= 40, f"{theme.name}.{tool}: payload too short ({len(body.split())} words)"


def test_payload_render_deterministic() -> None:
    from inference_perf.datagen.synthetic_agentic import _render_tool_arguments, theme_payload_words

    t = load_theme("db2_latency_incident")
    pool = theme_payload_words(t, 9, (68,))
    schema = t.tool_parameters["explain_sql"]
    a = _render_tool_arguments(schema, t, 9, (0, 0, 31, 0), pinned={}, word_pool=None, payload_pool=pool)
    b = _render_tool_arguments(schema, t, 9, (0, 0, 31, 0), pinned={}, word_pool=None, payload_pool=pool)
    assert a == b, "payload rendering must be deterministic for a given (seed, path)"


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


def _compaction_cfg(**kw: Any) -> SyntheticAgenticConfig:
    """A multi-round single-agent config (no tools in the loop, so rounds are the
    only growth) tuned for compaction tests in _WordTok units.

    shared_system_prompt_len is pinned to 0 so the compacted fresh principal is
    [user] (summary at messages[0]) and the tuned triggers stay in head-less
    token units; the head is a fixed one-time cost orthogonal to the round-chain
    growth these tests exercise (the default is now 1000)."""
    base = dict(
        num_sessions=1,
        seed=7,
        turns_per_session=Distribution(type="fixed", mean=6),
        fanout_probability=0.0,
        theme_mix={"generic": 1.0},
        tool_loop_depth=Distribution(type="fixed", mean=0),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=8),
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        shared_system_prompt_len=0,
    )
    base.update(kw)
    return SyntheticAgenticConfig(**base)


def _cc(trigger: Any, target: Any) -> Any:
    """Shorthand for a ContextCompactionConfig with fixed trigger/target token counts."""
    return ContextCompactionConfig(
        trigger_tokens=Distribution(type="fixed", mean=trigger),
        target_tokens=Distribution(type="fixed", mean=target),
    )


def _principal_segments(g: ReplayGraph) -> Any:
    """Ordered list of (event_id, [segment types]) for every :principal event.
    A fresh/compacted principal has NO input_segments (None or [])."""
    out = []
    for eid, ev in g.events.items():
        if eid.endswith(":principal"):
            segs = ev.call.input_segments or []
            out.append((eid, [s.type for s in segs]))
    return out


def test_compaction_off_by_default_is_byte_identical() -> None:
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


def test_compaction_trigger_high_never_compacts() -> None:
    # A trigger far above any achievable accumulation must behave exactly like
    # compaction-off: every r>=1 round still grows.
    cfg = _compaction_cfg(context_compaction=_cc(10_000_000, 12))
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    for eid, types in _principal_segments(g):
        if eid != "synthN0:r0:principal":
            assert types == ["shared", "output", "unique"], f"{eid} should GROW under a huge trigger"


def test_compaction_fires_mid_session() -> None:
    # A trigger inside the accumulation band compacts at least one mid-session
    # round: that round's principal is FRESH (all-unique, no shared/output),
    # i.e. it does NOT slice into the prior principal -> the transcript is dropped.
    cfg = _compaction_cfg(
        turns_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    seg_map = dict(_principal_segments(g))
    # some mid-session round (r>=1) reset to fresh
    compacted = [eid for eid, types in seg_map.items() if eid != "synthN0:r0:principal" and types == []]
    assert compacted, f"expected at least one mid-session compaction, got {seg_map}"
    # A compacted round is FRESH: the summary+objective user turn (plus, when the
    # round is also terminal, a trailing ROOT_ANSWER_DIRECTIVE user message). The
    # defining property is that it drops the transcript -- all messages are user-role
    # and there are NO shared/output segments slicing into the prior principal.
    for eid in compacted:
        msgs = g.events[eid].call.messages
        assert 1 <= len(msgs) <= 2, f"{eid} compacted principal should be the fresh turn (+opt nudge), got {len(msgs)}"
        assert all(m["role"] == "user" for m in msgs), f"{eid} compacted principal must be user-role, got {msgs}"
        assert (g.events[eid].call.input_segments or []) == [], f"{eid} must have NO shared/output segments"
        # ordering edge to the prior answer is preserved (session stays one chain)
        assert g.events[eid].predecessor_event_ids, f"{eid} should keep an ordering edge to the prior round"


def test_compaction_summary_block_present_and_sized() -> None:
    # The compacted round's user turn carries a seeded summary block (plus the
    # objective). With a small target the turn is much smaller than a grown round
    # would be -> the prefill drop.
    cfg = _compaction_cfg(
        turns_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    seg_map = dict(_principal_segments(g))
    compacted = [eid for eid, t in seg_map.items() if eid != "synthN0:r0:principal" and t == []]
    assert compacted
    content = g.events[compacted[0]].call.messages[0]["content"]
    assert "Summary of prior context:" in content, "compacted turn must carry the summary fixed-content"


def test_compaction_recap_names_real_subject_and_tools() -> None:
    # When the theme defines compaction_summary_template, the recap is a real
    # semantic handoff: it names the session's pinned subject and REAL tool names
    # from the catalog (not generic filler), so it reads like a genuine recap.
    cfg = _compaction_cfg(
        turns_per_session=Distribution(type="fixed", mean=8),
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


def test_compaction_recap_falls_back_to_bare_marker_without_template() -> None:
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
        turns_per_session=Distribution(type="fixed", mean=8),
        tool_catalog_size_per_agent=Distribution(type="fixed", mean=2),
        context_compaction=_cc(90, 12),
    )
    g = build_graph_for_session(cfg, bare, _WordTok(), 0)
    compacted = [eid for eid, t in _principal_segments(g) if eid != "synthN0:r0:principal" and t == []]
    assert compacted, "compaction fires regardless of whether the theme has a recap template"
    content = g.events[compacted[0]].call.messages[0]["content"]
    assert "Summary of prior context:" in content
    assert "So far: ran" not in content, "no recap sentence when the theme defines no template"


def test_accumulated_wire_tokens_includes_catalog() -> None:
    tok = _WordTok()
    defs = _tool_definitions(GENERIC_THEME, 8)
    msgs = [{"role": "user", "content": "one two three four five"}]
    with_cat = _accumulated_wire_tokens(tok, msgs, defs)
    without_cat = _accumulated_wire_tokens(tok, msgs, [])
    import json as _json

    assert with_cat - without_cat == tok.count_tokens(_json.dumps(defs)), "catalog tokens must be added"
    assert without_cat == tok.count_tokens("one two three four five")


def test_compaction_config_requires_both_fields() -> None:
    from pydantic import ValidationError

    # The nested block requires BOTH trigger_tokens and target_tokens.
    with pytest.raises(ValidationError):
        ContextCompactionConfig(trigger_tokens=Distribution(type="fixed", mean=655))  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ContextCompactionConfig(target_tokens=Distribution(type="fixed", mean=12))  # type: ignore[call-arg]
    # both set -> accepted, and attaches cleanly to the parent config
    _compaction_cfg(context_compaction=_cc(655, 12))
    # block omitted -> accepted (compaction off)
    _compaction_cfg()


def test_compaction_deterministic() -> None:
    cfg = _compaction_cfg(
        turns_per_session=Distribution(type="fixed", mean=8),
        context_compaction=_cc(655, 12),
    )
    g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 2)
    g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 2)
    assert list(g1.events.keys()) == list(g2.events.keys())
    for eid in g1.events:
        assert g1.events[eid].call.messages == g2.events[eid].call.messages
        assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments


# --- Async sub-agent notifications: the K-notification fan-out tail ---------
#
# A spawn ends in K SEQUENTIAL notification events: each dispatch's tool result
# is a static content-free launch ack, and each child's report arrives later as its
# own user-role notification message (`async_report`). Only the LAST notification is
# the agent terminal.


def _notify_chains(g: ReplayGraph) -> Any:
    """Group notification event ids per spawn: {agent_prefix: [ids in chain order]}."""
    by_prefix: Dict[str, Any] = {}
    for eid in g.events:
        if ":notify" not in eid:
            continue
        prefix, idx = eid.rsplit(":notify", 1)
        by_prefix.setdefault(prefix, []).append((int(idx), eid))
    return {p: [eid for _, eid in sorted(pairs)] for p, pairs in by_prefix.items()}


def _async_fanout_cfg(K: int = 2, **over: Any) -> SyntheticAgenticConfig:
    base: Dict[str, Any] = dict(
        theme_mix={"generic": 1.0},
        turns_per_session=Distribution(type="fixed", mean=1),
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="fixed", mean=K),
        max_depth=1,
        tool_loop_depth=Distribution(type="fixed", mean=0),
        max_events_per_session=512,
    )
    base.update(over)
    return _cfg(**base)


def test_spawn_emits_k_sequential_notifications_each_gated_on_its_own_child() -> None:
    # (a) K notification events per spawn, chained sequentially, each depending on
    # exactly two things: the prior link in the chain (the spawn, for the first) and
    # its OWN child's terminal. The per-child dependency is what makes the chain a
    # real async wait: a notification cannot fire before that child has finished.
    K = 3
    g = build_graph_for_session(_async_fanout_cfg(K), GENERIC_THEME, _WordTok(), session_index=0)

    chains = _notify_chains(g)
    assert chains, "notification chains materialized"
    child_terminals = {eid for eid in g.events if ":sub" in eid}

    for prefix, chain in chains.items():
        assert len(chain) == K, f"{prefix}: expected K={K} notifications, got {len(chain)}"
        spawn_id = f"{prefix}:spawn"
        ack_id = f"{prefix}:dispatch_ack"
        assert spawn_id in g.events, f"{prefix}: spawn event exists"

        # The immediate post-dispatch turn: gated on the SPAWN ALONE (no child), so the
        # launch acknowledgment is observed before any report arrives.
        assert ack_id in g.events, f"{prefix}: dispatch_ack event exists"
        assert g.events[ack_id].predecessor_event_ids == [spawn_id], (
            f"{ack_id}: must depend on the spawn ONLY, got {g.events[ack_id].predecessor_event_ids}"
        )
        assert not any(s.type == "async_report" for s in g.events[ack_id].call.input_segments), (
            f"{ack_id}: the post-dispatch turn must carry NO child report"
        )

        seen_children = []
        for i, eid in enumerate(chain):
            preds = g.events[eid].predecessor_event_ids
            assert len(preds) == 2, f"{eid}: expected exactly 2 predecessors, got {preds}"
            # the chain starts at the ack turn, then threads notification to notification
            expected_prev = ack_id if i == 0 else chain[i - 1]
            assert preds[0] == expected_prev, f"{eid}: chain predecessor must be {expected_prev}, got {preds[0]}"
            child = preds[1]
            assert child in child_terminals, f"{eid}: second predecessor {child} is not a child terminal"
            seen_children.append(child)
            # every predecessor is declared a dependency
            assert set(g.events[eid].predecessor_dependency_types) == set(preds)

        # each notification is gated on a DISTINCT child -> all K children covered once
        assert len(set(seen_children)) == K, f"{prefix}: notifications must cover K distinct children"


def test_only_last_notification_is_the_terminal() -> None:
    # (b) Only the LAST notification produces the answer/report; the earlier ones are
    # short ack turns. The last link is also the event the agent chain hands upward,
    # so for a root spawn it must be a graph terminal (nothing depends on it).
    from inference_perf.datagen.synthetic_agentic import ROOT_ANSWER_DIRECTIVE

    K = 3
    # Pin output_tokens_per_turn well above _FB_ACK_TOKENS so "the terminal answer is
    # sized larger than an ack" is a meaningful comparison rather than an artifact of
    # whatever the default happens to be.
    g = build_graph_for_session(
        _async_fanout_cfg(K, output_tokens_per_turn=Distribution(type="fixed", mean=256)),
        GENERIC_THEME,
        _WordTok(),
        session_index=0,
    )
    chains = _notify_chains(g)

    for prefix, chain in chains.items():
        *acks, last = chain
        assert len(acks) == K - 1

        # the ack turns carry NO terminal directive and are strictly shorter outputs
        for eid in acks:
            ev = g.events[eid]
            msgs = ev.call.messages
            assert msgs[-1]["role"] == "user", f"{eid}: ack input ends in the notification (user) message"
            assert ROOT_ANSWER_DIRECTIVE not in str(msgs[-1].get("content", "")), f"{eid}: ack must carry no directive"
            assert ev.call.expected_output, f"{eid}: ack still produces text"

        # the last link is the terminal: it appends the answer directive
        last_msgs = g.events[last].call.messages
        assert ROOT_ANSWER_DIRECTIVE in str(last_msgs[-1].get("content", "")), (
            f"{last}: the LAST notification must carry the terminal answer directive"
        )
        # and its expected output is a full answer, not an ack
        assert g.events[last].call.expected_output_tokens > g.events[acks[0]].call.expected_output_tokens, (
            "terminal answer must be sized larger than a non-terminal ack"
        )

        # nothing in the graph depends on the root chain's last link (it is terminal)
        if ":sub" not in prefix.rsplit(":d", 1)[0]:
            dependents = [e for e in g.events.values() if last in e.predecessor_event_ids]
            assert not dependents, f"{last}: root terminal must have no dependents, got {[d.event_id for d in dependents]}"


def test_every_notification_has_zero_wait_ms() -> None:
    # (c) Timing comes SOLELY from the child terminal's own live-measured LLM call
    # (real TTFT + decode), captured via the DAG dependency -- never from a fabricated
    # extra sleep. A non-zero wait here would double-count that latency.
    g = build_graph_for_session(_async_fanout_cfg(3, max_depth=2), GENERIC_THEME, _WordTok(), session_index=0)
    notifies = [ev for eid, ev in g.events.items() if ":notify" in eid]
    assert notifies, "notification events materialized"
    for ev in notifies:
        assert ev.wait_ms == 0, f"{ev.event_id}: notification wait_ms must be 0, got {ev.wait_ms}"


def test_notification_chain_rolls_back_atomically_when_over_budget() -> None:
    # (d) The whole spawn is atomic: if the K children + K notifications do not fit
    # the event budget, the spawn event AND any children already built are dropped, so
    # NO partial chain (and no orphaned spawn advertising unconsumed dispatch calls)
    # survives. Sweep budgets across the threshold and assert the graph is always
    # self-consistent.
    K = 3
    for budget in range(1, 3 + K * 2 + 3):
        cfg = _async_fanout_cfg(K, max_events_per_session=budget)
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
        assert len(g.events) <= budget, f"budget {budget}: emitted {len(g.events)} events"

        chains = _notify_chains(g)
        spawn_ids = [eid for eid in g.events if eid.endswith(":spawn")]

        # a spawn survives IFF its complete chain of K notifications survives
        for spawn_id in spawn_ids:
            prefix = spawn_id.rsplit(":spawn", 1)[0]
            assert prefix in chains, f"budget {budget}: spawn {spawn_id} survived with no notification chain"
            assert len(chains[prefix]) == K, (
                f"budget {budget}: spawn {spawn_id} kept a PARTIAL chain of {len(chains[prefix])} (expected {K})"
            )
        # and no chain survives without its spawn
        for prefix in chains:
            assert f"{prefix}:spawn" in g.events, f"budget {budget}: chain under {prefix} survived without its spawn"

        # every predecessor reference resolves (no dangling ids after a rollback)
        for ev in g.events.values():
            for p in ev.predecessor_event_ids:
                assert p in g.events, f"budget {budget}: {ev.event_id} references missing predecessor {p}"


def test_shared_segment_never_overclaims_its_source_message_count() -> None:
    # A `shared` segment is replayed as get_messages_by_event_id(src)[:message_count].
    # Claiming MORE messages than the source event actually has silently trips the
    # runtime's length-mismatch guard and falls back to the recorded prefix, losing the
    # live transcript. Cursor math (sum(message_count) == len(messages)) does NOT catch
    # this, so assert it directly across a deep fan-out graph.
    for idx in range(3):
        g = build_graph_for_session(
            _async_fanout_cfg(2, max_depth=2, tool_loop_depth=Distribution(type="fixed", mean=1)),
            GENERIC_THEME,
            _WordTok(),
            session_index=idx,
        )
        for eid, ev in g.events.items():
            for seg in ev.call.input_segments:
                if seg.type != "shared":
                    continue
                src = seg.source_event_id
                assert src in g.events, f"{eid}: shared segment sources missing event {src}"
                have = len(g.events[src].call.messages)
                assert seg.message_count <= have, (
                    f"{eid}: shared segment claims {seg.message_count} messages from {src}, which has only {have}"
                )


def test_async_report_segments_reference_real_child_terminals() -> None:
    # Every async_report segment must source an event that EXISTS and is also a
    # declared predecessor (the runtime awaits predecessors before substituting, so a
    # non-predecessor source would be read before it is recorded).
    g = build_graph_for_session(
        _async_fanout_cfg(3, max_depth=2, tool_loop_depth=Distribution(type="fixed", mean=1)),
        GENERIC_THEME,
        _WordTok(),
        session_index=0,
    )
    seen = 0
    for eid, ev in g.events.items():
        preds = set(ev.predecessor_event_ids)
        for seg in ev.call.input_segments:
            if seg.type != "async_report":
                continue
            seen += 1
            assert seg.message_count == 1, f"{eid}: async_report must cover exactly 1 message"
            assert seg.source_event_id in g.events, f"{eid}: async_report sources missing event {seg.source_event_id}"
            assert seg.source_event_id in preds, f"{eid}: async_report source {seg.source_event_id} is not a predecessor"
    assert seen, "async_report segments materialized"


def test_async_notification_chain_is_byte_identical_across_rebuilds() -> None:
    # Determinism across the new multi-event tail: same (config, index) -> identical
    # ids, order, messages, segments, and the independently-salted ack texts.
    cfg = _cfg(
        theme_mix={"generic": 1.0},
        fanout_probability=1.0,
        sub_agents_per_spawn=Distribution(type="uniform", min=2, max=3),
        max_depth=2,
        tool_loop_depth=Distribution(type="uniform", min=0, max=2),
        max_events_per_session=512,
    )
    for idx in range(3):
        g1 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        g2 = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=idx)
        assert list(g1.events.keys()) == list(g2.events.keys())
        for eid in g1.events:
            assert g1.events[eid].call.messages == g2.events[eid].call.messages
            assert g1.events[eid].call.input_segments == g2.events[eid].call.input_segments
            assert g1.events[eid].call.expected_output == g2.events[eid].call.expected_output
            assert g1.events[eid].predecessor_event_ids == g2.events[eid].predecessor_event_ids


def test_ack_text_is_short_form_and_distinct_from_the_terminal_answer() -> None:
    # Each ack is drawn through _ack_text with its OWN per-notification sub-seed
    # (…, c, 10/11) and sized from _FB_ACK_TOKENS, NOT output_tokens_per_turn.
    #
    # Note on what is deliberately NOT asserted: `fit_filler` pads with CYCLED pool
    # words and does not consult its `rng` for the filler body, so two acks of equal
    # target length are expected to be the same string. That is how every filler in
    # this module behaves (including _answer_text) -- so the meaningful invariant is
    # that an ack is short-form and clearly distinct from the terminal ANSWER, not
    # that the acks differ from each other.
    from inference_perf.datagen.synthetic_agentic import _FB_ACK_TOKENS

    ack_target = _FB_ACK_TOKENS.mean
    g = build_graph_for_session(
        _async_fanout_cfg(4, output_tokens_per_turn=Distribution(type="fixed", mean=256)),
        GENERIC_THEME,
        _WordTok(),
        session_index=0,
    )
    chains = _notify_chains(g)
    assert chains, "notification chains materialized"
    for prefix, chain in chains.items():
        acks = [g.events[eid].call for eid in chain[:-1]]
        assert len(acks) >= 2, "need >=2 acks to compare"
        terminal = g.events[chain[-1]].call
        for call in acks:
            # sized from the ack distribution, not from output_tokens_per_turn
            assert call.expected_output_tokens == ack_target, (
                f"{prefix}: ack sized {call.expected_output_tokens}, expected _FB_ACK_TOKENS={ack_target}"
            )
            assert call.expected_output, "ack produces text"
            assert call.expected_output != terminal.expected_output, "ack text must differ from the terminal answer"
        # the terminal really is the long-form answer
        assert terminal.expected_output_tokens == 256, "terminal sized from output_tokens_per_turn"


def test_orchestrator_flow_is_dispatch_ack_then_k_reports() -> None:
    """Pin the full orchestrator flow for one spawn, end to end:

        spawn         -> dispatch_agent x K
        dispatch_ack  -> "the agents are running"   (spawn-gated, NO report yet)
        notify0       -> + child 0's report         (ack)
        ...
        notify{K-1}   -> + the last report          (synthesis, TERMINAL)

    i.e. K+1 orchestrator turns per spawn, and the launch acknowledgment is observed
    BEFORE any child report -- the property that distinguishes a genuinely async
    dispatch from one that only appears async.
    """
    from inference_perf.datagen.synthetic_agentic import ASYNC_DISPATCH_STUB

    K = 3
    g = build_graph_for_session(_async_fanout_cfg(K), GENERIC_THEME, _WordTok(), session_index=0)

    chains = _notify_chains(g)
    assert chains, "notification chains materialized"
    for prefix, chain in chains.items():
        ack = g.events[f"{prefix}:dispatch_ack"]

        # K+1 orchestrator turns for this spawn
        assert len(chain) + 1 == K + 1, f"{prefix}: expected K+1={K + 1} orchestrator turns"

        # the ack turn's input already carries all K stub results, and no report
        stub_msgs = [m for m in ack.call.messages if m.get("role") == "tool"]
        assert len(stub_msgs) == K, f"{prefix}: ack turn sees all K launch acks"
        for m in stub_msgs:
            assert m["content"] == ASYNC_DISPATCH_STUB
        assert not any(s.type == "async_report" for s in ack.call.input_segments)
        # its input ENDS on the stubs (nothing appended after them)
        assert ack.call.messages[-1]["role"] == "tool", "post-dispatch turn ends on the stub results"

        # each notification adds exactly one report, and reports accumulate
        prior_reports = 0
        for i, eid in enumerate(chain):
            ev = g.events[eid]
            n_reports = sum(1 for s in ev.call.input_segments if s.type == "async_report")
            assert n_reports == 1, f"{eid}: exactly one NEW report per notification"
            # the growing transcript: each link is strictly longer than the previous
            prev_len = len(ack.call.messages) if i == 0 else len(g.events[chain[i - 1]].call.messages)
            assert len(ev.call.messages) > prev_len, f"{eid}: transcript must grow"
            prior_reports += 1
        assert prior_reports == K, "every child's report is delivered exactly once"


def test_dispatch_ack_turn_is_the_short_prefill_shape() -> None:
    """The post-dispatch turn is the SHORT prefix a real harness produces: strictly
    fewer input messages than any notification turn, since no report has arrived."""
    K = 3
    g = build_graph_for_session(_async_fanout_cfg(K), GENERIC_THEME, _WordTok(), session_index=0)
    for prefix, chain in _notify_chains(g).items():
        ack_len = len(g.events[f"{prefix}:dispatch_ack"].call.messages)
        for eid in chain:
            assert ack_len < len(g.events[eid].call.messages), (
                f"{prefix}: dispatch_ack ({ack_len} msgs) must be shorter than {eid}"
            )


def test_k1_spawn_degenerates_to_ack_then_single_terminal() -> None:
    """With K=1 there is no non-terminal report turn: the flow is spawn ->
    dispatch_ack -> notify0(TERMINAL). Guards the degenerate case."""
    from inference_perf.datagen.synthetic_agentic import ROOT_ANSWER_DIRECTIVE

    g = build_graph_for_session(_async_fanout_cfg(1), GENERIC_THEME, _WordTok(), session_index=0)
    chains = _notify_chains(g)
    assert chains, "a K=1 spawn still produces a notification chain"
    for prefix, chain in chains.items():
        assert len(chain) == 1, f"{prefix}: K=1 -> exactly one notification"
        ack_id = f"{prefix}:dispatch_ack"
        assert g.events[ack_id].predecessor_event_ids == [f"{prefix}:spawn"]
        # the single notification is the terminal and carries the answer directive
        term = g.events[chain[0]]
        assert term.predecessor_event_ids[0] == ack_id
        assert ROOT_ANSWER_DIRECTIVE in str(term.call.messages[-1].get("content", ""))


def _theme_tool_call_events_by_agent(g: Any) -> "collections.Counter[str]":
    """Count tool-EMITTING events per owning agent, excluding the dispatch spawn.

    An agent's tool-loop events are `:principal` and `:tN`; the `:spawn` event
    emits dispatch_agent (structural fan-out, not a tool-loop iteration), so it
    is not counted here.
    """
    per: "collections.Counter[str]" = collections.Counter()
    for eid, ev in g.events.items():
        c = ev.call
        if not c.expected_output_is_tool_call:
            continue
        names = c.expected_output_tool_names or []
        if names and all(n == DISPATCH_AGENT_NAME for n in names):
            continue
        per[re.sub(r":(principal|t\d+)$", "", eid)] += 1
    return per


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_spawning_agent_runs_exactly_tool_loop_depth_tool_calls(k: int) -> None:
    """A SPAWNING agent must run the same number of tool-emitting calls as a leaf
    with the same tool_loop_depth: exactly k.

    Regression: the tool loop's last turn used to be forced to emit a tool call for
    turn `k` whenever the agent was going to spawn (`turn_is_terminal` was False for
    a spawner, so the last iteration took the "output is the NEXT tool call" branch).
    That gave a spawner k+1 tool-emitting calls -- an orchestrator with
    tool_loop_depth=3 ran 4 tool calls while its own children ran 3 -- and the extra
    call was never answered, since the matching tool results only materialize in the
    following loop event, which does not exist past t=k-1.
    """
    for fanout in (0.0, 1.0):
        cfg = _async_fanout_cfg(
            2,
            tool_loop_depth=Distribution(type="fixed", mean=k),
            fanout_probability=fanout,
            max_events_per_session=4096,
        )
        g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)
        per = _theme_tool_call_events_by_agent(g)
        assert per, f"k={k} fanout={fanout}: expected tool-emitting events"
        for agent, n in per.items():
            assert n == k, f"k={k} fanout={fanout}: agent {agent} ran {n} tool-emitting calls, expected {k}"


def test_spawn_event_consumes_the_last_tool_result() -> None:
    """With a tool loop AND a spawn, the spawn event is the loop's last link: it
    carries the prior tool call's reply (`output` segment) plus the matching tool
    results, then asks for the delegation. This is what leaves no tool call
    unanswered while keeping the tool-call count at tool_loop_depth.
    """
    k = 3
    cfg = _async_fanout_cfg(2, tool_loop_depth=Distribution(type="fixed", mean=k), max_events_per_session=4096)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), session_index=0)

    spawn_ids = [eid for eid in g.events if eid.endswith(":spawn")]
    assert spawn_ids, "a spawn event materialized"
    for sid in spawn_ids:
        ev = g.events[sid]
        segs = ev.call.input_segments
        assert [s.type for s in segs] == ["shared", "output", "unique"], (
            f"{sid}: spawn must consume the prior tool-call reply via an output segment, got {[s.type for s in segs]}"
        )
        # the last loop turn's call is answered here, and the delegation ask is last
        assert any(m.get("role") == "tool" for m in ev.call.messages), f"{sid}: carries the final tool results"
        assert ev.call.messages[-1]["role"] == "user", f"{sid}: the delegation ask is the recency position"
        # cursor math stays exact
        assert sum(s.message_count for s in segs) == len(ev.call.messages), f"{sid}: segment sum == message count"
        # every tool_call in the transcript is matched by exactly one result
        call_ids = [tc["id"] for m in ev.call.messages for tc in (m.get("tool_calls") or [])]
        res_ids = [m["tool_call_id"] for m in ev.call.messages if m.get("role") == "tool"]
        assert sorted(call_ids) == sorted(res_ids), f"{sid}: no dangling tool_call ids"


def test_k0_spawn_keeps_the_shared_only_prefix() -> None:
    """The k=0 spawner has no outstanding tool call, so its spawn event keeps the
    shared-only prepend (no `output` segment to substitute). Guards the branch."""
    g = build_graph_for_session(_async_fanout_cfg(2), GENERIC_THEME, _WordTok(), session_index=0)
    spawn_ids = [eid for eid in g.events if eid.endswith(":spawn")]
    assert spawn_ids
    for sid in spawn_ids:
        segs = g.events[sid].call.input_segments
        assert [s.type for s in segs] == ["shared", "unique"], f"{sid}: k=0 spawn stays shared-only"
        assert sum(s.message_count for s in segs) == len(g.events[sid].call.messages)
