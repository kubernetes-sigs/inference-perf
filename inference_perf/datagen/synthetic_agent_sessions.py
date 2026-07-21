# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic multi-agent session generator.

This module builds synthetic agent-session replay graphs procedurally.
Determinism is a hard requirement: graph generation must be a pure function
of (config, session_index), reproducible byte-for-byte across processes
(e.g. a parent process and its worker processes). To achieve this we avoid
Python's salted `hash()` entirely and derive all randomness from `numpy`
`Generator` instances seeded from stable, path-derived integers.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from inference_perf.config import APIConfig, DataConfig
from inference_perf.config.datagen.replay import SyntheticAgentSessionsConfig
from inference_perf.config.common import Distribution
from inference_perf.datagen.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph
from inference_perf.datagen.synthetic_themes import GENERIC_THEME, Theme, load_theme
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.numeric.distribution.utils import sample_from_distribution

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager

logger = logging.getLogger(__name__)


def session_seed(base_seed: int, session_index: int) -> int:
    """Derive a stable per-session seed from a base seed and session index.

    Pure function of its inputs -- does NOT use Python's built-in `hash()`,
    which is salted per-process (via PYTHONHASHSEED) and would break
    reproducibility across processes.
    """
    digest = hashlib.blake2b(f"{base_seed}:{session_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def child_rng(parent_seed: int, *path: int) -> np.random.Generator:
    """Create a numpy random Generator derived from a parent seed and a graph path.

    Folding the path into the seed sequence means different positions in the
    generated graph draw from independent, reproducible random streams.
    """
    return np.random.default_rng([parent_seed, *path])


def sample_int(dist: Optional[Distribution], rng: np.random.Generator, fallback: Distribution) -> int:
    """Resolve `dist` (or `fallback` if None) and draw a single deterministic int.

    Always passes `rng` explicitly to `sample_from_distribution` -- the
    util's default (unseeded) RNG would break determinism.
    """
    d = dist if dist is not None else fallback
    val = sample_from_distribution(d, 1, rng=rng)[0]
    return int(val)


# --- Filler fitting -------------------------------------------------------
#
# Free-text turns (e.g. an agent's objective/summary line) are padded with
# filler so the turn's token count matches a sampled target, while keeping
# the "real" content the model should attend to distinguishable from the
# padding via FILLER_MARKER. TOOL_CALL_MARGIN is the token headroom reserved
# elsewhere in the generator so a tool-call turn's fixed overhead doesn't
# blow past its target; it lives here because it's part of the same
# token-budgeting vocabulary as fit_filler.

TOOL_CALL_MARGIN = 64
FILLER_MARKER = "[--- ignore the preceding filler; actual content follows ---]"

# Shakespeare corpus shipped with the repo; same file/location convention
# used by synthetic_datagen.py and weka_trace_replay_datagen.py for prompt
# corpora. Loaded lazily (not at import time) and cached in-process.
_SHAKESPEARE_PATH = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"
_corpus_words_cache: Optional[List[str]] = None


def _corpus_words() -> List[str]:
    """Return the Shakespeare corpus split into whitespace-delimited words.

    No shared corpus-word loader exists elsewhere in the codebase to reuse
    (synthetic_datagen.py / weka_trace_replay_datagen.py each inline their
    own read of assets/shakespeare.txt and feed it straight through the
    tokenizer, rather than exposing a word list); this mirrors their
    file-location convention. Falls back to a tiny built-in word list if the
    asset is missing so filler generation never hard-fails on that alone.
    """
    global _corpus_words_cache
    if _corpus_words_cache is None:
        if _SHAKESPEARE_PATH.is_file():
            _corpus_words_cache = _SHAKESPEARE_PATH.read_text(encoding="utf-8", errors="ignore").split()
        else:
            logger.debug("fit_filler: corpus file not found at %s; using fallback word list", _SHAKESPEARE_PATH)
            _corpus_words_cache = ["lorem", "ipsum", "dolor", "sit", "amet"]
    return _corpus_words_cache


def fit_filler(tokenizer, target_tokens: int, fixed_content: str, rng: Optional[np.random.Generator]) -> str:
    """Pad `fixed_content` with Shakespeare-corpus filler to approximate `target_tokens`.

    filler_budget = target_tokens - count_tokens(fixed_content + " " + FILLER_MARKER).

    Budget guard: if filler_budget <= 0 the target is too small to even fit the
    fixed content plus the marker -- flooring to `fixed_content` alone (no
    marker, no filler) is the only crash-free option, so that's what happens.
    This is logged at debug rather than raised, since a too-small target is an
    expected edge of the sampled-token-count distribution, not a bug.

    Otherwise, words are appended after the marker until the text reaches (or
    passes) target_tokens, tracking the best (closest-to-target) candidate
    seen across a bounded number of iterations, mirroring
    datagen_utils.converge_to_exact_length_text's iteration cap but wrapping
    it so imperfect convergence returns the closest text instead of raising
    -- fit_filler must never raise to its caller for length reasons.
    """
    marker_and_fixed = fixed_content + " " + FILLER_MARKER
    fixed_cost = tokenizer.count_tokens(marker_and_fixed)
    filler_budget = target_tokens - fixed_cost
    if filler_budget <= 0:
        logger.debug(
            "fit_filler: non-positive filler budget (target_tokens=%d, fixed_cost=%d); "
            "flooring to fixed_content with no marker/filler",
            target_tokens,
            fixed_cost,
        )
        return fixed_content

    words = _corpus_words()
    best_text, best_gap = marker_and_fixed, abs(fixed_cost - target_tokens)
    buf = marker_and_fixed
    idx = 0
    max_iterations = 20  # bounded, mirrors converge_to_exact_length_text's cap
    for _ in range(max_iterations):
        cur = tokenizer.count_tokens(buf)
        gap = abs(cur - target_tokens)
        if gap < best_gap:
            best_gap, best_text = gap, buf
        if cur >= target_tokens:
            break
        take = max(1, target_tokens - cur)
        if idx >= len(words):
            idx = 0  # wrap around a short/exhausted corpus rather than stalling
        chunk = words[idx : idx + take]
        if not chunk:
            chunk = words[: max(1, take)]
        buf = buf + " " + " ".join(chunk)
        idx += len(chunk)
    return best_text


# --- The seeded single-agent walk -----------------------------------------
#
# build_graph_for_session emits a valid SINGLE-AGENT replay graph for one
# session: N rounds, each of which is
#     [principal input] -> k tool-turns -> [answer]
# where a tool-turn is one assistant tool_call event immediately followed by
# a role:tool result event, wired via predecessor_event_ids. Fan-out
# (sub-agent spawning) is a separate task and is deliberately NOT done here.
#
# Determinism: every random draw comes from a child_rng derived from the
# per-session seed and a stable graph-path tuple; no wall-clock, no hash().

# Fallbacks for optional distributions (§8 documented defaults).
_FB_TOOL_TURNS = Distribution(type="fixed", mean=2)
_FB_TOOL_DEFS = Distribution(type="fixed", mean=8)
_FB_SUB_AGENTS = Distribution(type="uniform", min=2, max=4)
_FB_PARALLEL = Distribution(type="fixed", mean=1)

# Canonical structural tool used to spawn sub-agents. It is NOT a theme tool:
# it must be advertised on any event that FORCES it (dispatch events, via
# expected_output_tool_names) or EMITS it (the merge event, which stores
# dispatch_agent tool_calls in its message history). inv #2 requires every
# forced/emitted tool name to appear in that turn's tool_definitions with a
# top-level `name` key, so its shape mirrors `_tool_definitions` output exactly.
DISPATCH_AGENT_NAME = "dispatch_agent"
DISPATCH_AGENT_TOOL_DEF: Dict[str, Any] = {
    "name": DISPATCH_AGENT_NAME,
    "type": "function",
    "function": {
        "name": DISPATCH_AGENT_NAME,
        "parameters": {"type": "object", "properties": {"objective": {"type": "string"}}},
    },
}


def _tool_definitions(theme, n: int) -> List[Dict[str, Any]]:
    """Build `n` tool definitions, each with a TOP-LEVEL `name` key (inv #2).

    Cycles the theme's tool_names and suffixes duplicates so names stay
    unique when the requested catalog is larger than the theme's list.
    """
    out: List[Dict[str, Any]] = []
    names = theme.tool_names or ["noop"]
    for i in range(n):
        name = names[i % len(names)] + ("" if i < len(names) else f"_{i}")
        out.append(
            {
                "name": name,
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
            }
        )
    return out


def _render_objective(theme, rng: np.random.Generator) -> str:
    """Render a single principal objective string from the theme templates."""
    verb = theme.verbs[int(rng.integers(0, len(theme.verbs)))]
    subs: Dict[str, str] = {"verb": verb}
    for key, vals in theme.entities.items():
        if vals:
            subs[key] = vals[int(rng.integers(0, len(vals)))]
    try:
        return theme.objective_template.format(**subs)
    except (KeyError, IndexError):
        return f"{verb}: complete the task."


def _render_followup(theme, objective: str, rng: np.random.Generator) -> str:
    """Render a round-K (K>=1) follow-up principal turn from the theme.

    Uses the theme's followup_templates (optionally prefixed by a
    followup_connective) when present; otherwise falls back to the objective so
    the follow-up turn is never empty. The result is the fixed_content that
    input_tokens_per_turn sizing (fit_filler) later pads.
    """
    connective = ""
    if theme.followup_connectives:
        connective = theme.followup_connectives[int(rng.integers(0, len(theme.followup_connectives)))]
    if theme.followup_templates:
        tpl = theme.followup_templates[int(rng.integers(0, len(theme.followup_templates)))]
        subs: Dict[str, str] = {}
        for key, vals in theme.entities.items():
            if vals:
                subs[key] = vals[int(rng.integers(0, len(vals)))]
        try:
            return connective + tpl.format(**subs)
        except (KeyError, IndexError):
            return connective + objective
    return connective + objective


def build_graph_for_session(cfg, theme, tokenizer, session_index: int) -> ReplayGraph:
    """Build a single-agent replay graph for one synthetic session.

    Emits N rounds (from `rounds_per_session`); each round is a principal
    input event, `k` tool-turns (from `tool_turns_per_loop`, fallback fixed 2),
    and a final answer event. Round r+1's principal depends on round r's
    answer. Honors `max_events_per_session`: stops STARTING new rounds once the
    next round would overflow the budget (never truncates mid-round).
    """
    seed = session_seed(cfg.seed, session_index)
    sid = f"synthN{session_index}"
    events: Dict[str, GraphEvent] = {}
    root_ids: List[str] = []
    budget = cfg.max_events_per_session

    n_rounds = sample_int(cfg.rounds_per_session, child_rng(seed, 0), cfg.rounds_per_session)
    tool_defs_n = sample_int(cfg.tool_definitions_per_agent, child_rng(seed, 1), _FB_TOOL_DEFS)
    # §8: tool_definitions_per_agent=0 is the bare non-agentic / plain-chat
    # baseline — NO tools advertised at all. Floor at 0 (not 1) so that value
    # flows through to an empty catalog; `_tool_definitions(theme, 0)` returns [].
    tool_defs = _tool_definitions(theme, max(0, tool_defs_n))
    # Fan-out catalog: the theme tools PLUS the structural dispatch_agent tool.
    # Used ONLY on events that force or emit dispatch_agent (dispatch + merge),
    # so ordinary/non-fan-out events keep a clean catalog (dispatch_agent is
    # never advertised when there is no fan-out). Guard against duplication in
    # case a theme ever names a tool "dispatch_agent".
    if any(td.get("name") == DISPATCH_AGENT_NAME for td in tool_defs):
        fanout_tool_defs = tool_defs
    else:
        fanout_tool_defs = [*tool_defs, DISPATCH_AGENT_TOOL_DEF]

    system_msg: Optional[Dict[str, Any]] = None
    if cfg.shared_system_prompt_len > 0:
        content = fit_filler(tokenizer, cfg.shared_system_prompt_len, theme.system_prompt or "", rng=child_rng(seed, 2))
        system_msg = {"role": "system", "content": content}

    def _emit(event_id, messages, preds, dep_types, segs, wait_ms, is_tool_call, tool_names, defs=None):
        events[event_id] = GraphEvent(
            event_id=event_id,
            call=GraphCall(
                call_id=event_id,
                model="",
                messages=messages,
                expected_output="",
                input_segments=segs,
                total_input_tokens=0,
                expected_output_tokens=0,
                temperature=0.0,
                max_tokens_recorded=None,
                tool_definitions=tool_defs if defs is None else defs,
                expected_output_is_tool_call=is_tool_call,
                expected_output_tool_names=tool_names,
                attributes=None,
            ),
            predecessor_event_ids=preds,
            predecessor_dependency_types=dep_types,
            wait_ms=wait_ms,
            t_start_ms=0,
            t_end_ms=0,
        )

    def _system_head() -> Optional[Dict[str, Any]]:
        # Aliasing guard (§4.2/§6 option b): the invariant system head rides
        # EVERY agent's first call, but each event must own a DISTINCT dict so
        # that mutating one event's messages never corrupts another. Return a
        # fresh shallow copy each time (system_msg itself is treated read-only).
        return dict(system_msg) if system_msg is not None else None

    def _min_agent_cost() -> int:
        # Minimum events an agent occupies regardless of its tool-loop / spawn:
        # 1 principal + 1 answer. (Tool turns are >= 0; a spawn adds its own
        # cost, guarded separately at the spawn decision.)
        return 2

    # Per-round bookkeeping for §4.1 context growth: _build_agent (when is_root)
    # publishes the current round's principal event id + its input message count
    # here so the next round can build the shared/output segments that re-inject
    # the growing transcript.
    root_principal_meta: Dict[str, Any] = {}

    def _build_agent(
        depth: int,
        agent_prefix: str,
        task_msgs: List[Dict[str, Any]],
        preds: List[str],
        dep_types: Dict[str, str],
        principal_wait: int,
        is_root: bool,
        agent_seed_path: tuple,
        principal_segments: Optional[List[InputSegment]] = None,
    ) -> Optional[str]:
        """Build ONE agent's execution and return its terminal (answer) event id.

        An agent is: [principal input] -> k tool-turns -> optional fan-out
        (recurse into K sub-agents + one merge) -> [answer]. Sub-agents call
        this same builder at depth+1, so recursion is uniform.

        The FIRST call of every agent carries the byte-identical invariant
        system head (a per-event copy). Returns None if the agent's minimum
        cost (principal + answer) does not fit the remaining event budget — the
        caller must treat that as "no agent built".
        """
        if len(events) + _min_agent_cost() > budget:
            return None

        # --- principal input event (agent's FIRST call: carries system head) ---
        #
        # Size the user/text input turn to a sampled `input_tokens_per_turn`
        # target: the rendered objective/coherence text stays the fixed_content
        # (kept intact, prepended) and corpus filler pads up to the target. This
        # is what makes input_tokens_per_turn a real knob rather than a required
        # no-op. Reserved sub-index 50 (off agent_seed_path) is a FRESH path that
        # does not collide with any existing draw (100 tool-turns, 4/5 answer,
        # 7/8 spawn, 200+c children, per-t 3/9). Only the LAST message (the
        # user-role objective) is padded; the system head (if any) is untouched.
        principal_id = f"{agent_prefix}:principal"
        sized_task_msgs = list(task_msgs)
        if sized_task_msgs and sized_task_msgs[-1].get("role") == "user":
            in_tokens = sample_int(cfg.input_tokens_per_turn, child_rng(seed, *agent_seed_path, 50), cfg.input_tokens_per_turn)
            last = dict(sized_task_msgs[-1])
            last["content"] = fit_filler(
                tokenizer, in_tokens, last.get("content", ""), rng=child_rng(seed, *agent_seed_path, 51)
            )
            sized_task_msgs[-1] = last
        if principal_segments is not None:
            # §4.1 context-growth path: `task_msgs` is the FULL growing transcript
            # (already includes the system head as its first message, so the shared
            # segment — which sources the prior round's principal INPUT — covers it).
            # We must NOT prepend the head again here or it would double and break
            # the segment cursor math (sum(message_count) == len(principal_msgs)).
            principal_msgs = sized_task_msgs
            principal_segs: List[InputSegment] = principal_segments
        else:
            head = _system_head()
            principal_msgs = ([head] if head else []) + sized_task_msgs
            principal_segs = []
        _emit(principal_id, principal_msgs, preds, dep_types, principal_segs, principal_wait, False, None)
        if is_root and not root_ids:
            root_ids.append(principal_id)
        if is_root:
            # Publish this round's principal id + its INPUT length so the NEXT
            # round's `shared` segment can source it (§4.1): the shared prefix's
            # message_count MUST equal len(this principal's input) so the runtime
            # slice `get_messages_by_event_id(src)[:message_count]` matches exactly.
            root_principal_meta["id"] = principal_id
            root_principal_meta["input_len"] = len(principal_msgs)
        last_id = principal_id

        obj = task_msgs[-1].get("content", "") if task_msgs else ""

        # --- k tool-turns (ordinary single-tool turns) ---
        k = sample_int(cfg.tool_turns_per_loop, child_rng(seed, *agent_seed_path, 100), _FB_TOOL_TURNS)
        k = max(0, k)
        # §8 bare baseline: with an empty tool catalog (tool_definitions_per_agent=0)
        # a tool-loop turn cannot emit a valid forced call — the `name` lookup
        # `tool_defs[j % len(tool_defs)]` would divide by / index an empty list,
        # and inv #2 (call name must appear in tool_definitions) is unsatisfiable.
        # A catalog-less agent therefore emits ZERO tool turns and just answers,
        # which keeps invariants #2 (name-in-defs) and #3 (call/result pairing)
        # trivially valid.
        if not tool_defs:
            k = 0
        for t in range(k):
            if len(events) + 1 + 1 > budget:  # keep room for the answer
                break
            # K = parallel tool calls THIS ordinary tool turn emits. Applies ONLY
            # here (not to dispatch/merge/answer). Fresh seed sub-index (t, 30)
            # under the per-turn path: existing per-turn draws are (t, 3) [wait]
            # and (t, 9) [result n0], so 30 cannot collide. Clamp K >= 1 so a
            # turn always emits at least one call (keeps inv #3 well-defined).
            n_calls = sample_int(cfg.parallel_tool_calls_per_turn, child_rng(seed, *agent_seed_path, t, 30), _FB_PARALLEL)
            n_calls = max(1, n_calls)
            result_tpl = theme.result_templates.get("default", "result: {entity} {n0} {t0}")
            parallel_calls: List[Dict[str, Any]] = []
            result_msgs: List[Dict[str, Any]] = []
            call_names: List[str] = []
            for j in range(n_calls):
                # Cycle the theme tool catalog so each call name is a top-level
                # tool_definitions name (inv #2), as the single-call path did.
                call_name = tool_defs[j % len(tool_defs)]["name"]
                call_names.append(call_name)
                # Distinct ids across the K calls: call_{prefix}_{t}_{j}.
                tc_id = f"call_{agent_prefix}_{t}_{j}"
                parallel_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        # inv #1: arguments are json.dumps-ed strings.
                        "function": {"name": call_name, "arguments": json.dumps({"q": obj[:20], "i": j})},
                    }
                )
                try:
                    result = result_tpl.format(
                        entity="x",
                        n0=int(child_rng(seed, *agent_seed_path, t, 9, j).integers(0, 999)),
                        t0="t0",
                    )
                except (KeyError, IndexError):
                    result = "result"
                # EXACTLY one role:tool result per call, in matching positional
                # order, carrying that call's exact id (inv #3 positional).
                result_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            tool_call_msg = {"role": "assistant", "tool_calls": parallel_calls}
            turn_id = f"{agent_prefix}:t{t}"
            turn_wait = int(
                sample_from_distribution(cfg.tool_call_latency_sec, 1, rng=child_rng(seed, *agent_seed_path, t, 3))[0] * 1000
            )
            _emit(
                turn_id,
                [tool_call_msg, *result_msgs],
                [last_id],
                {last_id: "full_match"},
                [],
                turn_wait,
                True,
                call_names,
            )
            last_id = turn_id

        # --- optional fan-out: spawn K sub-agents + one merge ---
        spawn_roll = float(child_rng(seed, *agent_seed_path, 7).random())
        if spawn_roll < cfg.fanout_probability and depth < cfg.max_depth:
            K = sample_int(cfg.sub_agents_per_spawn, child_rng(seed, *agent_seed_path, 8), _FB_SUB_AGENTS)
            K = max(0, K)
            # Whole-spawn minimum cost: per child a dispatch event (1) + minimal
            # child (principal + answer = 2), plus one shared merge event, plus
            # this agent's own answer that still follows. Only spawn if it all
            # fits; otherwise this agent stays a plain leaf.
            min_spawn_cost = K * (1 + _min_agent_cost()) + 1
            if K > 0 and len(events) + min_spawn_cost + 1 <= budget:
                dispatch_pairs: List[tuple] = []  # (dispatch_id, tc_id)
                child_terminals: List[str] = []
                spawn_ok = True
                for c in range(K):
                    # --- single-call dispatch event (never dangles: one call) ---
                    disp_id = f"{agent_prefix}:d{depth}:disp{c}"
                    tc_id = f"dispatch_{agent_prefix}_{c}"
                    child_obj = _render_objective(theme, child_rng(seed, *agent_seed_path, c, 1))
                    # The dispatch event's MESSAGES are the input context; its
                    # OUTPUT is the single dispatch_agent tool_call the model
                    # generates (expected_output_is_tool_call=True, single-call
                    # so tool_choice forces the one name and it can never
                    # dangle). The generated call is later reconstructed in the
                    # merge via the "output" segment sourced from this event, so
                    # the assistant tool_call is NOT stored in these messages
                    # (which keeps inv #3 balanced: 0 calls, 0 results here).
                    dispatch_ctx = {"role": "user", "content": f"Dispatch a sub-agent to: {child_obj}"}
                    _emit(
                        disp_id,
                        [dispatch_ctx],
                        [last_id],
                        {last_id: "full_match"},
                        [],
                        0,
                        True,
                        [DISPATCH_AGENT_NAME],
                        defs=fanout_tool_defs,
                    )
                    # --- recurse into the child agent (depth+1) ---
                    child_prefix = f"{agent_prefix}:d{depth + 1}:sub{c}"
                    child_task = [{"role": "user", "content": child_obj}]
                    child_terminal = _build_agent(
                        depth + 1,
                        child_prefix,
                        child_task,
                        [disp_id],
                        {disp_id: "full_match"},
                        0,
                        False,
                        (*agent_seed_path, 200 + c),
                    )
                    if child_terminal is None:
                        # Budget ran out mid-spawn — abandon fan-out entirely so
                        # we never emit a merge referencing a missing child.
                        spawn_ok = False
                        break
                    dispatch_pairs.append((disp_id, tc_id))
                    child_terminals.append(child_terminal)

                if spawn_ok and dispatch_pairs:
                    # --- ONE merge event consuming output + tool_output per child ---
                    merge_msgs: List[Dict[str, Any]] = []
                    merge_segs: List[InputSegment] = []
                    merge_preds: List[str] = []
                    merge_deps: Dict[str, str] = {}
                    for (disp_id, tc_id), child_term in zip(dispatch_pairs, child_terminals, strict=True):
                        # Reconstruct the [assistant dispatch call, tool result]
                        # pair per child so inv #3 holds (one call, one result).
                        merge_msgs.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {"name": DISPATCH_AGENT_NAME, "arguments": json.dumps({})},
                                    }
                                ],
                            }
                        )
                        merge_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": "PLACEHOLDER"})
                        # Segments align 1:1 with messages by cursor: the output
                        # segment covers the assistant dispatch msg (replaced by
                        # the dispatch event's live tool call), tool_output covers
                        # the role:tool msg (content replaced by the child's live
                        # answer text, role + tool_call_id preserved).
                        merge_segs.append(InputSegment(type="output", message_count=1, token_count=0, source_event_id=disp_id))
                        merge_segs.append(
                            InputSegment(type="tool_output", message_count=1, token_count=0, source_event_id=child_term)
                        )
                        merge_preds += [disp_id, child_term]
                        merge_deps[disp_id] = "full_match"
                        merge_deps[child_term] = "full_match"
                    merge_id = f"{agent_prefix}:d{depth}:merge"
                    _emit(merge_id, merge_msgs, merge_preds, merge_deps, merge_segs, 0, False, None, defs=fanout_tool_defs)
                    last_id = merge_id

        # --- answer event (agent terminal) ---
        answer_id = f"{agent_prefix}:answer"
        out_tokens = sample_int(cfg.output_tokens_per_turn, child_rng(seed, *agent_seed_path, 4), cfg.output_tokens_per_turn)
        ans = fit_filler(tokenizer, out_tokens, "Summary:", rng=child_rng(seed, *agent_seed_path, 5))
        _emit(answer_id, [{"role": "assistant", "content": ans}], [last_id], {last_id: "full_match"}, [], 0, False, None)
        return answer_id

    prev_answer_id: Optional[str] = None
    # The running conversation transcript used to build round K>=1's growing
    # context (§4.1). After each round it becomes the placeholder prefix the
    # next round's `shared` segment covers; the shared segment re-injects the
    # LIVE version at replay, so the exact placeholder content only needs to be
    # coherent + deterministic and of the RIGHT length.
    transcript: List[Dict[str, Any]] = []

    for r in range(n_rounds):
        # Stop STARTING new rounds when even the minimum agent (principal +
        # answer) won't fit (§8) — never truncate mid-round. Deeper fan-out is
        # budget-guarded inside _build_agent.
        if len(events) + _min_agent_cost() > budget:
            break

        obj = _render_objective(theme, child_rng(seed, r, 1))
        # wait_ms: round 1 uses tool_call_latency; rounds 2..N use user_think_time if set.
        if r == 0:
            principal_wait = 0
        else:
            think_dist = cfg.user_think_time_sec if cfg.user_think_time_sec is not None else cfg.tool_call_latency_sec
            # Sample as a float and scale to ms BEFORE truncating to int, so a
            # fractional-second mean (e.g. 0.5s) doesn't collapse to 0/1s.
            principal_wait = int(sample_from_distribution(think_dist, 1, rng=child_rng(seed, r, 2))[0] * 1000)

        if r == 0 or prev_answer_id is None:
            # Round 0 (or defensive fallback): a fresh single-turn prompt. The
            # system head is prepended inside _build_agent; no input_segments.
            task_msgs: List[Dict[str, Any]] = [{"role": "user", "content": obj}]
            principal_segments: Optional[List[InputSegment]] = None
            preds = [prev_answer_id] if prev_answer_id else []
            dep_types = {prev_answer_id: "full_match"} if prev_answer_id else {}
        else:
            # Round K>=1 (§4.1 growing context). Layout of the principal's
            # original_messages and matching segments (cursor-aligned 1:1):
            #   [ transcript... , answer_placeholder , followup ]
            #   [ shared(count=len(transcript), src=prev principal)   ]  -> prior turns
            #   [ output(1, src=prev answer)                          ]  -> prior answer
            #   [ unique(1)                                           ]  -> new follow-up
            # sum(message_count) == len(original_messages), so the runtime cursor
            # math in _build_messages_with_substitution is exact (no IndexError).
            prev_principal_id = root_principal_meta["id"]
            prev_principal_len = root_principal_meta["input_len"]
            followup = _render_followup(theme, obj, child_rng(seed, r, 3))
            # The `shared` prefix must be exactly prev_principal_len messages; the
            # accumulated `transcript` is kept at that length as the placeholder.
            prefix_msgs = list(transcript)
            answer_placeholder = {"role": "assistant", "content": "PLACEHOLDER_PRIOR_ANSWER"}
            followup_msg = {"role": "user", "content": followup}
            task_msgs = [*prefix_msgs, answer_placeholder, followup_msg]
            principal_segments = [
                InputSegment(
                    type="shared", message_count=prev_principal_len, token_count=0, source_event_id=prev_principal_id
                ),
                InputSegment(type="output", message_count=1, token_count=0, source_event_id=prev_answer_id),
                InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
            ]
            # BOTH sources must also be predecessors so substitution runs after
            # require_async has awaited them (full_match is DOT-only).
            preds = [prev_principal_id, prev_answer_id]
            dep_types = {prev_principal_id: "full_match", prev_answer_id: "full_match"}

        terminal = _build_agent(
            0,
            f"{sid}:r{r}",
            task_msgs,
            preds,
            dep_types,
            principal_wait,
            True,
            (r,),
            principal_segments,
        )
        if terminal is None:
            break
        prev_answer_id = terminal
        # The next round's `shared` prefix must equal THIS round's principal
        # INPUT (its message_count is root_principal_meta["input_len"]). Take the
        # principal event's own messages verbatim — they ARE that input — as the
        # placeholder transcript, so len(prefix) == published input_len exactly.
        transcript = list(events[root_principal_meta["id"]].call.messages)

    return ReplayGraph(events=events, root_event_ids=root_ids, source_file="synthetic")


# --- The generator class (lazy build + theme weighting) -------------------
#
# Ties the pure graph builder above to the shared graph-backed session
# runtime. Mirrors OTelTraceReplayDataGenerator: require the replay config,
# pass it to the base as `replay_config=`, and register lazy session slots so
# get_session_count() works immediately while each graph is built on demand.


class SyntheticAgentSessionsDataGenerator(ReplayGraphSessionGeneratorBase):
    """Lazy, deterministic generator of synthetic multi-agent replay sessions.

    Each session's graph is a pure function of (config, session_index): the
    theme is chosen by a deterministic weighted draw over `theme_mix`, so two
    generator instances built from the same config emit byte-identical graphs.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional["SyncManager"] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        synthetic_config = getattr(config, "synthetic_agent_sessions", None)
        if synthetic_config is None:
            raise ValueError("synthetic_agent_sessions configuration is required for SyntheticAgentSessionsDataGenerator")

        self.synthetic_config: SyntheticAgentSessionsConfig = synthetic_config

        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            replay_config=self.synthetic_config,
        )

        # Map name -> Theme; "generic" resolves to the built-in without file IO.
        self._themes: Dict[str, Theme] = {
            name: (GENERIC_THEME if name == "generic" else load_theme(name)) for name in self.synthetic_config.theme_mix
        }

        session_ids = [f"synthN{i}" for i in range(self.synthetic_config.num_sessions)]
        self.initialize_sessions_lazy(session_ids)

    def _pick_theme(self, session_index: int) -> Theme:
        """Deterministic weighted draw of a theme for one session.

        Uses a fixed reserved RNG path (999) off the per-session seed so the
        theme choice is stable per (config, session_index) and independent of
        the graph's own random draws.
        """
        names = list(self.synthetic_config.theme_mix.keys())
        weights = np.array([self.synthetic_config.theme_mix[n] for n in names], dtype=np.float64)
        weights = weights / weights.sum()
        rng = child_rng(session_seed(self.synthetic_config.seed, session_index), 999)
        return self._themes[names[int(rng.choice(len(names), p=weights))]]

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        theme = self._pick_theme(session_index)
        graph = build_graph_for_session(self.synthetic_config, theme, self.tokenizer, session_index)
        if not graph.events:
            return None
        sid = f"synthN{session_index}"
        return ReplaySession(session_id=sid, source_id=sid, session_index=session_index, graph=graph)
