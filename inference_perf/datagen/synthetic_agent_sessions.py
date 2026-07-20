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
from typing import Any, Dict, List, Optional

import numpy as np

from inference_perf.config.common import Distribution
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph
from inference_perf.utils.numeric.distribution.utils import sample_from_distribution

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
    tool_defs = _tool_definitions(theme, max(1, tool_defs_n))

    system_msg: Optional[Dict[str, Any]] = None
    if cfg.shared_system_prompt_len > 0:
        content = fit_filler(
            tokenizer, cfg.shared_system_prompt_len, theme.system_prompt or "", rng=child_rng(seed, 2)
        )
        system_msg = {"role": "system", "content": content}

    def _emit(event_id, messages, preds, dep_types, segs, wait_ms, is_tool_call, tool_names):
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
                tool_definitions=tool_defs,
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

    def _build_agent(
        depth: int,
        agent_prefix: str,
        task_msgs: List[Dict[str, Any]],
        preds: List[str],
        dep_types: Dict[str, str],
        principal_wait: int,
        is_root: bool,
        agent_seed_path: tuple,
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
        principal_id = f"{agent_prefix}:principal"
        head = _system_head()
        principal_msgs = ([head] if head else []) + list(task_msgs)
        _emit(principal_id, principal_msgs, preds, dep_types, [], principal_wait, False, None)
        if is_root and not root_ids:
            root_ids.append(principal_id)
        last_id = principal_id

        obj = task_msgs[-1].get("content", "") if task_msgs else ""

        # --- k tool-turns (ordinary single-tool turns) ---
        k = sample_int(cfg.tool_turns_per_loop, child_rng(seed, *agent_seed_path, 100), _FB_TOOL_TURNS)
        k = max(0, k)
        for t in range(k):
            if len(events) + 1 + 1 > budget:  # keep room for the answer
                break
            call_name = tool_defs[0]["name"]
            tc_id = f"call_{agent_prefix}_{t}"
            tool_call_msg = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        # inv #1: arguments are json.dumps-ed strings.
                        "function": {"name": call_name, "arguments": json.dumps({"q": obj[:20]})},
                    }
                ],
            }
            result_tpl = theme.result_templates.get("default", "result: {entity} {n0} {t0}")
            try:
                result = result_tpl.format(
                    entity="x", n0=int(child_rng(seed, *agent_seed_path, t, 9).integers(0, 999)), t0="t0"
                )
            except (KeyError, IndexError):
                result = "result"
            tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result}
            turn_id = f"{agent_prefix}:t{t}"
            turn_wait = int(
                sample_from_distribution(cfg.tool_call_latency_sec, 1, rng=child_rng(seed, *agent_seed_path, t, 3))[0]
                * 1000
            )
            _emit(turn_id, [tool_call_msg, tool_msg], [last_id], {last_id: "full_match"}, [], turn_wait, True, [call_name])
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
                        ["dispatch_agent"],
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
                                        "function": {"name": "dispatch_agent", "arguments": json.dumps({})},
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
                    _emit(merge_id, merge_msgs, merge_preds, merge_deps, merge_segs, 0, False, None)
                    last_id = merge_id

        # --- answer event (agent terminal) ---
        answer_id = f"{agent_prefix}:answer"
        out_tokens = sample_int(cfg.output_tokens_per_turn, child_rng(seed, *agent_seed_path, 4), cfg.output_tokens_per_turn)
        ans = fit_filler(tokenizer, out_tokens, "Summary:", rng=child_rng(seed, *agent_seed_path, 5))
        _emit(answer_id, [{"role": "assistant", "content": ans}], [last_id], {last_id: "full_match"}, [], 0, False, None)
        return answer_id

    prev_answer_id: Optional[str] = None

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

        terminal = _build_agent(
            0,
            f"{sid}:r{r}",
            [{"role": "user", "content": obj}],
            [prev_answer_id] if prev_answer_id else [],
            {prev_answer_id: "full_match"} if prev_answer_id else {},
            principal_wait,
            True,
            (r,),
        )
        if terminal is None:
            break
        prev_answer_id = terminal

    return ReplayGraph(events=events, root_event_ids=root_ids, source_file="synthetic")
