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


# Number of corpus words tokenized ONCE to estimate the corpus's average
# tokens-per-word ratio. Kept comfortably below any real tokenizer's
# model_max_length (8192) so this measurement is never truncated -- that is
# the whole point: we measure a small, un-truncated sample and extrapolate,
# instead of re-tokenizing a growing multi-thousand-token buffer (which both
# saturates at the truncation ceiling AND is O(target) slow).
_RATIO_SAMPLE_WORDS = 512


def _cycled_words(words: List[str], count: int, start: int = 0) -> List[str]:
    """Return `count` words drawn from `words`, CYCLING when it runs out.

    The corpus is large (~1M words) but a realistic 100K+ token target can
    still demand more words than it holds, so we must repeat rather than cap
    at len(words). `start` lets callers offset into the cycle.
    """
    n = len(words)
    if n == 0:
        return []
    return [words[(start + i) % n] for i in range(max(0, count))]


def _untruncated_len(tokenizer, text: str) -> int:
    """Token length of `text` WITHOUT the model_max_length truncation.

    CustomTokenizer.count_tokens truncates at model_max_length (a shared
    utility we must NOT change), so it saturates and cannot MEASURE a string
    longer than the ceiling. For fit_filler's own internal sizing we go one
    level down to the raw HF tokenizer with truncation=False. If that path is
    unavailable (e.g. a lightweight fake tokenizer in tests that raises from
    get_tokenizer), fall back to count_tokens -- fakes there don't truncate,
    so the fallback is exact for them.
    """
    try:
        hf = tokenizer.get_tokenizer()
        return len(hf(text, truncation=False, add_special_tokens=False)["input_ids"])
    except Exception:
        return tokenizer.count_tokens(text)


def fit_filler(tokenizer, target_tokens: int, fixed_content: str, rng: Optional[np.random.Generator]) -> str:
    """Pad `fixed_content` with Shakespeare-corpus filler to approximate `target_tokens`.

    filler_budget = target_tokens - count_tokens(fixed_content + " " + FILLER_MARKER).

    Budget guard: if filler_budget <= 0 the target is too small to even fit the
    fixed content plus the marker -- flooring to `fixed_content` alone (no
    marker, no filler) is the only crash-free option, so that's what happens.
    This is logged at debug rather than raised, since a too-small target is an
    expected edge of the sampled-token-count distribution, not a bug.

    Sizing is ANALYTIC, not an iterative re-tokenizing loop. The old loop
    re-tokenized a growing buffer each iteration, which (1) saturated at the
    tokenizer's model_max_length truncation ceiling (~8192) so it could never
    MEASURE -- let alone reach -- a larger target, silently capping realistic
    100K+ prompts, and (2) was O(target) slow (tens of seconds per turn).

    Instead we tokenize a small fixed-size word SAMPLE once (below the ceiling,
    so it's never truncated) to get an average tokens-per-word ratio, compute
    the number of words needed = ceil((target - fixed_cost) / ratio), and emit
    that many CYCLED corpus words in one shot. A single bounded correction pass
    (measured untruncated) refines the ratio for a slight over/undershoot. This
    reaches any target regardless of corpus size, in well under a second.
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
    if not words:
        return marker_and_fixed

    # Average tokens-per-word from a small, un-truncated sample (measured once).
    sample = _cycled_words(words, min(_RATIO_SAMPLE_WORDS, len(words)))
    sample_text = " ".join(sample)
    sample_tokens = _untruncated_len(tokenizer, sample_text)
    tokens_per_word = (sample_tokens / len(sample)) if sample and sample_tokens > 0 else 1.0

    def _emit(n_words: int) -> str:
        chunk = _cycled_words(words, max(1, n_words))
        return marker_and_fixed + " " + " ".join(chunk)

    # Analytic estimate: how many words to cover the remaining budget.
    n_words = max(1, int(np.ceil(filler_budget / tokens_per_word)))
    buf = _emit(n_words)

    # One bounded correction pass: measure the real (untruncated) length of the
    # emitted text and re-derive the word count from the OBSERVED filler ratio,
    # correcting any systematic bias between the sample and the emitted filler.
    # This runs at most once -- it never loops, so it stays fast.
    actual = _untruncated_len(tokenizer, buf)
    filler_actual = actual - fixed_cost
    if actual != target_tokens and filler_actual > 0:
        observed_ratio = filler_actual / n_words
        corrected = max(1, int(np.ceil(filler_budget / observed_ratio)))
        if corrected != n_words:
            n_words = corrected
            buf = _emit(n_words)
    return buf


# --- The seeded single-agent walk -----------------------------------------
#
# build_graph_for_session emits a valid replay graph for one session: N rounds,
# each an accumulating chain of LLM calls
#     principal -> t0 -> t1 -> ... -> t{k-1}
# where EACH event is one call whose INPUT is the cumulative transcript ending
# in a user or tool message, and whose OUTPUT (expected_output) is that call's
# assistant reply — a tool call for the intermediate turns, the plain answer for
# the terminal event. There is NO separate lone-assistant "answer" event: the
# answer is the LAST call's output. With k=0 the principal itself is terminal
# (one event). Fan-out replaces the terminal with a merge event whose output is
# the answer. All wiring is via predecessor_event_ids + input_segments.
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


def _is_time_field(field: str) -> bool:
    f = field.lower()
    return f.startswith("t") and (f in ("t", "time") or f[1:].isdigit() or "time" in f or f.endswith("_t") or "_t" in f)


def _is_numeric_field(field: str) -> bool:
    f = field.lower()
    if any(c.isdigit() for c in f):
        return True
    return any(tok in f for tok in ("n", "ms", "count", "wait"))


def _seeded_time_value(rng: np.random.Generator) -> str:
    hh = int(rng.integers(0, 24))
    mm = int(rng.integers(0, 60))
    ss = int(rng.integers(0, 60))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _seeded_entity_value(theme, rng: np.random.Generator) -> str:
    pool: List[str] = []
    for vals in theme.entities.values():
        pool.extend(vals)
    if not pool:
        return f"entity-{int(rng.integers(0, 999))}"
    return pool[int(rng.integers(0, len(pool)))]


def _render_tool_result(theme, call_name: str, seed: int, path: tuple) -> str:
    """Render a tool-result content string from the theme's PER-TOOL template
    for `call_name` (falling back to 'default' only if the tool has none),
    filling EVERY placeholder the chosen template declares with a real,
    deterministically-seeded value -- never a literal stand-in, never left
    unfilled.

    `path` is the seed sub-path prefix (e.g. (*agent_seed_path, t, 9, j)) under
    which per-field sub-seeds are derived (appending a field index), so
    different fields in the same call draw from independent streams and
    different calls/turns/agents never collide.
    """
    import string

    tpl = theme.result_templates.get(call_name, theme.result_templates.get("default", "result: {entity} {n0} {t0}"))
    fields: List[str] = []
    for _, field_name, _, _ in string.Formatter().parse(tpl):
        if field_name:
            fields.append(field_name)

    values: Dict[str, str] = {}
    for idx, field in enumerate(fields):
        field_rng = child_rng(seed, *path, idx)
        if field == "entity":
            values[field] = _seeded_entity_value(theme, field_rng)
        elif _is_time_field(field):
            values[field] = _seeded_time_value(field_rng)
        elif _is_numeric_field(field):
            values[field] = str(int(field_rng.integers(0, 999)))
        else:
            # Unknown field: seeded token, never left unfilled.
            values[field] = _seeded_entity_value(theme, field_rng)

    try:
        return tpl.format_map(values)
    except (KeyError, IndexError):
        return "result"


def build_graph_for_session(cfg, theme, tokenizer, session_index: int) -> ReplayGraph:
    """Build a replay graph for one synthetic session.

    Emits N rounds (from `rounds_per_session`); each round is an accumulating
    chain of `k+1` calls (from `tool_turns_per_loop`, fallback fixed 2):
    a principal call plus `k` tool-turn calls, where each event's INPUT is the
    growing transcript and the LAST call's OUTPUT is the answer (no separate
    answer event — k=0 collapses to a single principal call). Round r+1's
    principal depends on round r's terminal call. Honors
    `max_events_per_session`: stops STARTING new rounds once even a minimal
    (single-call) agent would overflow the budget.
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

    def _emit(
        event_id,
        messages,
        preds,
        dep_types,
        segs,
        wait_ms,
        is_tool_call,
        tool_names,
        defs=None,
        expected_output="",
        expected_output_tokens=0,
    ):
        events[event_id] = GraphEvent(
            event_id=event_id,
            call=GraphCall(
                call_id=event_id,
                model="",
                messages=messages,
                expected_output=expected_output,
                input_segments=segs,
                total_input_tokens=0,
                expected_output_tokens=expected_output_tokens,
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
        # Minimum events an agent occupies regardless of its tool-loop / spawn.
        # Under the corrected event model (each call carries the cumulative
        # transcript; the terminal answer is the LAST call's OUTPUT, not a
        # separate lone-assistant event) the smallest possible agent is a single
        # principal call whose expected_output IS the answer (k=0, no spawn).
        # Tool turns are >= 0; a spawn adds its own cost, guarded separately at
        # the spawn decision.
        return 1

    # Per-round bookkeeping for §4.1 context growth: _build_agent (when is_root)
    # publishes the current round's principal event id + its input message count
    # here so the next round can build the shared/output segments that re-inject
    # the growing transcript.
    root_principal_meta: Dict[str, Any] = {}

    def _answer_text(agent_seed_path: tuple) -> tuple:
        """Render the agent's terminal answer text + its sampled token size.

        Reuses the pre-existing (…, 4) [size] and (…, 5) [filler] sub-seeds the
        old separate answer event drew from, so determinism paths are stable.
        """
        out_tokens = sample_int(cfg.output_tokens_per_turn, child_rng(seed, *agent_seed_path, 4), cfg.output_tokens_per_turn)
        ans = fit_filler(tokenizer, out_tokens, "Summary:", rng=child_rng(seed, *agent_seed_path, 5))
        return ans, out_tokens

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
        """Build ONE agent's execution and return its TERMINAL event id.

        Under the corrected event model each event is exactly one LLM call whose
        INPUT is the cumulative conversation transcript ending in a user or tool
        message; the assistant reply that call produces is the event's OUTPUT
        (expected_output), NOT a separate lone-assistant event. The agent is a
        linear accumulating chain:

            principal -> t0 -> t1 -> ... -> t{k-1}(terminal)   [k tool results]

        where the principal outputs the first tool call, each ':tN' event's
        input re-injects the prior event's tool-call reply (output segment) plus
        that call's result (unique segment), and the LAST event's OUTPUT is the
        plain answer. With k=0 the principal itself is terminal (its output IS
        the answer) — a bare single-round agent is EXACTLY one event. Fan-out
        replaces the terminal with a merge event whose output is the answer.

        The FIRST call of every agent carries the byte-identical invariant
        system head (a per-event copy). Returns None if the agent's minimum cost
        (one terminal call) does not fit the remaining event budget.
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

        obj = task_msgs[-1].get("content", "") if task_msgs else ""

        # --- decide k tool-turns up front (governs whether principal is terminal) ---
        k = sample_int(cfg.tool_turns_per_loop, child_rng(seed, *agent_seed_path, 100), _FB_TOOL_TURNS)
        k = max(0, k)
        # §8 bare baseline: with an empty tool catalog (tool_definitions_per_agent=0)
        # a tool-loop turn cannot emit a valid forced call — the `name` lookup
        # `tool_defs[j % len(tool_defs)]` would divide by / index an empty list,
        # and inv #2 (call name must appear in tool_definitions) is unsatisfiable.
        # A catalog-less agent therefore emits ZERO tool turns and just answers.
        if not tool_defs:
            k = 0

        # Will this agent spawn sub-agents? Decided up front (same reserved seed
        # path as before) so we know whether the tool loop's last event is the
        # agent terminal (plain answer output) or a hand-off into the fan-out.
        spawn_roll = float(child_rng(seed, *agent_seed_path, 7).random())
        will_spawn = spawn_roll < cfg.fanout_probability and depth < cfg.max_depth

        # Per-turn parallel-call helper: build the K calls + K results for turn t.
        def _turn_calls_and_results(t: int) -> tuple:
            n_calls = sample_int(cfg.parallel_tool_calls_per_turn, child_rng(seed, *agent_seed_path, t, 30), _FB_PARALLEL)
            n_calls = max(1, n_calls)
            calls: List[Dict[str, Any]] = []
            results: List[Dict[str, Any]] = []
            names: List[str] = []
            for j in range(n_calls):
                call_name = tool_defs[j % len(tool_defs)]["name"]
                names.append(call_name)
                tc_id = f"call_{agent_prefix}_{t}_{j}"
                calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        # inv #1: arguments are json.dumps-ed strings.
                        "function": {"name": call_name, "arguments": json.dumps({"q": obj[:20], "i": j})},
                    }
                )
                result = _render_tool_result(theme, call_name, seed, (*agent_seed_path, t, 9, j))
                results.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            return calls, results, names

        # The principal's OUTPUT: the first tool call (turn 0) if the loop runs,
        # else the plain answer. If it will spawn but has no tool turns, the
        # principal still outputs plain text and the merge follows.
        principal_is_terminal = (k == 0) and not will_spawn
        if k >= 1:
            first_calls, _, first_names = _turn_calls_and_results(0)
            _emit(
                principal_id,
                principal_msgs,
                preds,
                dep_types,
                principal_segs,
                principal_wait,
                True,
                first_names,
            )
        else:
            ans_text, ans_tokens = (_answer_text(agent_seed_path) if principal_is_terminal else ("", 0))
            _emit(
                principal_id,
                principal_msgs,
                preds,
                dep_types,
                principal_segs,
                principal_wait,
                False,
                None,
                expected_output=ans_text,
                expected_output_tokens=ans_tokens,
            )
        if is_root and not root_ids:
            root_ids.append(principal_id)
        if is_root:
            # Publish this round's principal id + its INPUT length so the NEXT
            # round's `shared` segment can source it (§4.1): the shared prefix's
            # message_count MUST equal len(this principal's input) so the runtime
            # slice `get_messages_by_event_id(src)[:message_count]` matches exactly.
            root_principal_meta["id"] = principal_id
            root_principal_meta["input_len"] = len(principal_msgs)

        # Per-agent accumulation cursor. `prev` tracks the immediately-prior
        # event of THIS agent so each successor can build its growing transcript
        # via shared(prev input) + output(prev reply) + unique(new turn):
        #   id          -- prior event id (shared + output source)
        #   input_len   -- prior event's REPLAY-recorded input length (== shared
        #                  message_count; keeps the runtime cursor exact)
        #   msgs        -- prior event's build-time input placeholder (the shared
        #                  prefix content — length must equal input_len)
        #   out_calls   -- the tool calls the prior event OUTPUTS, so this event
        #                  can build the matching result messages + an output-slot
        #                  placeholder assistant carrying those same ids (build-time
        #                  inv #3). Empty when the prior output was plain text.
        prev_id = principal_id
        prev_input_len = len(principal_msgs)
        prev_msgs: List[Dict[str, Any]] = list(principal_msgs)
        prev_out_calls: List[Dict[str, Any]] = first_calls if k >= 1 else []

        # --- k tool-turn events (accumulating chain) ---
        # Event ':tN' (N = 0..k-1) re-injects the prior event's tool-call reply
        # (output segment) plus that call's results (unique segment). Its OWN
        # output is the next tool call (N < k-1) or, if it is the agent terminal
        # (last turn AND no spawn), the plain answer.
        for t in range(k):
            # Room for this event; if it won't fit, stop the loop early. The
            # prior event remains a valid terminal (its output stays a tool call
            # if we truncate here, which is acceptable — it simply never gets a
            # follow-up; no dangling result is created because the result only
            # materializes in THIS event which we skip).
            if len(events) + 1 > budget:
                break
            _, results, _ = _turn_calls_and_results(t)
            # The output-slot placeholder assistant carries the prior event's
            # emitted calls (same ids as `results`), so build-time inv #3 holds:
            # exactly these calls are matched by exactly these results.
            output_placeholder = {"role": "assistant", "tool_calls": [dict(c) for c in prev_out_calls]}
            turn_msgs = [*prev_msgs, output_placeholder, *results]
            turn_segs = [
                InputSegment(type="shared", message_count=prev_input_len, token_count=0, source_event_id=prev_id),
                InputSegment(type="output", message_count=1, token_count=0, source_event_id=prev_id),
                InputSegment(type="unique", message_count=len(results), token_count=0, source_event_id=None),
            ]
            turn_id = f"{agent_prefix}:t{t}"
            turn_wait = int(
                sample_from_distribution(cfg.tool_call_latency_sec, 1, rng=child_rng(seed, *agent_seed_path, t, 3))[0] * 1000
            )
            is_last_turn = t == k - 1
            turn_is_terminal = is_last_turn and not will_spawn
            if turn_is_terminal:
                # OUTPUT is the plain answer.
                ans_text, ans_tokens = _answer_text(agent_seed_path)
                _emit(
                    turn_id,
                    turn_msgs,
                    [prev_id],
                    {prev_id: "full_match"},
                    turn_segs,
                    turn_wait,
                    False,
                    None,
                    expected_output=ans_text,
                    expected_output_tokens=ans_tokens,
                )
                next_out_calls: List[Dict[str, Any]] = []
            else:
                # OUTPUT is the NEXT tool call (turn t+1); force it via tool_names.
                next_calls, _, next_names = _turn_calls_and_results(t + 1)
                _emit(
                    turn_id,
                    turn_msgs,
                    [prev_id],
                    {prev_id: "full_match"},
                    turn_segs,
                    turn_wait,
                    True,
                    next_names,
                )
                next_out_calls = next_calls
            prev_id = turn_id
            prev_input_len = len(turn_msgs)
            prev_msgs = turn_msgs
            prev_out_calls = next_out_calls

        # --- optional fan-out: spawn K sub-agents + one merge (merge is terminal) ---
        if will_spawn:
            K = sample_int(cfg.sub_agents_per_spawn, child_rng(seed, *agent_seed_path, 8), _FB_SUB_AGENTS)
            K = max(0, K)
            # Whole-spawn minimum cost: per child a dispatch event (1) + minimal
            # child (one terminal call), plus one shared merge event (which is
            # this agent's terminal — no separate answer follows). Only spawn if
            # it all fits; otherwise this agent stays a plain leaf whose current
            # `prev` event must become terminal.
            min_spawn_cost = K * (1 + _min_agent_cost()) + 1
            spawned = False
            if K > 0 and len(events) + min_spawn_cost <= budget:
                dispatch_pairs: List[tuple] = []  # (dispatch_id, tc_id)
                child_terminals: List[str] = []
                spawn_ok = True
                for c in range(K):
                    # --- single-call dispatch event (fresh [user] context; its
                    # OUTPUT is the one dispatch_agent tool_call — never dangles,
                    # and a fresh [user] input never ends in assistant) ---
                    disp_id = f"{agent_prefix}:d{depth}:disp{c}"
                    tc_id = f"dispatch_{agent_prefix}_{c}"
                    child_obj = _render_objective(theme, child_rng(seed, *agent_seed_path, c, 1))
                    dispatch_ctx = {"role": "user", "content": f"Dispatch a sub-agent to: {child_obj}"}
                    _emit(
                        disp_id,
                        [dispatch_ctx],
                        [prev_id],
                        {prev_id: "full_match"},
                        [],
                        0,
                        True,
                        [DISPATCH_AGENT_NAME],
                        defs=fanout_tool_defs,
                    )
                    # --- recurse into the child agent (depth+1); child starts CLEAN ---
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
                        spawn_ok = False
                        break
                    dispatch_pairs.append((disp_id, tc_id))
                    child_terminals.append(child_terminal)

                if spawn_ok and dispatch_pairs:
                    # --- ONE merge event: the parent's pre-spawn transcript
                    # (shared-only prepend — introduces NO unmatched tool_call, so
                    # it can never dangle) followed by [dispatch call, child
                    # result] pairs. The merge is the agent TERMINAL: its OUTPUT is
                    # the plain answer (no trailing answer event). ---
                    merge_segs: List[InputSegment] = [
                        InputSegment(type="shared", message_count=prev_input_len, token_count=0, source_event_id=prev_id),
                    ]
                    merge_msgs: List[Dict[str, Any]] = list(prev_msgs)
                    merge_preds: List[str] = [prev_id]
                    merge_deps: Dict[str, str] = {prev_id: "full_match"}
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
                        merge_segs.append(InputSegment(type="output", message_count=1, token_count=0, source_event_id=disp_id))
                        merge_segs.append(
                            InputSegment(type="tool_output", message_count=1, token_count=0, source_event_id=child_term)
                        )
                        merge_preds += [disp_id, child_term]
                        merge_deps[disp_id] = "full_match"
                        merge_deps[child_term] = "full_match"
                    merge_id = f"{agent_prefix}:d{depth}:merge"
                    ans_text, ans_tokens = _answer_text(agent_seed_path)
                    _emit(
                        merge_id,
                        merge_msgs,
                        merge_preds,
                        merge_deps,
                        merge_segs,
                        0,
                        False,
                        None,
                        defs=fanout_tool_defs,
                        expected_output=ans_text,
                        expected_output_tokens=ans_tokens,
                    )
                    prev_id = merge_id
                    spawned = True
            # If the spawn was rolled but did not fit / produced no children
            # (spawned is False), the current `prev` event may still advertise a
            # tool call as its output; the final normalization below re-emits it
            # as a plain-answer terminal.
            _ = spawned

        # Final normalization: the terminal event must OUTPUT the plain answer,
        # never a forced-but-unconsumed tool call. This only fires when the tool
        # loop was truncated early by the budget (so the last turn never became
        # terminal) and no spawn happened; the normal paths already set the
        # terminal's output to the answer above.
        term_ev = events[prev_id]
        if term_ev.call.expected_output_is_tool_call:
            ans_text, ans_tokens = _answer_text(agent_seed_path)
            _emit(
                prev_id,
                term_ev.call.messages,
                term_ev.predecessor_event_ids,
                term_ev.predecessor_dependency_types,
                term_ev.call.input_segments,
                term_ev.wait_ms,
                False,
                None,
                defs=term_ev.call.tool_definitions,
                expected_output=ans_text,
                expected_output_tokens=ans_tokens,
            )

        return prev_id

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
