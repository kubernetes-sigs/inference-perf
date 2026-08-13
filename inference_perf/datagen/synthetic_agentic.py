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
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inference_perf.config import APIConfig, DataConfig
from inference_perf.config.datagen.replay import SyntheticAgenticConfig
from inference_perf.config.common import Distribution
from inference_perf.datagen.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph
from inference_perf.datagen.synthetic_themes import (
    GENERIC_THEME,
    ROOT_SYSTEM_PROMPTS,
    SUBAGENT_SYSTEM_PROMPTS,
    Theme,
    load_theme,
)
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


def _pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    """Deterministically pick one element of `seq` using `rng` (one draw)."""
    return seq[int(rng.integers(0, len(seq)))]


# --- Filler fitting -------------------------------------------------------
#
# Free-text turns (e.g. an agent's objective/summary line) are padded with filler
# so the turn's token count matches a sampled target, while keeping the "real"
# content the model should attend to distinguishable from the padding. The padding
# is wrapped in a <context>...</context> block that bounds it on both sides and
# frames it as supplementary background; the real content is emitted LAST, after the
# block (the highest-attention recency slot). The framing is soft ("supplementary
# background, focus on the task below") rather than "ignore this filler", because a
# theme's filler is domain-relevant (log lines / metric rows), so calling it
# garbage-to-ignore would read oddly; steering the model to the task keeps the
# padding non-load-bearing without claiming plausible content is noise. The framing
# lives INSIDE the block so it stays self-contained if this content is later
# re-injected into a growing transcript.
#
# TOOL_CALL_MARGIN is token headroom reserved so a tool-call turn's fixed overhead
# doesn't blow past its target; it lives here as part of the same token-budgeting
# vocabulary as fit_filler.

TOOL_CALL_MARGIN = 64
# Opening head of the filler block (emitted BEFORE the padding words) and its
# closing tag (emitted AFTER them). The real content follows the close tag.
FILLER_OPEN = (
    "<context>\n"
    "Supplementary background (logs / metrics / notes gathered while triaging). "
    "This is context only, not the primary task; focus on the task stated after this block.\n"
)
FILLER_CLOSE = "\n</context>"


def _tool_call_max_tokens(tokenizer: CustomTokenizer, calls: List[Dict[str, Any]]) -> int:
    """max_tokens for a FORCED tool-call turn = tokens(serialized calls) + margin.

    A forced tool-call event must give the replay model enough budget to emit the
    WHOLE tool call (function name + arguments + the model's tool-call framing). Too
    small a budget truncates generation mid-JSON, and a real model then leaks
    chat-template control tokens (<|im_end|> etc.) into the `arguments` string;
    replaying that malformed assistant message 400s the next turn. TOOL_CALL_MARGIN
    covers the model's per-call preamble on top of the exact serialized calls.
    Deterministic: a pure function of `calls` + the tokenizer, no rng."""
    if not calls:
        return TOOL_CALL_MARGIN
    return int(tokenizer.count_tokens(json.dumps(calls))) + TOOL_CALL_MARGIN


def _accumulated_wire_tokens(
    tokenizer: CustomTokenizer, transcript_msgs: List[Dict[str, Any]], tool_defs: List[Dict[str, Any]]
) -> int:
    """Approximate the prefill a round's principal would send: accumulated message
    content + the advertised tool catalog.

    Used by context compaction to decide -- BEFORE building a round -- whether the
    grown transcript has crossed the trigger. `transcript_msgs` is the prior
    principal's full input placeholder, so its content already includes every prior
    turn AND every prior (budgeted) assistant output folded in. The tool catalog is
    added because the server prefills it too and it dominates a large catalog (the
    same catalog the client-side prompt_tokens count omits). Deterministic: a pure
    function of the inputs."""
    content = "\n".join(str(m.get("content") or "") for m in transcript_msgs)
    catalog = int(tokenizer.count_tokens(json.dumps(tool_defs))) if tool_defs else 0
    return int(tokenizer.count_tokens(content)) + catalog


# Shakespeare corpus shipped with the repo; same file/location convention
# used by synthetic_datagen.py and weka_trace_replay_datagen.py for prompt
# corpora. Loaded lazily (not at import time) and cached in-process.
_SHAKESPEARE_PATH = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"
_corpus_words_cache: Optional[List[str]] = None


def _corpus_words() -> List[str]:
    """Return the Shakespeare corpus split into whitespace-delimited words.

    This is the FALLBACK filler source, used only for themes that do NOT carry
    their own `filler_templates`. No shared corpus-word loader exists elsewhere
    in the codebase to reuse (synthetic_datagen.py / weka_trace_replay_datagen.py
    each inline their own read of assets/shakespeare.txt and feed it straight
    through the tokenizer, rather than exposing a word list); this mirrors their
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


# Number of filler_templates snippets rendered (each with its own seeded field
# values) to build a theme's filler word pool. A handful of rendered lines gives
# enough lexical variety; the pool is then CYCLED to reach any target size, so
# this need not be large. Kept small so pool construction stays cheap.
_FILLER_POOL_RENDER_COUNT = 24


def _render_word_pool(theme: Theme, templates: List[str], seed: int, path: Tuple[int, ...]) -> Optional[List[str]]:
    """Render a list of theme snippet templates into a whitespace-split word pool.

    Renders `_FILLER_POOL_RENDER_COUNT` snippets (cycling `templates`), each seeded
    independently so the pool reads like a paste of MORE of the same content.
    Returns None if `templates` is empty. Deterministic for a given (seed, path).
    """
    if not templates:
        return None
    words: List[str] = []
    for i in range(_FILLER_POOL_RENDER_COUNT):
        tpl = templates[i % len(templates)]
        rendered = _render_theme_template(theme, tpl, seed, (*path, i))
        words.extend(rendered.split())
    return words or None


def theme_filler_words(theme: Theme, seed: int, path: Tuple[int, ...]) -> Optional[List[str]]:
    """Build a DOMAIN-appropriate filler word pool from a theme's `filler_templates`
    (log lines / metric rows / stack frames), used to pad turns. Returns None when the
    theme carries none, signalling `fit_filler` to fall back to the shared Shakespeare
    corpus. `path` is the reserved seed sub-path prefix. Deterministic per (seed, path).
    """
    return _render_word_pool(theme, theme.filler_templates or [], seed, path)


def theme_payload_words(theme: Theme, seed: int, path: Tuple[int, ...]) -> Optional[List[str]]:
    """Build the word pool for LARGE tool-call PAYLOAD args (content/code/patch/...).

    Uses the theme's `payload_templates` (domain payload shape: code / SQL / a drafted
    answer) so a payload looks like what the tool carries, NOT like the telemetry that
    pads turns. Falls back to `filler_templates` when a theme declares no payload
    templates, so a payload then reads like that theme's filler. Deterministic per
    (seed, path).
    """
    pool = _render_word_pool(theme, theme.payload_templates or [], seed, path)
    return pool if pool is not None else theme_filler_words(theme, seed, path)


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


def _untruncated_len(tokenizer: CustomTokenizer, text: str) -> int:
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


def fit_filler(
    tokenizer: CustomTokenizer,
    target_tokens: int,
    fixed_content: str,
    rng: Optional[np.random.Generator],
    word_pool: Optional[List[str]] = None,
) -> str:
    """Pad `fixed_content` with filler to approximate `target_tokens`.

    Filler source: `word_pool` when supplied (a theme's own domain word pool,
    from `theme_filler_words`, so the padding reads like more pasted log/metric
    content), otherwise the shared Shakespeare corpus. Falling back to the
    corpus keeps themes without `filler_templates` working unchanged.

    Layout: the filler is wrapped in a <context>...</context> block emitted FIRST,
    then the real `fixed_content` LAST (see FILLER_OPEN/FILLER_CLOSE), so the
    model is steered to the real content in the highest-attention recency slot
    while the padding is framed as supplementary background (not noise).

    filler_budget = target_tokens - count_tokens(FILLER_OPEN + FILLER_CLOSE + " " + fixed_content).

    Budget guard: if filler_budget <= 0 the target is too small to even fit the
    fixed content plus the wrapper -- flooring to `fixed_content` alone (no
    wrapper, no filler) is the only crash-free option, so that's what happens.
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
    # Fixed cost = wrapper tags + inline instruction + the real content: all
    # mandatory (unpaddable) tokens the filler budget must account for.
    fixed_cost = tokenizer.count_tokens(FILLER_OPEN + FILLER_CLOSE + " " + fixed_content)
    filler_budget = target_tokens - fixed_cost
    if filler_budget <= 0:
        logger.debug(
            "fit_filler: non-positive filler budget (target_tokens=%d, fixed_cost=%d); "
            "flooring to fixed_content with no wrapper/filler",
            target_tokens,
            fixed_cost,
        )
        return fixed_content

    # Theme word pool if provided (domain-appropriate padding), else the shared
    # Shakespeare corpus. Both are just word lists cycled to reach the target.
    words = word_pool if word_pool else _corpus_words()
    if not words:
        # No corpus to pad with -> emit the real content alone (a hollow
        # <context></context> block would signal nothing and just waste tokens).
        return fixed_content

    # Average tokens-per-word from a small, un-truncated sample (measured once).
    sample = _cycled_words(words, min(_RATIO_SAMPLE_WORDS, len(words)))
    sample_text = " ".join(sample)
    sample_tokens = _untruncated_len(tokenizer, sample_text)
    tokens_per_word = (sample_tokens / len(sample)) if sample and sample_tokens > 0 else 1.0

    def _emit(n_words: int) -> str:
        # Wrapped filler FIRST, real content LAST.
        chunk = " ".join(_cycled_words(words, max(1, n_words)))
        return f"{FILLER_OPEN}{chunk}{FILLER_CLOSE} {fixed_content}"

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


# Header that introduces the filler padding appended AFTER a real system prompt,
# when the prompt is shorter than the requested head length. Unlike fit_filler
# (which frames filler as supplementary <context> and puts the real content
# last), the system head keeps the REAL prompt FIRST -- it is the standing
# instruction the agent must attend to -- and pads the remainder with a clearly
# labeled operational-context block, so the head reads like a real system prompt
# topped up with background rather than duplicated or filler-fronted text.
_SYSTEM_HEAD_FILLER_HEADER = "\n\n## Operational context\n"


def _render_system_head(
    tokenizer: CustomTokenizer,
    target_tokens: int,
    is_root: bool,
    rng: np.random.Generator,
    word_pool: Optional[List[str]] = None,
) -> str:
    """Render an agent's system head: a real, role-appropriate system prompt fitted
    to `target_tokens`.

    A prompt is drawn (seeded) from ROOT_SYSTEM_PROMPTS (orchestrator/assistant) or
    SUBAGENT_SYSTEM_PROMPTS (spawned worker) by role. The REAL prompt is kept whole
    and FIRST; if it is shorter than the target, a labeled "## Operational context"
    block padded with filler makes up the remainder. If the prompt alone already
    meets or exceeds the target, it is truncated (token-wise) to the target -- no
    filler, no duplication. Deterministic given `rng`.
    """
    pool = ROOT_SYSTEM_PROMPTS if is_root else SUBAGENT_SYSTEM_PROMPTS
    prompt = str(_pick(rng, pool))
    prompt_tokens = tokenizer.count_tokens(prompt)

    if prompt_tokens >= target_tokens:
        # Real prompt already fills the budget: truncate to the target (keep the
        # opening, which carries the role + policy) rather than pad. Truncate by
        # words and correct down until it fits, so we never exceed the target.
        words = prompt.split()
        # Proportional first cut, then trim word-by-word to land at/under target.
        keep = max(1, int(len(words) * target_tokens / max(1, prompt_tokens)))
        while keep > 1 and tokenizer.count_tokens(" ".join(words[:keep])) > target_tokens:
            keep -= 1
        return " ".join(words[:keep])

    # Prompt fits with room to spare: append a labeled filler block for the
    # remainder. filler_budget accounts for the prompt + the header (mandatory).
    words = word_pool if word_pool else _corpus_words()
    header_cost = tokenizer.count_tokens(prompt + _SYSTEM_HEAD_FILLER_HEADER)
    filler_budget = target_tokens - header_cost
    if filler_budget <= 0 or not words:
        return prompt

    sample = _cycled_words(words, min(_RATIO_SAMPLE_WORDS, len(words)))
    sample_tokens = _untruncated_len(tokenizer, " ".join(sample))
    tokens_per_word = (sample_tokens / len(sample)) if sample and sample_tokens > 0 else 1.0

    def _emit(n_words: int) -> str:
        chunk = " ".join(_cycled_words(words, max(1, n_words)))
        return f"{prompt}{_SYSTEM_HEAD_FILLER_HEADER}{chunk}"

    n_words = max(1, int(np.ceil(filler_budget / tokens_per_word)))
    buf = _emit(n_words)
    actual = _untruncated_len(tokenizer, buf)
    filler_actual = actual - header_cost
    if actual != target_tokens and filler_actual > 0:
        observed_ratio = filler_actual / n_words
        corrected = max(1, int(np.ceil(filler_budget / observed_ratio)))
        if corrected != n_words:
            buf = _emit(corrected)
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

# Fallbacks for optional distribution knobs (used when the config leaves them unset).
_FB_STEPS = Distribution(type="fixed", mean=2)
_FB_TOOL_DEFS = Distribution(type="fixed", mean=8)
_FB_SUB_AGENTS = Distribution(type="uniform", min=2, max=4)
_FB_PARALLEL = Distribution(type="fixed", mean=1)
# Wait-time fallbacks (seconds), each set to a realistic value for what it models:
# a tool round-trip is ~1s; a human reading an answer and typing a follow-up is ~10s.
_FB_TOOL_LATENCY = Distribution(type="fixed", mean=1)
_FB_USER_THINK = Distribution(type="fixed", mean=10)
# Output size of a non-terminal async-notification ack turn ("one sub-agent reported,
# still waiting on the rest").
#
# Must stay ABOVE fit_filler's unpaddable fixed cost for the ack's own fixed content
# (wrapper tags + the sentence, ~34 tokens): below that the budget guard floors to the
# fixed content alone, which would make every ack in a chain the identical literal and
# render the per-notification RNG salts inert. 48 clears that floor while keeping the
# turn short-form.
_FB_ACK_TOKENS = Distribution(type="fixed", mean=48)

# Minimum events an agent occupies: one principal call whose output IS the answer
# (tool loop depth 0, no spawn). A tool loop or a spawn adds its own events on top,
# guarded separately at each emit.
_MIN_AGENT_COST = 1

# Appended to a SPAWNED sub-agent's objective so its terminal turn concludes with
# a prose SUMMARY (the report the orchestrator receives via the async notification's
# `async_report` slot) instead of tool-call text. Real sub-agents return a textual
# summary of their outcome, not a tool call or raw scratch work. Kept THEME-NEUTRAL
# ("outcome of your task", not "findings") because a sub-agent may perform an
# action/migration/generation, not only an investigation. The two load-bearing
# parts (verified against the live model to yield prose + finish=stop, with the
# tool catalog still advertised so the prefix cache is preserved) are "plain prose"
# and "do not call a tool in your final message". Only children get this; a
# root/single-agent answer is not a report to anyone.
SUBAGENT_REPORT_DIRECTIVE = (
    "This is your final turn; no further tool calls will be executed. Report back to the "
    "orchestrator now: summarize what you found and concluded in 2-3 sentences of plain prose. "
    "Do not emit a tool call."
)

# Appended to the ROOT agent's terminal turn (the turn that answers the USER, not an
# orchestrator) so its final message is prose, not tool-call text. Without it a tool-primed
# root, whose transcript ends in tool results, tends to emit a `<tool_call>{...}` block as
# plain text on the answer turn even under tool_choice="none" (the API suppresses a
# STRUCTURED call, but the model can still WRITE tool-call syntax). Same two load-bearing
# parts as the sub-agent directive ("plain prose" + "do not call a tool"), reworded for a
# user-facing answer. Cosmetic/realism only: benchmark-neutral (request shape, growth, and
# termination are unaffected). The root/single-agent answer is to the user, not a report.
ROOT_ANSWER_DIRECTIVE = (
    "This is your final turn; no further tool calls will be executed. Provide your final answer "
    "to the user now, in plain prose. Do not emit a tool call."
)

# The tool RESULT for an async dispatch_agent call. Content-free by design: an async
# harness acknowledges the launch immediately and delivers the child's report LATER,
# out of band, as its own user-role notification message.
ASYNC_DISPATCH_STUB = "Async agent launched successfully. You will be notified automatically when it completes."

# Canonical structural tool used to spawn sub-agents. It is NOT a theme tool:
# it must be advertised on any event that FORCES it (dispatch events, via
# expected_output_tool_names) or EMITS it (the merge event, which stores
# dispatch_agent tool_calls in its message history). Every forced/emitted tool name
# must appear in that turn's tool_definitions with a top-level `name` key, so its
# shape mirrors `_tool_definitions` output exactly.
# Generic non-empty parameter schema for any tool a theme did not give an
# explicit spec (and for synthetic suffixed duplicates). A single required
# string field is enough to anchor the model's forced tool-call generation so it
# emits `{"query": "..."}` and stops cleanly, instead of empty `{}` + token leak.
_FALLBACK_TOOL_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Free-text query for the operation."}},
    "required": ["query"],
}

# String tool-call args whose whole point is a LARGE payload (a file's new
# contents, a patch, a code block, a query body). For these, the tiny `f"{prop}-NNN"`
# stub is unrealistic; instead draw a chunk of the theme's own filler pool
# (code-shaped for a coding theme, log/metric-shaped for an ops theme) so the call
# carries realistic content + token weight. Non-payload string args (path, symbol,
# pattern) keep the short stub.
#
# SIZE is a PER-TOOL, PER-ARGUMENT property, set by the theme on the argument's schema
# via `"x-payload-tokens": N` (in words). This lives with the tool (like its schema,
# description, and result template) because how big a payload a given tool carries is a
# property of THAT tool -- a `write_file` writes a whole file; an `apply_patch` a diff;
# an `execute_code` a script. When a payload arg has no `x-payload-tokens`, it falls
# back to `_DEFAULT_PAYLOAD_WORDS` (a modest chunk). The user picks the workload by
# choosing the theme; there is no confusing global knob whose effect depends on which
# tools happen to be advertised.
_PAYLOAD_ARG_NAMES = frozenset({"content", "code", "patch", "diff", "body", "new_string", "old_string", "text", "snippet"})
_DEFAULT_PAYLOAD_WORDS = 48  # fallback when the theme sets no x-payload-tokens on the arg
_PAYLOAD_TOKENS_KEY = "x-payload-tokens"  # per-arg schema hint (in words) for payload size

DISPATCH_AGENT_NAME = "dispatch_agent"
# The dispatch tool is ASYNC: the call returns a content-free acknowledgment
# immediately (ASYNC_DISPATCH_STUB) and each sub-agent's actual report arrives
# LATER, out of band, as its own user-role notification message. The description
# states that contract, including what to do on a notification that still leaves
# reports outstanding -- so the "acknowledge briefly and keep waiting" behavior
# comes from the TOOL's contract.
DISPATCH_AGENT_DESCRIPTION = (
    "Launch a sub-agent to work on `objective` asynchronously. Returns immediately with a "
    "launch acknowledgment, NOT the sub-agent's result: each sub-agent's report is delivered "
    "later as a separate user message wrapping that report in "
    "<task-notification><result>...</result></task-notification>. Reports arrive ONE AT A TIME "
    "and in completion order. Reply to each incoming report with a "
    "one-sentence acknowledgment (e.g., 'Sub-agent has finished, waiting for others'), and keep waiting."
)
DISPATCH_AGENT_TOOL_DEF: Dict[str, Any] = {
    "name": DISPATCH_AGENT_NAME,
    "type": "function",
    "description": DISPATCH_AGENT_DESCRIPTION,
    "function": {
        "name": DISPATCH_AGENT_NAME,
        "description": DISPATCH_AGENT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {"objective": {"type": "string", "description": "The sub-agent's task."}},
            "required": ["objective"],
        },
    },
}


def _tool_definitions(theme: Theme, n: int) -> List[Dict[str, Any]]:
    """Build `n` tool definitions, each with a TOP-LEVEL `name` key (required so a
    forced/emitted call name is always an advertised tool) and a human-readable
    description.

    Cycles the theme's tool_names and suffixes duplicates so names stay unique
    when the requested catalog is larger than the theme's list. The description
    is looked up in `theme.tool_descriptions` by the BASE tool name (so a
    synthetic suffixed duplicate `get_bp_stats_7` reuses `get_bp_stats`'s
    description); missing entries fall back to a generic sentence. It is emitted
    into BOTH the top-level `description` and the nested `function.description`
    (OpenAI tool schema) -- the top-level `name` is preserved for the runtime.
    """
    out: List[Dict[str, Any]] = []
    names = theme.tool_names  # _validate guarantees non-empty
    descriptions = theme.tool_descriptions or {}
    parameters = theme.tool_parameters or {}
    for i in range(n):
        base = names[i % len(names)]
        name = base + ("" if i < len(names) else f"_{i}")
        desc = descriptions.get(base) or f"Perform the {base} operation and return its result."
        # Real parameter schema from the theme (keyed by BASE name, so suffixed
        # duplicates reuse it). Fall back to a generic non-empty schema so NO tool
        # is ever parameterless: a forced tool_choice on a no-arg tool makes some
        # models emit empty args then fail to stop, leaking template tokens into
        # `arguments` (observed on Qwen -> the tool-call 400s on replay).
        params = parameters.get(base) or _FALLBACK_TOOL_PARAMS
        out.append(
            {
                "name": name,
                "description": desc,
                "type": "function",
                "function": {"name": name, "description": desc, "parameters": params},
            }
        )
    return out


def _entity_subs(theme: Theme, rng: np.random.Generator, pinned: Dict[str, str]) -> Dict[str, str]:
    """Template substitutions for a subject line: a leading `verb` draw, then one
    value per declared entity category. A pinned category uses its fixed value and
    skips the draw, so the text references the session's fixed subject.
    """
    subs: Dict[str, str] = {"verb": _pick(rng, theme.verbs)}
    for key, vals in theme.entities.items():
        if vals:
            subs[key] = pinned.get(key) or _pick(rng, vals)
    return subs


def _render_objective(theme: Theme, rng: np.random.Generator, pinned: Optional[Dict[str, str]] = None) -> str:
    """Render a single principal objective string from the theme templates.

    `pinned` (optional) maps an entity category name -> a fixed value that
    overrides the per-category draw, so the objective references the SAME
    primary subject (e.g. `service`/`db_instance`, `symptom`) as the round's
    intro doc + follow-ups. `verb` is drawn from `theme.verbs` (never pinned).
    """
    subs = _entity_subs(theme, rng, pinned or {})
    try:
        return theme.objective_template.format(**subs)
    except (KeyError, IndexError):
        return f"{subs['verb']}: complete the task."


def _render_followup(theme: Theme, objective: str, rng: np.random.Generator, pinned: Optional[Dict[str, str]] = None) -> str:
    """Render a round-K (K>=1) follow-up principal turn from the theme.

    Uses the theme's followup_templates (optionally prefixed by a
    followup_connective) when present; otherwise falls back to the objective so
    the follow-up turn is never empty. The result is the fixed_content that
    input_tokens_per_turn sizing (fit_filler) later pads.

    `pinned` overrides the per-category entity draw (same contract as
    `_render_objective`) so multi-round conversations stay on the same subject.
    """
    pinned = pinned or {}
    connective = ""
    if theme.followup_connectives:
        connective = _pick(rng, theme.followup_connectives)
    if theme.followup_templates:
        tpl = _pick(rng, theme.followup_templates)
        subs: Dict[str, str] = {}
        for key, vals in theme.entities.items():
            if vals:
                subs[key] = pinned.get(key) or _pick(rng, vals)
        try:
            rendered = tpl.format(**subs)
        except (KeyError, IndexError):
            return connective + objective
        return connective + _join_connective_case(connective, rendered, theme)
    return connective + objective


def _join_connective_case(connective: str, rendered: str, theme: Theme) -> str:
    """Fix the casing seam when a connective is prepended to a follow-up.

    A connective like "Following up, " ends mid-sentence (word + comma + space),
    so the template that follows should continue in lower case: raw concatenation
    yields "Following up, Are other services..." — a capital right after a
    lowercase connective, an obvious concatenation seam. We lowercase the first
    alphabetic character of `rendered` ONLY when it is safe: the first token must
    be a COMMON word (nothing that would be corrupted by lowercasing).

    A first token is left untouched (proper noun / entity / acronym) when it
    contains a digit (e.g. `Db2`, `DBP1`), is all-caps (an acronym), or is an
    entity value from any of the theme's pools (e.g. a service name like
    `cart-service`). Otherwise it is a common word and we lowercase its first
    letter. If unsure we leave it as-is, so entity names are never corrupted.

    Deterministic + side-effect free: no rng, no wall-clock.
    """
    if not connective or not rendered:
        return rendered
    # Only touch a connective that continues mid-sentence: ends with a space
    # (the template text follows it on the same clause). A connective without a
    # trailing space would abut the text directly and is not our seam case.
    if not connective.endswith(" "):
        return rendered
    first_token = rendered.split(maxsplit=1)[0]
    # Entity values (service/db_instance/... names) are proper nouns -> preserve.
    # An entity value may itself contain a space (e.g. a symptom phrase), so we
    # also guard on the rendered text STARTING with any entity value, not just
    # the whitespace-split first token.
    entity_pool = {v for vals in theme.entities.values() for v in vals if v}
    has_digit = any(c.isdigit() for c in first_token)
    is_all_caps = first_token.isupper()
    is_entity = first_token in entity_pool or (bool(entity_pool) and rendered.startswith(tuple(entity_pool)))
    if has_digit or is_all_caps or is_entity:
        return rendered
    # Common word: lowercase the FIRST alphabetic character only.
    for i, ch in enumerate(rendered):
        if ch.isalpha():
            return rendered[:i] + ch.lower() + rendered[i + 1 :]
        if not ch.isspace():
            # first non-space is punctuation/symbol -> not a word to case-fix
            return rendered
    return rendered


def _render_compaction_summary(
    theme: Theme, tool_defs: List[Dict[str, Any]], rng: np.random.Generator, pinned: Optional[Dict[str, str]] = None
) -> str:
    """Render a semantic recap sentence for a context-compaction round.

    Composed from what the generator already knows at build time -- the session's
    pinned subject (service/symptom/region/... via `pinned`) and the real tools the
    agent advertises (`tool_defs`) -- so the recap names the true investigation and
    the real tools it used, WITHOUT summarizing live results (which don't exist until
    replay) and WITHOUT a model call (determinism). Fills the theme's
    `compaction_summary_template` with `{verb}` + entity/pinned placeholders (like
    `_render_objective`) plus `{tool_a}`/`{tool_b}`/`{tool_c}` drawn from the catalog.
    Returns "" when the theme defines no template (caller then uses the bare marker).
    Deterministic: all draws come from `rng`.
    """
    tpl = theme.compaction_summary_template
    if not tpl:
        return ""
    subs = _entity_subs(theme, rng, pinned or {})
    # Tool slots: fill up to three from the advertised catalog (repeat the last / a
    # placeholder when the catalog is smaller), so the recap lists REAL tools. These
    # are deterministic index picks, not rng draws.
    names = [str(td.get("name", "")) for td in tool_defs if td.get("name")]
    for idx, slot in enumerate(("tool_a", "tool_b", "tool_c")):
        subs[slot] = names[idx] if idx < len(names) else (names[-1] if names else "the tools")
    try:
        return tpl.format(**subs)
    except (KeyError, IndexError):
        return ""


def _is_time_field(field: str) -> bool:
    f = field.lower()
    return f.startswith("t") and (f in ("t", "time") or f[1:].isdigit() or "time" in f or f.endswith("_t") or "_t" in f)


def _is_numeric_field(field: str) -> bool:
    f = field.lower()
    if any(c.isdigit() for c in f):
        return True
    return any(tok in f for tok in ("n", "ms", "count", "wait"))


# --- Bounded numeric classes -------------------------------------------------
#
# A bare `{nN}` renders as an int 0..998 (fine for opaque tallies), but many
# fields are semantically BOUNDED and an out-of-range value gives the synthetic
# origin away ("success rate 273%"). The renderer cannot sniff surrounding text
# reliably, so the placeholder NAME carries the intent: templates use names like
# `{hit_ratio0}`/`{p99_ms}`/`{status0}` and these classifiers (checked BEFORE
# the generic `_is_numeric_field` -> 0..998 fallback) render a plausible value.
#
# A trailing digit index is stripped first so `pct`, `pct0`, `pct1` all match
# the same class (and duplicate distinct names still get independent sub-seeds).


def _strip_index(field: str) -> str:
    """Lowercase `field` and strip any trailing run of digits (the index)."""
    f = field.lower()
    return f.rstrip("0123456789")


def _is_percent_field(field: str) -> bool:
    """percent / ratio / rate class -> a bounded 0..100 value."""
    f = _strip_index(field)
    return f in ("pct", "percent", "ratio", "rate", "hit_ratio") or f.endswith("_pct") or f.endswith("_ratio")


def _is_latency_ms_field(field: str) -> bool:
    """latency-in-ms class -> a plausible millisecond magnitude (1..2000)."""
    f = _strip_index(field)
    if f.endswith("_ms") or f.startswith("ms"):
        return True
    return f in ("p50", "p90", "p95", "p99", "p50_ms", "p99_ms", "latency", "latency_ms", "dur", "duration")


def _is_error_pct_field(field: str) -> bool:
    """error-rate percent class -> a LOW percentage (0..15).

    An error rate is a percentage, but unlike hit-ratio/budget/success-rate
    (which read high) a healthy-ish service under a SEV sits LOW. Any field
    whose name mentions an error AND reads as a rate/percent (`error_rate_pct`,
    `err_pct`, `err_rate`) belongs here. Checked BEFORE `_is_percent_field` so
    an `_pct`/`rate` error field is not caught by the high-percent class, and
    BEFORE `_is_count_field` so `err_rate` is not misread as a raw tally.
    Raw error COUNTS (`errors`, `err`) do NOT match -- they carry no rate/pct
    signal and stay in the count class.
    """
    f = _strip_index(field)
    if "err" not in f:
        return False
    return f.endswith("_pct") or f.endswith("_rate") or f in ("err_rate", "err_pct", "error_rate")


def _is_count_field(field: str) -> bool:
    """small-count class -> a modest integer 0..500."""
    f = _strip_index(field)
    if f.startswith("count"):
        return True
    return f in (
        "retries",
        "errors",
        "err",
        "rps",
        "req_per_sec",
        "spans",
        "in_use",
        "idle",
        "max",
        "healthy",
        "unhealthy",
    )


def _is_status_code_field(field: str) -> bool:
    """HTTP-status class -> one of a realistic weighted set (200 common)."""
    return _strip_index(field) in ("status", "status_code", "http_status")


# Realistic HTTP status set with 200 weighted common (deterministic pick below).
_STATUS_CODES = (200, 200, 200, 301, 400, 404, 429, 500, 502, 503, 504)


def _seeded_percent_value(rng: np.random.Generator) -> str:
    """A plausible percentage as a one-decimal float in [80.0, 100.0].

    Chosen over a plain 0..100 int because SLO/hit-ratio/success-rate fields
    read most like real telemetry when they sit high with one decimal place;
    still strictly bounded to <= 100 so nothing ever exceeds a valid ratio.
    """
    return f"{float(rng.integers(800, 1001)) / 10.0:.1f}"


def _seeded_error_pct_value(rng: np.random.Generator) -> str:
    """A LOW error percentage as a one-decimal float in [0.0, 15.0].

    Error rates read low even during an incident (a few percent 5xx is a real
    SEV); kept bounded so it is plausible AND always <= 100.
    """
    return f"{float(rng.integers(0, 151)) / 10.0:.1f}"


def _seeded_latency_ms_value(rng: np.random.Generator) -> str:
    return str(int(rng.integers(1, 2001)))


def _seeded_count_value(rng: np.random.Generator) -> str:
    return str(int(rng.integers(0, 501)))


def _seeded_status_code_value(rng: np.random.Generator) -> str:
    return str(_pick(rng, _STATUS_CODES))


def _seeded_time_value(rng: np.random.Generator) -> str:
    hh = int(rng.integers(0, 24))
    mm = int(rng.integers(0, 60))
    ss = int(rng.integers(0, 60))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _seeded_entity_value(theme: Theme, rng: np.random.Generator) -> str:
    pool: List[str] = []
    for vals in theme.entities.values():
        pool.extend(vals)
    if not pool:
        return f"entity-{int(rng.integers(0, 999))}"
    return str(_pick(rng, pool))


def _seeded_typed_entity_value(theme: Theme, category: str, rng: np.random.Generator) -> str:
    """Draw from ONE named entity category's pool (e.g. `plan`, `db_instance`).

    Lets templates use TYPED placeholders (`{plan}`, `{table}`) that stay in
    their own domain -- a `PLAN={plan}` field never renders a symptom string --
    which is what keeps the rendered docs internally coherent. Falls back to the
    mixed pool if the category is empty (defensive; callers only pass known keys).
    """
    vals = theme.entities.get(category) or []
    if not vals:
        return _seeded_entity_value(theme, rng)
    return str(_pick(rng, vals))


# The primary-subject entity categories a round pins to keep its intro doc,
# objective, and follow-ups referencing the SAME subject. `service`/`db_instance`
# is the principal system under investigation across the generic + db2 themes;
# `symptom` is the shared incident kind both objective and intro docs name;
# `region` is where the incident lives — an unpinned region let a follow-up ask
# about eu-central-1 while the incident was in us-east-1 (a live model flags the
# drift), so it is pinned per session too. Only categories the theme actually
# declares are pinned (defensive for other themes: a theme without `region`
# simply gets no `region` entry, unaffected).
_PRIMARY_ENTITY_CATEGORIES = ("service", "db_instance", "symptom", "region")


def _pinned_primary_entities(theme: Theme, rng: np.random.Generator) -> Dict[str, str]:
    """Draw ONE fixed value per primary-subject category the theme declares.

    Returned dict is passed to `_render_objective`, `_render_intro_doc`, and
    `_render_followup` so a round's task text and its pasted document name the
    SAME service/subsystem + symptom (a live model flags a doc about one service
    paired with a task about another). Empty when the theme declares none of the
    primary categories, in which case the renderer draws every field freely.
    """
    out: Dict[str, str] = {}
    for cat in _PRIMARY_ENTITY_CATEGORIES:
        vals = theme.entities.get(cat) or []
        if vals:
            out[cat] = _pick(rng, vals)
    return out


def _pinned_focus_entities(theme: Theme, rng: np.random.Generator) -> Dict[str, str]:
    """Draw ONE fixed value per EVERY entity category the theme declares.

    This is the per-session "focus" of an agent's investigation: the ONE file /
    symbol / test / dependency the whole tool loop references. Merged UNDER the
    primary pin (which wins on shared keys) and threaded into every tool CALL's
    args AND every tool RESULT (via result_pinned), so a loop reads as a coherent
    single-target investigation -- `list_dir(focus) -> read_file(focus_file) ->
    grep(focus_symbol) -> run_tests(focus_test)` all reference the same target that
    prior results showed, instead of each turn drawing an unrelated entity.
    Deterministic; drawn once per session at a stable seed sub-path. For a theme with
    only the primary categories, this pin covers exactly those categories.
    """
    out: Dict[str, str] = {}
    for cat, vals in theme.entities.items():
        if vals:
            out[cat] = _pick(rng, vals)
    return out


# --- Numeric invariants in rendered results ---------------------------------
#
# The renderer draws each numeric field from an independent seeded stream, so a
# template with sibling metrics can render an IMPOSSIBLE ordering (p99 < p50,
# heap_used > heap_max) that gives the synthetic origin away. The two helpers
# below repair ordering AFTER all fields are drawn — pure functions of the
# already-seeded values, so determinism is preserved (no new rng draws).
#
# Rendered values are STRINGS in the `values` map (ints or one-decimal floats).
# A field is "numeric-parseable" if it is an int-looking or one-decimal-float
# string; a group containing any non-parseable sibling is skipped safely so we
# never corrupt an entity/time value that happens to share a name pattern.

# Percentile ranks in ascending order; the sort reassigns present siblings so
# p50 <= p90 <= p95 <= p99 within a shared suffix.
_PERCENTILE_RANKS = ("p50", "p90", "p95", "p99")


def _parse_number(s: str) -> Optional[float]:
    """Parse a rendered numeric STRING (int- or one-decimal-float-looking) to a
    float, or None if it is not a plain number. Used to decide whether a paired
    group is numeric-parseable before reordering it."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _format_like(template_value: str, number: float) -> str:
    """Re-render `number` in the SAME textual format as `template_value`: an
    int-looking source stays int-looking, a float-looking source keeps one
    decimal place. Keeps the reassigned value indistinguishable in shape from
    what the draw produced."""
    if "." in template_value:
        return f"{number:.1f}"
    return str(int(round(number)))


def _clamp_le_by_suffix(values: Dict[str, str], lo_base: str, hi_base: str) -> None:
    """Clamp each `{lo_base}{suffix}` to its sibling `{hi_base}{suffix}` (same
    trailing suffix) IN PLACE, wherever both are present + numeric-parseable, so
    `lo <= hi` always holds (e.g. in_use<=max, heap_used<=heap_max).

    Deterministic: operates only on already-seeded string values. A non-numeric
    sibling pair is skipped safely (no corruption of entity/time values).
    """
    for field in list(values):
        base = _strip_index(field)
        if base != lo_base:
            continue
        suffix = field[len(lo_base) :]  # the trailing index, e.g. "" or "0"
        hi_field = hi_base + suffix
        if hi_field not in values:
            continue
        lo = _parse_number(values[field])
        hi = _parse_number(values[hi_field])
        if lo is None or hi is None:
            continue
        if lo > hi:
            values[field] = _format_like(values[field], hi)


def _sort_percentiles_by_suffix(values: Dict[str, str]) -> None:
    """Enforce `p50 <= p90 <= p95 <= p99` among percentile siblings that share a
    suffix, IN PLACE. Handles bare (`p50`,`p99`), `_ms` (`p50_ms`,`p99_ms`), and
    indexed (`p99_0`,`p50_0`, `p50_ms0`,`p99_ms0`) forms.

    A percentile field name is `p{rank}` + a suffix, where `{rank}` is one of
    50/90/95/99; the suffix (whatever trails the rank, e.g. ``, `_ms`, `0`,
    `_ms0`) groups siblings. Within each group we sort the PRESENT values
    ascending and reassign them to the ranks in ascending order, so the smallest
    value lands on the lowest present percentile. Only percentile fields are
    touched; a group with any non-numeric sibling is skipped safely.

    Deterministic: a pure function of the already-seeded values (no new rng).
    """
    # Group field names by their shared suffix; remember each field's rank.
    groups: Dict[str, List[Tuple[int, str]]] = {}  # suffix -> [(rank_index, field), ...]
    for field in values:
        low = field.lower()
        for ri, rank in enumerate(_PERCENTILE_RANKS):
            if low.startswith(rank) and (len(low) == len(rank) or not low[len(rank)].isdigit()):
                # `p50`/`p50_ms`/`p50_ms0` match rank p50; `p500` does NOT
                # (the char after the rank is a digit -> a different number).
                suffix = field[len(rank) :]
                groups.setdefault(suffix, []).append((ri, field))
                break
    for _suffix, members in groups.items():
        if len(members) < 2:
            continue  # nothing to order within a lone percentile field
        members.sort()  # ascending rank order (p50 < p90 < p95 < p99)
        nums = [_parse_number(values[f]) for _, f in members]
        if any(n is None for n in nums):
            continue  # a non-numeric sibling -> skip this group safely
        ordered = sorted(n for n in nums if n is not None)
        for (_, field), num in zip(members, ordered, strict=True):
            values[field] = _format_like(values[field], num)


def _render_theme_template(
    theme: Theme, tpl: str, seed: int, path: Tuple[int, ...], pinned: Optional[Dict[str, str]] = None
) -> str:
    """Fill EVERY `{placeholder}` in a theme template with a real, seeded value.

    Shared by tool-result rendering, filler-snippet rendering, and intro-doc
    rendering so all three classify placeholders identically. Resolution order
    per field name (first match wins):
      0. `pinned[field]` supplied (for `entity` or an `entities` key) -> USE IT
         instead of drawing, so a caller can force the SAME primary subject
         (e.g. `service`/`db_instance`) across the objective, intro doc, and
         follow-ups of one round (coherence).
      1. exactly `entity`                 -> a value from the MIXED entity pool
      2. a key in `theme.entities`        -> a value from THAT category's pool
         (typed placeholder, e.g. `{plan}`/`{db_instance}` -- checked BEFORE the
         numeric/time heuristics so a name like `plan` that happens to contain
         an 'n' is not misread as numeric)
      3. `_is_time_field`                 -> HH:MM:SS
      4. `_is_status_code_field`          -> a realistic HTTP status (200,404,...)
      5. `_is_error_pct_field`            -> a LOW error percent (float in [0,15])
      6. `_is_percent_field`              -> a bounded percent (float in [80,100])
      7. `_is_latency_ms_field`           -> a plausible latency in ms (1..2000)
      8. `_is_count_field`                -> a modest count (0..500)
      9. `_is_numeric_field`              -> int 0..998 (generic fallback)
     10. otherwise                        -> a value from the mixed entity pool
    Duplicate placeholders in one template resolve to the SAME value (one
    sub-seed per DISTINCT field name), keeping rendered docs internally
    consistent.

    After all fields are drawn, a paired-field pass enforces `in_use{N} <=
    max{N}` (same trailing index) wherever both co-occur in the template, so a
    pool line never reads an implausible `in_use 900/500`.

    `pinned` (optional) maps an entity category name (or `entity`) to a fixed
    value; when a field resolves to that category the pinned value overrides the
    per-field draw. Only entity/typed-entity fields honor pinning; numeric/time
    fields are unaffected. This is how one round pins a shared primary subject.

    `path` is the seed sub-path prefix under which per-field sub-seeds are
    derived (appending a stable per-field index), so different fields draw from
    independent streams and different renders never collide.
    """
    pinned = pinned or {}

    # Distinct field names in first-appearance order -> one stable sub-seed each.
    seen: Dict[str, int] = {}
    for _, field_name, _, _ in string.Formatter().parse(tpl):
        if field_name and field_name not in seen:
            seen[field_name] = len(seen)

    values: Dict[str, str] = {}
    for field, idx in seen.items():
        field_rng = child_rng(seed, *path, idx)
        if field == "entity":
            # Pinned mixed-entity override, else a mixed-pool draw.
            values[field] = pinned.get("entity") or _seeded_entity_value(theme, field_rng)
        elif field in theme.entities:
            # Typed entity placeholder -- checked BEFORE the numeric/time
            # heuristics so a category name like `plan`/`db_instance` (which
            # contains an 'n') is not misclassified as numeric. A pinned value
            # for this category (from the caller) overrides the per-field draw.
            values[field] = pinned.get(field) or _seeded_typed_entity_value(theme, field, field_rng)
        elif _is_time_field(field):
            values[field] = _seeded_time_value(field_rng)
        elif _is_status_code_field(field):
            values[field] = _seeded_status_code_value(field_rng)
        elif _is_error_pct_field(field):
            values[field] = _seeded_error_pct_value(field_rng)
        elif _is_percent_field(field):
            values[field] = _seeded_percent_value(field_rng)
        elif _is_latency_ms_field(field):
            values[field] = _seeded_latency_ms_value(field_rng)
        elif _is_count_field(field):
            values[field] = _seeded_count_value(field_rng)
        elif _is_numeric_field(field):
            values[field] = str(int(field_rng.integers(0, 999)))
        else:
            # Unknown field: seeded token, never left unfilled.
            values[field] = _seeded_entity_value(theme, field_rng)

    # Paired-field pass: enforce numeric invariants over the drawn values so a
    # rendered result never reads an impossible ordering. Deterministic — each
    # helper is a pure function of the already-seeded values (no new rng draws):
    #   - `in_use{N} <= max{N}`         (connection-pool saturation)
    #   - `heap_used{N} <= heap_max{N}` (heap can't exceed its ceiling)
    #   - `p50 <= p90 <= p95 <= p99`    (percentile ordering, per shared suffix,
    #     across bare / `_ms` / indexed forms)
    _clamp_le_by_suffix(values, "in_use", "max")
    _clamp_le_by_suffix(values, "heap_used", "heap_max")
    _sort_percentiles_by_suffix(values)

    try:
        return tpl.format_map(values)
    except (KeyError, IndexError):
        return tpl


def _render_tool_result(
    theme: Theme, call_name: str, seed: int, path: Tuple[int, ...], pinned: Optional[Dict[str, str]] = None
) -> str:
    """Render a tool-result content string from the theme's PER-TOOL template
    for `call_name` (falling back to 'default' only if the tool has none),
    filling EVERY placeholder the chosen template declares with a real,
    deterministically-seeded value -- never a literal stand-in, never left
    unfilled.

    `path` is the seed sub-path prefix (e.g. (*agent_seed_path, t, 9, j)) under
    which per-field sub-seeds are derived, so different calls/turns/agents
    never collide.

    `pinned` (optional) forwards the ARGUMENTS the call was made with, so a
    result placeholder that matches an argument key (e.g. `{service}` when the
    call passed `service=...`) ECHOES the requested value instead of drawing an
    independent one -- a real tool answers about the entity it was called with,
    so a `get_service_health(service="x")` result must say `service=x`, not a
    different service.
    """
    tpl = theme.result_templates.get(call_name, theme.result_templates.get("default", "result: {entity} {n0} {t0}"))
    out = _render_theme_template(theme, tpl, seed, path, pinned=pinned)
    return out if out else "result"


def _render_tool_arguments(
    params_schema: Dict[str, Any],
    theme: Theme,
    seed: int,
    path: Tuple[int, ...],
    pinned: Optional[Dict[str, str]] = None,
    word_pool: Optional[List[str]] = None,
    payload_pool: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build an arguments object that CONFORMS to a tool's `parameters` schema.

    Fills every `required` property with a real, deterministically-seeded value by
    type + name:
      - a property NAMED like an entity category (or a pinned key) -> that entity
        (pinned value if the round pinned it, else the typed pool) so args thread
        the same subject as the objective/doc;
      - enum -> a seeded choice from the enum;
      - integer/number -> a seeded int;
      - boolean -> a seeded bool;
      - a large-PAYLOAD arg (content/code/patch/...) -> a chunk of the theme's filler
        pool sized by the arg's `x-payload-tokens` schema hint (words), else a default;
      - string (default) -> a seeded entity token.
    Only `required` props are emitted, so the JSON is always schema-valid and
    non-empty (a non-empty forced tool call is what stops the model cleanly).
    Deterministic: per-property sub-seeds off `path`; no rng outside child_rng.
    """
    pinned = pinned or {}
    props: Dict[str, Any] = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
    required: List[str] = list(params_schema.get("required", [])) if isinstance(params_schema, dict) else []
    args: Dict[str, Any] = {}
    for idx, prop in enumerate(required):
        spec = props.get(prop, {}) if isinstance(props, dict) else {}
        ptype = spec.get("type", "string") if isinstance(spec, dict) else "string"
        prop_rng = child_rng(seed, *path, idx)
        enum = spec.get("enum") if isinstance(spec, dict) else None
        if enum:
            args[prop] = enum[int(prop_rng.integers(0, len(enum)))]
        elif prop in pinned:
            args[prop] = pinned[prop]
        elif prop in theme.entities:
            args[prop] = _seeded_typed_entity_value(theme, prop, prop_rng)
        elif ptype in ("integer", "number"):
            args[prop] = int(prop_rng.integers(0, 999))
        elif ptype == "boolean":
            args[prop] = bool(int(prop_rng.integers(0, 2)))
        elif _strip_index(prop) in _PAYLOAD_ARG_NAMES and (payload_pool or word_pool):
            # A big-payload prop (content/patch/new_string/...): a write/edit/execute
            # tool's whole point is the payload it carries. Draw a chunk from the theme's
            # PAYLOAD pool (domain payload shape: code / SQL / a drafted answer) so the
            # call has realistic content + token weight instead of a tiny `content-417`
            # stub; falls back to the turn-filler pool if the theme has no payload
            # templates. SIZE is a per-arg property the theme declares on the schema
            # (`x-payload-tokens`, in words); absent -> _DEFAULT_PAYLOAD_WORDS. Seeded
            # start offset so multiple payload args in one call differ.
            pool: List[str] = payload_pool or word_pool or []
            n_words = (
                spec.get(_PAYLOAD_TOKENS_KEY, _DEFAULT_PAYLOAD_WORDS) if isinstance(spec, dict) else _DEFAULT_PAYLOAD_WORDS
            )
            n_words = max(1, int(n_words))
            start = int(prop_rng.integers(0, max(1, len(pool))))
            args[prop] = " ".join(_cycled_words(pool, n_words, start=start))
        else:
            # Plain string prop that is NOT an entity category and has no enum
            # (e.g. `bufferpool`, `endpoint`, `job_id`). Draw a token DERIVED FROM
            # THE PROP NAME, not a random cross-category entity -- else a
            # `bufferpool` arg renders as a table/symptom name from the mixed pool.
            args[prop] = f"{prop}-{int(prop_rng.integers(0, 999))}"
    return args


def _render_intro_doc(theme: Theme, seed: int, path: Tuple[int, ...], pinned: Optional[Dict[str, str]] = None) -> str:
    """Render ONE long, realistic intro document from the theme's templates.

    Picks a template deterministically (seeded off `path`), fills its
    placeholders via the shared renderer, and returns the rendered doc. Returns
    "" when the theme carries no `intro_doc_templates`, so the objective is
    emitted alone (unchanged behavior for themes without an intro doc).

    `pinned` (optional) forwards fixed primary-subject entity values to the
    renderer so the doc's `{service}`/`{db_instance}`/`{symptom}` match the
    round's objective (coherence: the doc + the task name the SAME entity).
    """
    templates = theme.intro_doc_templates or []
    if not templates:
        return ""
    pick_rng = child_rng(seed, *path)
    tpl = templates[int(pick_rng.integers(0, len(templates)))]
    return _render_theme_template(theme, tpl, seed, (*path, 1), pinned=pinned)


def build_graph_for_session(
    cfg: SyntheticAgenticConfig, theme: Theme, tokenizer: CustomTokenizer, session_index: int
) -> ReplayGraph:
    """Build a replay graph for one synthetic session.

    Emits N rounds (from `turns_per_session`); each round is an accumulating
    chain of `k+1` calls (from `tool_loop_depth`, fallback fixed 2):
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

    # Theme-relevant filler word pool, built ONCE per session and reused by every
    # fit_filler call so padding reads like more of the theme's own pasted
    # content (logs/metrics/frames) rather than Shakespeare. None -> fit_filler
    # falls back to the shared corpus (themes without `filler_templates`).
    # Reserved seed sub-path 60 (fresh; does not collide with 0/1/2 or any
    # per-agent path). Per-snippet sub-seeds append an index under (60,).
    filler_pool = theme_filler_words(theme, seed, (60,))
    # Distinct pool for large tool-call payload args (code/SQL/drafted-answer shape),
    # from the theme's payload_templates (falls back to filler_pool). Fresh sub-path 68.
    payload_pool = theme_payload_words(theme, seed, (68,))

    n_rounds = sample_int(cfg.turns_per_session, child_rng(seed, 0), cfg.turns_per_session)
    tool_defs_n = sample_int(cfg.tool_catalog_size_per_agent, child_rng(seed, 1), _FB_TOOL_DEFS)
    # tool_catalog_size_per_agent=0 is the bare non-agentic / plain-chat
    # baseline — NO tools advertised at all. Floor at 0 (not 1) so that value
    # flows through to an empty catalog; `_tool_definitions(theme, 0)` returns [].
    tool_defs = _tool_definitions(theme, max(0, tool_defs_n))
    # Fan-out catalog: the theme tools PLUS the structural dispatch_agent tool.
    # Used ONLY on events that force or emit dispatch_agent (the spawn event and the
    # notification events, whose message history carries the dispatch calls), so
    # ordinary/non-fan-out events keep a clean catalog (dispatch_agent is never
    # advertised when there is no fan-out). Guard against duplication in case a
    # theme ever names a tool "dispatch_agent".
    if any(td.get("name") == DISPATCH_AGENT_NAME for td in tool_defs):
        fanout_tool_defs = tool_defs
    else:
        fanout_tool_defs = [*tool_defs, DISPATCH_AGENT_TOOL_DEF]

    # The system head is built PER AGENT (see _system_head below): a real,
    # role-appropriate system prompt (root vs sub-agent) is drawn per agent and
    # fitted to shared_system_prompt_len, so the root and its sub-agents carry
    # DIFFERENT heads -- like a real harness, where an orchestrator and a spawned
    # worker ship different system prompts.

    def _emit(
        event_id: str,
        messages: List[Dict[str, Any]],
        preds: List[str],
        dep_types: Dict[str, str],
        segs: List[InputSegment],
        wait_ms: int,
        is_tool_call: bool,
        tool_names: Optional[List[str]],
        defs: Optional[List[Dict[str, Any]]] = None,
        expected_output: str = "",
        expected_output_tokens: int = 0,
    ) -> None:
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

    def _system_head(is_root: bool, agent_seed_path: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        # Build this agent's system head: a real, role-appropriate system prompt
        # (root orchestrator vs spawned worker) fitted to shared_system_prompt_len.
        # None when the head length is 0 (head-less baseline). The pick is seeded
        # on the agent's own path (+ reserved sub-index 2, the former head path) so
        # it is deterministic per agent and byte-identical per (config, seed), while
        # different agents/roles get different heads. Each call returns a fresh dict
        # (every event owns a DISTINCT message dict; the head rides each agent's
        # FIRST call, and events must never alias a shared dict).
        if cfg.shared_system_prompt_len <= 0:
            return None
        content = _render_system_head(
            tokenizer,
            cfg.shared_system_prompt_len,
            is_root=is_root,
            rng=child_rng(seed, *agent_seed_path, 2),
            word_pool=filler_pool,
        )
        return {"role": "system", "content": content}

    # Per-round bookkeeping for context growth: _build_agent (when is_root) publishes
    # the current round's TERMINAL event id + its full input length here, so the next
    # round can build shared/output segments that re-inject the whole prior turn (its
    # tool loop and its answer) as growing context.
    root_terminal_meta: Dict[str, Any] = {}

    def _answer_text(agent_seed_path: Tuple[Any, ...]) -> Tuple[str, int]:
        """Render the agent's terminal answer text + its sampled token size.

        Draws the size from sub-seed (…, 4) and the filler from (…, 5), off the
        agent's seed path, so the answer is deterministic per agent.
        """
        out_tokens = sample_int(cfg.output_tokens_per_turn, child_rng(seed, *agent_seed_path, 4), cfg.output_tokens_per_turn)
        ans = fit_filler(tokenizer, out_tokens, "Summary:", rng=child_rng(seed, *agent_seed_path, 5), word_pool=filler_pool)
        return ans, out_tokens

    def _ack_text(agent_seed_path: Tuple[Any, ...], c: int) -> Tuple[str, int]:
        """Render a non-terminal async-notification ack + its sampled token size.

        A notification that still leaves sub-agent reports outstanding cannot be
        answered yet, so the orchestrator's turn is a brief acknowledgment (the
        behavior the dispatch tool's own description asks for). Sized from the small
        fixed `_FB_ACK_TOKENS` rather than `output_tokens_per_turn` -- this is a
        throwaway one-liner, not a full answer.

        Draws the size from sub-seed (…, c, 10) and the filler from (…, c, 11), off
        the agent's seed path, so each ack is seeded per-notification and
        deterministic.
        """
        out_tokens = sample_int(_FB_ACK_TOKENS, child_rng(seed, *agent_seed_path, c, 10), _FB_ACK_TOKENS)
        ack = fit_filler(
            tokenizer,
            out_tokens,
            "Acknowledged; awaiting the remaining sub-agent reports.",
            rng=child_rng(seed, *agent_seed_path, c, 11),
            word_pool=filler_pool,
        )
        return ack, out_tokens

    def _build_agent(
        depth: int,
        agent_prefix: str,
        task_msgs: List[Dict[str, Any]],
        preds: List[str],
        dep_types: Dict[str, str],
        principal_wait: int,
        is_root: bool,
        agent_seed_path: Tuple[Any, ...],
        principal_segments: Optional[List[InputSegment]] = None,
        pinned: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Build ONE agent's execution and return its TERMINAL event id.

        Each event is exactly one LLM call whose INPUT is the cumulative conversation
        transcript ending in a user or tool message; the assistant reply that call
        produces is the event's OUTPUT (expected_output), not a separate lone-assistant
        event. The agent is a linear accumulating chain:

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
        if len(events) + _MIN_AGENT_COST > budget:
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
                tokenizer, in_tokens, last.get("content", ""), rng=child_rng(seed, *agent_seed_path, 51), word_pool=filler_pool
            )
            sized_task_msgs[-1] = last
        if principal_segments is not None:
            # Context-growth path: `task_msgs` is the FULL growing transcript
            # (already includes the system head as its first message, so the shared
            # segment — which sources the prior round's principal INPUT — covers it).
            # We must NOT prepend the head again here or it would double and break
            # the segment cursor math (sum(message_count) == len(principal_msgs)).
            principal_msgs = sized_task_msgs
            principal_segs: List[InputSegment] = principal_segments
        else:
            head = _system_head(is_root, agent_seed_path)
            principal_msgs = ([head] if head else []) + sized_task_msgs
            principal_segs = []

        # --- decide k tool-turns up front (governs whether principal is terminal) ---
        k = sample_int(cfg.tool_loop_depth, child_rng(seed, *agent_seed_path, 100), _FB_STEPS)
        k = max(0, k)
        # Bare baseline: with an empty tool catalog (tool_catalog_size_per_agent=0)
        # a tool-loop turn cannot emit a valid forced call — the `name` lookup
        # `tool_defs[j % len(tool_defs)]` would divide by / index an empty list,
        # cannot satisfy the rule that a forced call name must be an advertised tool.
        # A catalog-less agent therefore emits ZERO tool-loop steps and just answers.
        if not tool_defs:
            k = 0

        # Will this agent spawn sub-agents? Decided up front so we know whether the
        # tool loop's last event is the agent terminal (plain answer output) or a
        # hand-off into the fan-out.
        spawn_roll = float(child_rng(seed, *agent_seed_path, 7).random())
        will_spawn = spawn_roll < cfg.fanout_probability and depth < cfg.max_depth

        # Per-turn parallel-call helper: build the K calls + K results for turn t.
        def _turn_calls_and_results(t: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
            n_calls = sample_int(cfg.parallel_tool_calls_per_step, child_rng(seed, *agent_seed_path, t, 30), _FB_PARALLEL)
            n_calls = max(1, n_calls)
            calls: List[Dict[str, Any]] = []
            results: List[Dict[str, Any]] = []
            names: List[str] = []
            for j in range(n_calls):
                # Advance the tool by TURN (t), so a multi-turn loop VARIES its tools
                # (list_dir -> read_file -> grep -> run_tests) instead of repeating
                # tool_defs[0] every turn; `+ j` keeps parallel calls within a turn
                # distinct. Pure index math -> deterministic.
                tool_def = tool_defs[(t + j) % len(tool_defs)]
                call_name = tool_def["name"]
                names.append(call_name)
                tc_id = f"call_{agent_prefix}_{t}_{j}"
                # Arguments CONFORM to the tool's advertised parameter schema
                # (required props filled, entity-threaded to the round's pinned
                # subject where the prop is an entity category). A non-empty,
                # schema-valid forced call is what stops the model cleanly instead
                # of emitting empty `{}` + leaking template tokens (Qwen). Fresh
                # per-call sub-path (t, 31, j) — does not collide with n_calls
                # (t,30) or the result draw (t,9,j).
                params_schema = tool_def.get("function", {}).get("parameters", {})
                arg_obj = _render_tool_arguments(
                    params_schema,
                    theme,
                    seed,
                    (*agent_seed_path, t, 31, j),
                    pinned=pinned,
                    word_pool=filler_pool,
                    payload_pool=payload_pool,
                )
                calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        # tool-call arguments are json.dumps-ed strings.
                        "function": {"name": call_name, "arguments": json.dumps(arg_obj)},
                    }
                )
                # The result ECHOES the call's arguments: a placeholder in the
                # result template matching an argument key (e.g. `{service}`)
                # resolves to what THIS call passed, so the result answers about
                # the entity it was called with (not an independent draw). The
                # round's pinned subject fills any entity placeholder the args
                # don't cover; call args take precedence (str-coerced for text
                # substitution).
                result_pinned = {**(pinned or {}), **{k: str(v) for k, v in arg_obj.items()}}
                result = _render_tool_result(theme, call_name, seed, (*agent_seed_path, t, 9, j), pinned=result_pinned)
                results.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            return calls, results, names

        # The principal's OUTPUT: the first tool call (turn 0) if the loop runs,
        # else the plain answer. If it will spawn but has no tool-loop steps, the
        # principal still outputs plain text and the merge follows.
        principal_is_terminal = (k == 0) and not will_spawn
        first_calls: List[Dict[str, Any]] = []  # populated only when k >= 1 (silences strict unbound check)
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
                expected_output_tokens=_tool_call_max_tokens(tokenizer, first_calls),
            )
        else:
            ans_text, ans_tokens = _answer_text(agent_seed_path) if principal_is_terminal else ("", 0)
            # k=0 terminal principal (the agent answers with NO tool loop): append a
            # trailing "answer in plain prose, no tool call" nudge as the LAST message
            # so its one answer is PROSE, not tool-call text. A SUB-AGENT gets the report
            # directive; the ROOT gets the user-answer directive.
            #   - No input_segments (a fresh sub-agent task, or a round-0 root prompt):
            #     the runtime replays original_messages verbatim -> just append, no
            #     segment bookkeeping.
            #   - WITH input_segments (a round r>=1 root follow-up: shared+output+unique):
            #     append AND bump the trailing `unique` segment's message_count by 1 so
            #     sum(message_count) == len(principal_msgs) stays exact. The next round's
            #     shared prefix covers the nudge automatically, since it sources this
            #     terminal event's full (nudge-included) input.
            # A no-tools agent cannot emit tool-call text, so the ROOT answer nudge is
            # pointless there -> gate the root path on a non-empty catalog (this keeps the
            # bare/no-tools baseline a clean single [user] turn). A sub-agent always gets
            # the report directive.
            want_nudge = principal_is_terminal and (not is_root or bool(tool_defs))
            if want_nudge:
                directive = SUBAGENT_REPORT_DIRECTIVE if not is_root else ROOT_ANSWER_DIRECTIVE
                principal_msgs = [*principal_msgs, {"role": "user", "content": directive}]
                if principal_segs:
                    tail = principal_segs[-1]
                    assert tail.type == "unique", "expected trailing unique segment on a round r>=1 principal"
                    principal_segs = [
                        *principal_segs[:-1],
                        InputSegment(
                            type="unique",
                            message_count=tail.message_count + 1,
                            token_count=tail.token_count,
                            source_event_id=tail.source_event_id,
                        ),
                    ]
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
        #                  placeholder assistant carrying those same ids (so each
        #                  tool_call is matched by exactly one role:tool result).
        #                  Empty when the prior output was plain text.
        prev_id = principal_id
        prev_input_len = len(principal_msgs)
        prev_msgs: List[Dict[str, Any]] = list(principal_msgs)
        prev_out_calls: List[Dict[str, Any]] = first_calls if k >= 1 else []

        # --- k tool-turn events (accumulating chain) ---
        # Event ':tN' (N = 0..k-1) re-injects the prior event's tool-call reply
        # (output segment) plus that call's results (unique segment). Its OWN
        # output is the next tool call (N < k-1) or, if it is the agent terminal
        # (last turn AND no spawn), the plain answer.
        #
        # A SPAWNING agent runs one fewer event here: the spawn event itself plays
        # the role of the last ':tN' (it consumes the final tool results via its own
        # output+unique segments -- see the fan-out block below). Without this, the
        # last turn would be forced to emit a tool call for turn `k` that nothing
        # ever answers, making a spawner run k+1 tool-emitting calls where a leaf
        # with the same tool_loop_depth runs k.
        n_turn_events = k - 1 if will_spawn else k
        for t in range(n_turn_events):
            # Room for this event; if it won't fit, stop the loop early. The
            # prior event remains a valid terminal (its output stays a tool call
            # if we truncate here, which is acceptable — it simply never gets a
            # follow-up; no dangling result is created because the result only
            # materializes in THIS event which we skip).
            if len(events) + 1 > budget:
                break
            _, results, _ = _turn_calls_and_results(t)
            # The output-slot placeholder assistant carries the prior event's
            # emitted calls (same ids as `results`), so exactly these calls are
            # matched by exactly these results.
            output_placeholder = {"role": "assistant", "tool_calls": [dict(c) for c in prev_out_calls]}
            turn_msgs = [*prev_msgs, output_placeholder, *results]
            turn_segs = [
                InputSegment(type="shared", message_count=prev_input_len, token_count=0, source_event_id=prev_id),
                InputSegment(type="output", message_count=1, token_count=0, source_event_id=prev_id),
                InputSegment(type="unique", message_count=len(results), token_count=0, source_event_id=None),
            ]
            turn_id = f"{agent_prefix}:t{t}"
            tool_latency = cfg.tool_call_latency_sec or _FB_TOOL_LATENCY
            turn_wait = int(sample_from_distribution(tool_latency, 1, rng=child_rng(seed, *agent_seed_path, t, 3))[0] * 1000)
            is_last_turn = t == k - 1
            turn_is_terminal = is_last_turn and not will_spawn
            if turn_is_terminal:
                # OUTPUT is the plain answer.
                ans_text, ans_tokens = _answer_text(agent_seed_path)
                # Terminal turn: append a trailing "answer in plain prose, no tool call"
                # nudge as the LAST message, so the final turn produces PROSE instead of
                # tool-call text. It must be at the END (recency) -- a tool-primed model
                # mimics the recent tool-call cadence if the transcript ends in a tool
                # result; a directive buried at turn 0 loses. The nudge is fresh content,
                # so it rides the `unique` segment (message_count += 1) -> cursor math
                # stays exact. A SUB-AGENT gets the report directive (its answer is a
                # report to the orchestrator); the ROOT gets the user-answer directive
                # (its answer goes to the user). The nudge is part of this terminal turn's
                # transcript, so the next round carries it forward like any other message.
                directive = SUBAGENT_REPORT_DIRECTIVE if not is_root else ROOT_ANSWER_DIRECTIVE
                nudge = {"role": "user", "content": directive}
                turn_msgs = [*turn_msgs, nudge]
                # The nudge is one more fresh message, so extend the trailing `unique`
                # segment by 1 to keep sum(message_count) == len(turn_msgs).
                last_seg = turn_segs[-1]
                turn_segs = [
                    *turn_segs[:-1],
                    InputSegment(
                        type="unique",
                        message_count=last_seg.message_count + 1,
                        token_count=last_seg.token_count,
                        source_event_id=last_seg.source_event_id,
                    ),
                ]
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
                    expected_output_tokens=_tool_call_max_tokens(tokenizer, next_calls),
                )
                next_out_calls = next_calls
            prev_id = turn_id
            prev_input_len = len(turn_msgs)
            prev_msgs = turn_msgs
            prev_out_calls = next_out_calls

        # --- optional fan-out: ONE spawn event (parallel dispatch_agent calls) +
        # the spawned children + post-dispatch ack + K sequential notification events (the LAST is this
        # agent's terminal) ---
        #
        # This mirrors how a real harness (e.g. Claude Code) spawns sub-agents ASYNC:
        # the agent's own reasoning turn emits N parallel Agent/dispatch tool_calls in
        # a SINGLE assistant message (carrying the agent's full head + accumulated
        # context). There is NO separate headless "dispatch-only" call per child. Each
        # dispatch's tool RESULT is a content-free launch ack -- the child's actual
        # report arrives LATER, out of band, as its own user-role notification, and the
        # K reports land ONE AT A TIME. So the tail is:
        #   spawn event   : input = parent transcript, OUTPUT = [dispatch_agent x K]
        #   notify{c}     : input = parent transcript + [that assistant] + [K stub tool
        #                   results] + [child c's report]; OUTPUT = a brief ack while
        #                   reports are still outstanding, and for c == K-1 the plain
        #                   answer synthesizing across all children (agent terminal).
        if will_spawn:
            K = sample_int(cfg.sub_agents_per_spawn, child_rng(seed, *agent_seed_path, 8), _FB_SUB_AGENTS)
            K = max(0, K)
            # Whole-spawn minimum cost: one spawn event + one minimal child per K +
            # (K + 1) orchestrator turns -- the immediate post-dispatch ack turn plus
            # one notification per child report (the LAST is this agent's terminal).
            # Only spawn if it all fits; else the agent stays a plain leaf whose current
            # `prev` event becomes terminal.
            min_spawn_cost = 1 + K * _MIN_AGENT_COST + (K + 1)
            if K > 0 and len(events) + min_spawn_cost <= budget:
                # Child objectives are pinned to the PARENT's subject entity, so the
                # whole fan-out is ONE coherent investigation (the orchestrator's
                # incident) rather than each child probing an unrelated service. The
                # VERB is still drawn freely, so children take different ANGLES on the
                # same subject ("Analyze" vs "Triage" the same service) -- exactly the
                # real parallel-sub-agent pattern. One dispatch_agent call per child;
                # its args CONFORM to DISPATCH_AGENT_TOOL_DEF (objective required) so
                # the forced call is non-empty + valid, and the call ids match the K
                # tool results the merge reconstructs below.
                spawn_id = f"{agent_prefix}:d{depth}:spawn"
                dispatch_calls: List[Dict[str, Any]] = []
                child_objs: List[str] = []
                for c in range(K):
                    child_obj = _render_objective(theme, child_rng(seed, *agent_seed_path, c, 1), pinned=pinned)
                    child_objs.append(child_obj)
                    dispatch_calls.append(
                        {
                            "id": f"dispatch_{agent_prefix}_{c}",
                            "type": "function",
                            "function": {"name": DISPATCH_AGENT_NAME, "arguments": json.dumps({"objective": child_obj})},
                        }
                    )
                # The spawn event rides the parent chain: its input CONTINUES from the
                # parent's last input (shared-only prepend, exactly like the old merge's
                # prefix -- introduces NO unmatched prior tool_call, so nothing dangles)
                # plus a fresh `unique` user step asking to delegate. Its OUTPUT is the
                # K parallel dispatch_agent calls (matched by the K stub tool results).
                #
                # This message asks ONLY for the delegation, and deliberately does NOT
                # say "synthesize their findings": synthesis is requested
                # once, at the point it is actually due: the LAST notification appends
                # ROOT_ANSWER_DIRECTIVE / SUBAGENT_REPORT_DIRECTIVE.
                spawn_ctx = {
                    "role": "user",
                    "content": (
                        f"Delegate this work to {K} sub-agent(s). Their reports will arrive one at a "
                        f"time as each sub-agent completes."
                    ),
                }
                # When the tool loop ran, this event is ALSO the loop's last link: it
                # consumes the final outstanding tool call's results (output segment =
                # the prior event's tool-call reply, unique segment = those results),
                # exactly like a ':tN' event, and only then asks for the delegation.
                # That is what keeps a spawner's tool-emitting call count equal to
                # tool_loop_depth and leaves no tool call unanswered.
                # `prev_out_calls` is empty when there was no loop (k == 0, an empty
                # catalog, or a budget-truncated loop) -- then the input continues from
                # the parent's last input with a shared-only prepend, which introduces
                # no unmatched prior tool_call, so nothing dangles either way.
                if prev_out_calls:
                    _, spawn_results, _ = _turn_calls_and_results(n_turn_events)
                    spawn_output_placeholder = {"role": "assistant", "tool_calls": [dict(c) for c in prev_out_calls]}
                    spawn_msgs = [*prev_msgs, spawn_output_placeholder, *spawn_results, spawn_ctx]
                    spawn_segs = [
                        InputSegment(type="shared", message_count=prev_input_len, token_count=0, source_event_id=prev_id),
                        InputSegment(type="output", message_count=1, token_count=0, source_event_id=prev_id),
                        InputSegment(type="unique", message_count=len(spawn_results) + 1, token_count=0, source_event_id=None),
                    ]
                else:
                    spawn_msgs = [*prev_msgs, spawn_ctx]
                    spawn_segs = [
                        InputSegment(type="shared", message_count=prev_input_len, token_count=0, source_event_id=prev_id),
                        InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
                    ]
                # Emit the spawn event, then build the K children from it. The spawn's
                # K dispatch calls are matched at replay by the merge's K tool results,
                # and the runtime rewrites those results' ids to the model's LIVE calls
                # IN ORDER (output-segment id-rewrite), so the merge MUST supply exactly
                # K results -- fewer would leave live calls unmatched (dangling). We
                # therefore spawn ATOMICALLY: build all K children, and if any child
                # fails to fit, roll the whole spawn back (drop the spawn event + any
                # children already built) and fall back to the pre-spawn terminal.
                # min_spawn_cost already reserved 1 (spawn) + K minimal children + K
                # (notifications) + dispatch ack; a child can still exceed its minimum (own loop /
                # nested spawn), so the per-child re-check catches an over-budget child.
                spawn_input_len = len(spawn_msgs)
                events_before_spawn = list(events.keys())
                _emit(
                    spawn_id,
                    spawn_msgs,
                    [prev_id],
                    {prev_id: "full_match"},
                    spawn_segs,
                    0,
                    True,
                    [DISPATCH_AGENT_NAME] * K,
                    defs=fanout_tool_defs,
                    expected_output_tokens=_tool_call_max_tokens(tokenizer, dispatch_calls),
                )
                child_terminals: List[str] = []
                for c in range(K):
                    # Room for this child's minimum PLUS the still-pending orchestrator
                    # turns. The ack turn and ALL K notifications are still unbuilt at
                    # this point (they are emitted only after every child succeeds), so
                    # the reservation is a flat K+1 here, not K-c.
                    if len(events) + _MIN_AGENT_COST + (K + 1) > budget:
                        break
                    child_prefix = f"{agent_prefix}:d{depth + 1}:sub{c}"
                    child_task = [{"role": "user", "content": child_objs[c]}]
                    child_terminal = _build_agent(
                        depth + 1,
                        child_prefix,
                        child_task,
                        [spawn_id],
                        {spawn_id: "full_match"},
                        0,
                        False,
                        (*agent_seed_path, 200 + c),
                        pinned=pinned,  # child's own turns stay on the parent's subject
                    )
                    if child_terminal is None:
                        break
                    child_terminals.append(child_terminal)

                if len(child_terminals) == K and len(events) + (K + 1) <= budget:
                    # --- ASYNC fan-out tail. The orchestrator's flow is:
                    #
                    #   spawn          OUTPUT = dispatch_agent x K
                    #   dispatch_ack   input ends in the K stub tool results, no report
                    #                  yet -> "the agents are running" (fires IMMEDIATELY:
                    #                  gated on the spawn alone, no child dependency)
                    #   notify0        + child 0's report as a user notification -> ack
                    #   ...
                    #   notify{K-1}    + the last report -> the synthesis (TERMINAL)
                    #
                    # so a spawn costs K+1 orchestrator turns. With K=1 the flow
                    # degenerates gracefully: dispatch_ack then a single terminal notify0.
                    #
                    # Base transcript every notification builds on: the SPAWN event's
                    # input (shared) + the spawn's assistant reply (the K dispatch
                    # calls, one `output` message sourcing the spawn) + K static stub
                    # tool results. inv #3 still holds -- the K stub results
                    # carry the K dispatch call ids (rewritten to the live spawn calls
                    # in order at replay). ---
                    base_msgs: List[Dict[str, Any]] = list(spawn_msgs)
                    base_segs: List[InputSegment] = [
                        InputSegment(type="shared", message_count=spawn_input_len, token_count=0, source_event_id=spawn_id),
                    ]
                    base_msgs.append({"role": "assistant", "tool_calls": [dict(c) for c in dispatch_calls]})
                    base_segs.append(InputSegment(type="output", message_count=1, token_count=0, source_event_id=spawn_id))
                    for c in range(K):
                        base_msgs.append(
                            {"role": "tool", "tool_call_id": dispatch_calls[c]["id"], "content": ASYNC_DISPATCH_STUB}
                        )
                    base_segs.append(InputSegment(type="unique", message_count=K, token_count=0, source_event_id=None))

                    # (1) The IMMEDIATE post-dispatch turn. Gated on the SPAWN ALONE --
                    # deliberately NO child dependency -- so it fires as soon as the
                    # dispatch calls come back.
                    # It carries the full `base_segs` layout. Collapsing it into one
                    # `shared` would (a) claim len(base_msgs) messages from the spawn
                    # event, which only has spawn_input_len, tripping the runtime's
                    # length-mismatch fallback, and (b) drop the `output` segment, so the
                    # spawn's LIVE assistant tool-call reply would never be substituted
                    # in -- leaving the stub results' tool_call_ids dangling.
                    ack_id = f"{agent_prefix}:d{depth}:dispatch_ack"
                    dispatch_ack_text, dispatch_ack_tokens = _ack_text(agent_seed_path, K)
                    _emit(
                        ack_id,
                        base_msgs,
                        [spawn_id],
                        {spawn_id: "full_match"},
                        list(base_segs),
                        0,
                        False,
                        None,
                        defs=fanout_tool_defs,
                        expected_output=dispatch_ack_text,
                        expected_output_tokens=dispatch_ack_tokens,
                    )

                    # (2) K notification events, one per child report, chained after the
                    # ack turn. Each one's predecessors are the PRIOR link in the chain
                    # and its OWN child's terminal, so it cannot fire until that child has
                    # actually finished -- the child terminal's live-measured call duration
                    # (real TTFT + decode) IS the timing signal for "how long this child
                    # took". wait_ms is therefore 0: a second, independently-sampled delay
                    # on top would double-count that latency with no principled way to
                    # decompose the two.
                    notify_prev_id = ack_id
                    notify_prev_msgs = base_msgs
                    notify_prev_len = len(base_msgs)
                    notify_ids: List[str] = []
                    for c, child_term in enumerate(child_terminals):
                        is_last = c == K - 1
                        notif_id = f"{agent_prefix}:d{depth}:notify{c}"
                        # The notification itself: a user message whose CONTENT is
                        # replaced at replay by this child's live report text
                        # (`async_report`, sourcing the child terminal). The prior link
                        # (the ack turn, or the previous notification) supplies the whole
                        # prefix and really does have notify_prev_len messages, so one
                        # `shared` sourcing it is exact.
                        notif_msgs: List[Dict[str, Any]] = [
                            *notify_prev_msgs,
                            {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT"},
                        ]
                        notif_segs: List[InputSegment] = [
                            InputSegment(
                                type="shared",
                                message_count=notify_prev_len,
                                token_count=0,
                                source_event_id=notify_prev_id,
                            ),
                            InputSegment(type="async_report", message_count=1, token_count=0, source_event_id=child_term),
                        ]
                        if is_last:
                            # Only the LAST notification is the agent TERMINAL: every
                            # report is now in, so this is the turn that synthesizes
                            # across all K children. Like any terminal ending in
                            # non-assistant content it gets a trailing answer nudge so
                            # its output is PROSE, not tool-call text. A NON-ROOT
                            # terminal reports to its parent (report directive); the
                            # ROOT's is the final answer to the USER (answer directive).
                            # The nudge is fresh content -> a trailing `unique` segment
                            # keeps cursor math exact.
                            directive = ROOT_ANSWER_DIRECTIVE if is_root else SUBAGENT_REPORT_DIRECTIVE
                            notif_msgs = [*notif_msgs, {"role": "user", "content": directive}]
                            notif_segs = [
                                *notif_segs,
                                InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
                            ]
                            expected_output, expected_output_tokens = _answer_text(agent_seed_path)
                        else:
                            # Reports still outstanding -> a brief ack and keep waiting.
                            # No directive MESSAGE is injected: that behavior is part of
                            # the dispatch tool's own contract
                            # (DISPATCH_AGENT_DESCRIPTION), the way a real harness
                            # documents it. It could not live in the notification message
                            # anyway -- that content is replaced at replay by the child's
                            # live report.
                            expected_output, expected_output_tokens = _ack_text(agent_seed_path, c)
                        _emit(
                            notif_id,
                            notif_msgs,
                            [notify_prev_id, child_term],
                            {notify_prev_id: "full_match", child_term: "full_match"},
                            notif_segs,
                            0,
                            False,
                            None,
                            defs=fanout_tool_defs,
                            expected_output=expected_output,
                            expected_output_tokens=expected_output_tokens,
                        )
                        notify_ids.append(notif_id)
                        notify_prev_id = notif_id
                        notify_prev_msgs = notif_msgs
                        notify_prev_len = len(notif_msgs)
                    # The LAST notification is this agent's terminal.
                    prev_id = notify_ids[-1]
                else:
                    # Atomic rollback: not all K children fit (or the K notification
                    # events would not), so the spawn can't be completed danglelessly.
                    # Drop the spawn event AND any children already built, restoring the
                    # graph to exactly its pre-spawn state. Deleting by "id not in the
                    # pre-spawn snapshot" covers however many events were added, so it
                    # needs no adjustment for the notification chain. prev_id stays at
                    # the pre-spawn terminal; the final normalization re-emits that as a
                    # plain answer.
                    for eid in list(events.keys()):
                        if eid not in events_before_spawn:
                            del events[eid]
            # If the spawn was rolled but did not fit / produced no children, the
            # current `prev` event may still advertise a tool call as its output;
            # the final normalization below re-emits it as a plain-answer terminal.

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

        if is_root:
            # Publish the TERMINAL event's id + its full input length so the next
            # round carries the whole turn forward (its tool loop AND answer), like a
            # real multi-turn agent. The next round's `shared` prefix sources this
            # terminal and its message_count MUST equal len(the terminal's input), so
            # the runtime slice get_messages_by_event_id(terminal)[:message_count]
            # matches exactly.
            root_terminal_meta["id"] = prev_id
            root_terminal_meta["input_len"] = len(events[prev_id].call.messages)

        return prev_id

    prev_answer_id: Optional[str] = None
    # The running conversation transcript used to build a round's growing context.
    # After each round it becomes the placeholder prefix the next round's `shared`
    # segment covers; the shared segment re-injects the LIVE version at replay, so
    # the placeholder only needs to be coherent, deterministic, and the right length.
    transcript: List[Dict[str, Any]] = []

    # Context compaction: sample the per-session trigger/target ONCE (so a
    # whole session shares one compaction budget; different sessions differ). Fresh
    # seed sub-paths (64)/(65) — do not collide with the primary pin (62), focus pin
    # (63), or any per-round draw. None trigger => compaction is entirely inert and
    # the graph is byte-identical to a no-compaction build.
    compaction = cfg.context_compaction
    compaction_trigger: Optional[int] = (
        sample_int(compaction.trigger_tokens, child_rng(seed, 64), compaction.trigger_tokens)
        if compaction is not None
        else None
    )
    compaction_target: Optional[int] = (
        sample_int(compaction.target_tokens, child_rng(seed, 65), compaction.target_tokens) if compaction is not None else None
    )

    for r in range(n_rounds):
        # Stop starting new rounds once even the minimum agent (principal + answer)
        # won't fit the event budget; never truncate mid-round. Deeper fan-out is
        # budget-guarded inside _build_agent.
        if len(events) + _MIN_AGENT_COST > budget:
            break

        # Session-scoped subject pin, threaded into every round's objective, intro
        # doc, follow-up, and every tool call/result so the whole session references
        # one coherent subject + focus (a live model flags a doc about one service
        # paired with a task about another, or a conversation that drifts subjects).
        # Two pins, merged with the primary winning on shared keys:
        #   - focus pin (every entity category, sub-path 63): the ONE file/symbol/
        #     test/dep the tool loop investigates.
        #   - primary pin (service/db_instance/symptom/region, sub-path 62): the
        #     round's subject, kept fixed across all rounds.
        # Both omit `r` so every round shares them.
        pinned = {
            **_pinned_focus_entities(theme, child_rng(seed, 63)),
            **_pinned_primary_entities(theme, child_rng(seed, 62)),
        }
        obj = _render_objective(theme, child_rng(seed, r, 1), pinned=pinned)
        # Round 0 opens the session with a long, realistic "someone pasted this"
        # intro document (an incident ticket / metrics dump / config excerpt),
        # PREPENDED to the objective so it rides the first user turn's content
        # and stays intact as fit_filler's fixed_content (preserved after the
        # </context> block). Later rounds are terse follow-ups (no re-paste).
        # Reserved seed sub-path (r, 61) — fresh, does not collide with the
        # round's other draws (objective (r,1), think-time (r,2), followup (r,3),
        # pinned-entity (r,62)). The pinned dict makes the doc's primary subject
        # match the objective's.
        if r == 0:
            intro = _render_intro_doc(theme, seed, (r, 61), pinned=pinned)
            if intro:
                obj = f"{intro}\n{obj}"
        # Round 0's principal has no preceding wait; rounds 2..N wait a human
        # read/think/reply gap (user_think_time_sec) before the follow-up turn.
        if r == 0:
            principal_wait = 0
        else:
            think_dist = cfg.user_think_time_sec or _FB_USER_THINK
            # Sample as a float and scale to ms BEFORE truncating to int, so a
            # fractional-second mean (e.g. 0.5s) doesn't collapse to 0/1s.
            principal_wait = int(sample_from_distribution(think_dist, 1, rng=child_rng(seed, r, 2))[0] * 1000)

        # Context compaction decision (PRE-turn): if extending the grown transcript
        # into THIS round would cross the trigger, compact instead — start fresh with
        # a summary block replacing the history. Measured against message content +
        # tool catalog (the true prefill we control; see _accumulated_wire_tokens).
        # We check before building so the over-threshold turn is never emitted.
        should_compact = (
            compaction_trigger is not None
            and prev_answer_id is not None
            and r > 0
            and _accumulated_wire_tokens(tokenizer, transcript, tool_defs) >= compaction_trigger
        )

        if r == 0 or prev_answer_id is None or should_compact:
            # Fresh single-turn prompt. Round 0, defensive fallback, OR a compaction
            # round. The system head is prepended inside _build_agent; no input_segments
            # -> the runtime builds this round's input purely from its own `unique`
            # messages, so the grown transcript is genuinely dropped (compaction) rather
            # than re-injected via shared/output.
            if should_compact:
                # Prepend a seeded summary block (sized to the per-session target)
                # standing in for the compacted history, so the fresh prompt reads as
                # "here's a summary of what we did, now continue". Same fit_filler
                # machinery as answers/objectives -> the summary is domain-shaped and
                # deterministic. Fresh sub-paths (r,66)=filler, (r,68)=recap — do not
                # collide with objective (r,1)/think (r,2)/followup (r,3)/intro (r,61).
                # The prepend mirrors the r==0 intro-doc prepend above. `compaction_target`
                # is guaranteed set here: should_compact implies the trigger is set, and
                # the config validator requires the target whenever the trigger is set.
                assert compaction_target is not None  # validator guarantees both-or-neither
                # A semantic recap of the session (real subject + real tools) when the
                # theme provides a template; else the bare marker. This is the fixed
                # content fit_filler keeps intact while padding to the target size.
                recap = _render_compaction_summary(theme, tool_defs, child_rng(seed, r, 68), pinned=pinned)
                summary_fixed = f"Summary of prior context: {recap}" if recap else "Summary of prior context:"
                summary = fit_filler(
                    tokenizer,
                    compaction_target,
                    summary_fixed,
                    rng=child_rng(seed, r, 66),
                    word_pool=filler_pool,
                )
                obj = f"{summary}\n{obj}"
            task_msgs: List[Dict[str, Any]] = [{"role": "user", "content": obj}]
            principal_segments: Optional[List[InputSegment]] = None
            # Keep an ORDERING edge to the prior answer (the session stays one
            # connected chain; this round runs after it) but NOT as a substitution
            # source — principal_segments is None, so nothing slices into the prior
            # round. On a compaction round this is what makes the prefill drop.
            preds = [prev_answer_id] if prev_answer_id else []
            dep_types = {prev_answer_id: "full_match"} if prev_answer_id else {}
        else:
            # Round K>=1 (growing context). Carry the WHOLE prior turn forward, like a
            # real multi-turn agent: the prior round's terminal INPUT (its full tool
            # loop) plus the terminal's OUTPUT (the answer), then the new follow-up.
            # Layout of the principal's original_messages and matching segments
            # (cursor-aligned 1:1):
            #   [ transcript... , answer_placeholder , followup ]
            #   [ shared(count=len(transcript), src=prev terminal) ]  -> prior turn's full loop
            #   [ output(1, src=prev terminal)                     ]  -> prior turn's answer
            #   [ unique(1)                                        ]  -> new follow-up
            # sum(message_count) == len(original_messages), so the runtime cursor
            # math in _build_messages_with_substitution is exact (no IndexError).
            prev_terminal_id = root_terminal_meta["id"]
            prev_terminal_len = root_terminal_meta["input_len"]
            followup = _render_followup(theme, obj, child_rng(seed, r, 3), pinned=pinned)
            # The `shared` prefix must be exactly prev_terminal_len messages; the
            # accumulated `transcript` is kept at that length as the placeholder.
            prefix_msgs = list(transcript)
            answer_placeholder = {"role": "assistant", "content": "PLACEHOLDER_PRIOR_ANSWER"}
            followup_msg = {"role": "user", "content": followup}
            task_msgs = [*prefix_msgs, answer_placeholder, followup_msg]
            principal_segments = [
                InputSegment(type="shared", message_count=prev_terminal_len, token_count=0, source_event_id=prev_terminal_id),
                InputSegment(type="output", message_count=1, token_count=0, source_event_id=prev_terminal_id),
                InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
            ]
            # The terminal source must also be a predecessor so substitution runs after
            # require_async has awaited it (full_match is DOT-only).
            preds = [prev_terminal_id]
            dep_types = {prev_terminal_id: "full_match"}

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
            pinned=pinned,
        )
        if terminal is None:
            break
        prev_answer_id = terminal
        # The next round's `shared` prefix must equal THIS round's terminal INPUT
        # (its message_count is root_terminal_meta["input_len"]). Take the terminal
        # event's own messages verbatim — they ARE that input (the full tool loop) —
        # as the placeholder transcript, so len(prefix) == published input_len exactly.
        transcript = list(events[root_terminal_meta["id"]].call.messages)

    return ReplayGraph(events=events, root_event_ids=root_ids, source_file="synthetic")


# --- The generator class (lazy build + theme weighting) -------------------
#
# Ties the pure graph builder above to the shared graph-backed session
# runtime. Mirrors OTelTraceReplayDataGenerator: require the replay config,
# pass it to the base as `replay_config=`, and register lazy session slots so
# get_session_count() works immediately while each graph is built on demand.


class SyntheticAgenticDataGenerator(ReplayGraphSessionGeneratorBase):
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
        synthetic_config = config.synthetic_agentic
        if synthetic_config is None:
            raise ValueError("synthetic_agentic configuration is required for SyntheticAgenticDataGenerator")

        self.synthetic_config: SyntheticAgenticConfig = synthetic_config

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
        theme_weights = self.synthetic_config.theme_weights()
        names = list(theme_weights.keys())
        weights = np.array([theme_weights[n] for n in names], dtype=np.float64)
        weights = weights / weights.sum()
        rng = child_rng(session_seed(self.synthetic_config.seed, session_index), 999)
        return self._themes[names[int(rng.choice(len(names), p=weights))]]

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        theme = self._pick_theme(session_index)
        # The graph builder sizes every turn against the tokenizer (input/output token
        # targets, filler fitting), so it cannot run without one. The base class types
        # `tokenizer` as Optional; fail loudly here rather than at the first count_tokens.
        if self.tokenizer is None:
            raise ValueError("synthetic_agentic requires a tokenizer to size its turns")
        graph = build_graph_for_session(self.synthetic_config, theme, self.tokenizer, session_index)
        if not graph.events:
            return None
        sid = f"synthN{session_index}"
        return ReplaySession(session_id=sid, source_id=sid, session_index=session_index, graph=graph)
