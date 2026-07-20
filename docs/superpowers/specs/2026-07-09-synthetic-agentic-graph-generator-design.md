# Synthetic Agent Sessions Data Generator — Design

**Date:** 2026-07-09
**Status:** Design (approved for spec review)
**Branch:** `workload-catalog/agentic-trace-replay`

## 1. Summary

A new data generator, `SyntheticAgentSessionsDataGenerator`, that **procedurally generates**
multi-agent agentic sessions (`ReplaySession(ReplayGraph)` objects — DAGs of LLM calls) for
the inference-perf replay runtime, directly from config knobs. Its sole job is to produce
those graphs from user-controlled knobs instead of reading them from recorded OpenTelemetry
traces.

It subclasses the same `ReplayGraphSessionGeneratorBase` as `OTelTraceReplayDataGenerator`, so
it reuses the base-class replay **runtime** — DAG scheduling, per-worker execution, output
substitution, tool-call handling, completion tracking, memory eviction. For every non-fan-out
shape (autonomous, tool-loop, interactive multi-round) the runtime is reused **unchanged**; the
only additions are config/`main.py` wiring (§10). The **one** fan-out (sub-agent) shape
requires a **single new runtime primitive** — a `tool_output` input-segment type (§4.1a).

The headline workload is an **autonomous agent run**: a single task given up front, then
the agent executes end-to-end — a long tool-loop that may **spawn concurrent sub-agents**,
which themselves run tools or recursively spawn, merging back — with no further external
input (think an agent in auto / skip-permissions mode). Exgentic
`agent-llm-traces` is the **simple special case** of this: a single autonomous agent that
runs tools but never spawns sub-agents. The recursive multi-agent fan-out is the capability
that does **not** exist in the source data and is the primary motivation. Interactive
multi-turn conversations (a human injecting follow-ups) are supported as the same structure
with repeated principal input (§4). It is thus "like `conversation_replay`, but richer" —
`conversation_replay` models only a single linear human conversation.

**Scale target:** up to **10,000 agents**. Each session is built by a **session-local**
seeded pre-order tree walk (seed derived stably from `(config.seed, session_index)`, §2.3a)
with O(1) text production per node, lazily built per session and evicted after completion,
so graph size scales without holding all sessions in memory. `_build_session(N)` is a **pure
function of `(config, N)`** — the lazy runtime builds each graph in both the parent and its
worker, so the two must be byte-identical (§2.3a).

**Fidelity scope (validated against real data).** Verified against the real
`Exgentic/agent-llm-traces-v2` HF dataset (§2.5): its sessions are **flat,
single-agent** linear conversations (0/72 sampled had any sub-agent nesting), so
the **multi-agent parallel topology is a capability the published dataset lacks** —
it is synthetic-only relative to *that source*, **not** a shape that never occurs.
A separately collected real Claude Code trace (§2.6) exhibits exactly this fan-out
(one turn spawning three concurrent sub-agents), confirming the pattern is real and
that our knobs reproduce it. Conversely, the generator must also reproduce the
**simple single-agent shapes** that dominate the real data (plain answer, tool-loop,
multi-turn) — see §2.5 and §4.2. The design must not conflate "faithful to the
replay runtime" with "faithful to the real data"; §2.5/§2.6 record where they differ.

## 2. Background & constraints (from codebase analysis)

### 2.1 The target format (`replay_graph_types.py`)
- `ReplayGraph{events: Dict[event_id, GraphEvent], root_event_ids, source_file}`
- `GraphEvent{event_id, call: GraphCall, predecessor_event_ids,
  predecessor_dependency_types, wait_ms, t_start_ms, t_end_ms}`
- `GraphCall{call_id, model, messages: List[Dict], expected_output,
  input_segments: List[InputSegment], total_input_tokens, expected_output_tokens,
  temperature, max_tokens_recorded, tool_definitions, expected_output_is_tool_call,
  expected_output_tool_names, attributes}`
- `InputSegment{type: "shared"|"output"|"unique", message_count, token_count,
  source_event_id}` — drives the KV-cache-aware substitution engine. **This design adds a
  fourth `InputSegment` type, `"tool_output"`** (§4.1a); it is the only runtime change.
- `messages` are plain OpenAI-style JSON dicts (`{role, content}`, plus
  `tool_calls` / `tool_call_id`), NOT `ReplayMessage` objects.

### 2.2 The base-class contract (`ReplayGraphSessionGeneratorBase`)
We use the **lazy** init path (mirror `OTelTraceReplayDataGenerator.__init__` as the
template). The subclass's own `__init__` MUST:
1. Build `SyntheticAgentSessionsConfig` (which **subclasses `SessionReplayConfig`** — see §8)
   and pass it as `replay_config=` into `super().__init__(...)`. The base reads
   `inject_random_session_id`, `max_wait_ms`, `override_tool_call_max_tokens` off
   `self.replay_config`; if it is a plain `BaseModel` these knobs silently no-op (or raise
   `AttributeError`). Synthetic mode **pins `inject_random_session_id=False` and
   `duplicate_sessions_target=None`** (§2.2a/§6/C6) — the base still reads them, but they are
   inert and not exposed. `max_wait_ms` and the `override_tool_call_max_tokens` pinning behave
   as documented (§8/C3). (`duplicate_sessions_target` is not auto-applied on the lazy path
   anyway — the base only duplicates in the *eager* `_duplicate_sessions_if_needed` (`:1097`),
   which the lazy generator bypasses — so pinning it `None` is consistent with the lazy path.)
2. Call `self.initialize_sessions_lazy(session_ids)` itself — **the base does not** — passing
   the `num_sessions` distinct IDs (no `_dup{n}` expansion; duplication is not supported, §2.2a).
3. Override `_build_session(session_index) -> Optional[ReplaySession]`, returning a
   `ReplaySession` carrying `session_id` / `source_id` / `session_index` (the base
   synthesizes no IDs).

### 2.2a `duplicate_sessions_target` policy: not supported in synthetic mode
`duplicate_sessions_target` is **not supported in synthetic mode** — it is pinned `None` and
not exposed. Rationale: `num_sessions` already sets the session count directly, and each
session is distinct **and** reproducible (§2.3a), so the OTel use case for this knob —
corpus-amplification of a *small recorded trace corpus* with controlled near-duplicates —
is meaningless here (there is no small corpus to amplify; you simply set `num_sessions`).
Pinning it inert also avoids conflict with the determinism (§2.3a) and cross-session
KV-cache-sharing (§6) stories. Same "pinned, not exposed" pattern the spec uses for
`override_tool_call_max_tokens` (§8).

Everything else (session lifecycle, `load_lazy_data`, substitution, tool handling,
eviction) is inherited unchanged — **except** the one additive `tool_output` branch in
the substitution engine that fan-out needs (§4.1a); the OTel/Weka paths never emit that
segment type, so they are behaviorally untouched. (The synthetic generator emits `ReplayGraph`
objects directly and never calls `build_graph`, which is offline-only — §2.4.)

### 2.3 Load-bearing invariants (what MUST be real)
Benchmarking measures **token counts** and **prefix-sharing (KV-cache)** over
token ids; semantic meaning is not measured. But these are structurally required:
1. Tool-call `arguments` must be a `json.loads`-able **string** (strict vLLM 400s
   on object args — see the OTel tool-call work).
2. Every forced tool name must appear in that turn's `tool_definitions` **with a
   top-level `name` field** — the replay forcing lookup reads `t["name"]`
   (`replay_graph_session_datagen.py:367`), NOT nested `function.name`. If the emitted
   `tool_definitions` use only the nested OpenAI shape, `available` is empty and
   single-name forcing silently degrades to `"required"` (see C1/§4).
3. `tool_call_id` must link a producer's tool call to the consumer's `role:tool`
   message; `#role:tool == #tool_calls` in matching order so the **positional**
   `tool_call_id` rewrite works (`replay_graph_session_datagen.py:723–739`).
4. Every emitted `GraphEvent` must have non-empty `call.messages` — the scheduler skips
   empty-message events (`:1292–1294`; it logs a `warning` but does not fail, so the skip is
   easy to miss); a predecessor that was skipped makes its successor wait on a never-scheduled
   id → session hang/timeout.
5. Token-count **shape** must be realistic (Exgentic reference: prompts bimodal —
   a ~800–4000 cluster and a dense ~22K `available_tools` cluster; outputs
   right-skewed with many tiny routing/tool-call turns).
6. A `tool_calls`-bearing assistant message drops its **`content`** at load (upstream/main
   `:1530–1535`: the built message keeps only `role` + `tool_calls` + `reasoning_content`,
   never `content`). So any **prose/filler `content`** must be its **own** content-only
   assistant turn, never merged onto a tool-call turn. This applies to `content` **only** —
   `reasoning_content` is a **distinct field that IS preserved** on a tool-call message
   (`:1531/:1533`). The two are not in tension: the invariant governs `content`, while
   reasoning (if ever emitted, §12) rides the same tool-call turn as `reasoning_content`,
   **not** as a separate turn. v1 emits neither reasoning nor inline filler on tool-call
   turns, so the invariant is simply "filler is its own turn."

### 2.3a Determinism contract: `_build_session(N)` MUST be a pure, index-addressed function
The lazy runtime builds each session's graph **twice, in two different processes**:
`_ensure_session_built` runs it once in the **parent** (on dispatch) and again in the
**owning worker** (on replay) — see the docstring at `replay_graph_session_datagen.py:1120`
("triggered the first time the session is dispatched (parent) or replayed (worker)") and
`:1141` (`self._build_session(session_index)`). The request is addressed by
`(session_index, local_event_index)` (`:61–67`), so the parent-built and worker-built graphs
for the same index **must be byte-identical** — same event count, same dict insertion order,
same event IDs, input segments, theme selection, branching, sampled token counts, and tool
catalogs. If they diverge, the worker replays a graph the parent never scheduled → wrong
`local_event_index` → corruption. This makes `_build_session(N)` a **pure function of
`(config, N)`**, and forces three rules:

1. **No generator-level mutable RNG.** A single advancing `self._rng` is unsafe: the parent
   and worker build sessions in **different orders** (parent by dispatch order, worker by
   which sessions it owns), so a shared stream desyncs. **Each session derives its own
   session-local RNG from a stable seed**, computed with no cross-call state.
2. **Stable seed derivation — NOT `hash(session_id)`.** Python salts `str.__hash__` per
   interpreter unless `PYTHONHASHSEED` is pinned, so `hash(...)` differs across processes.
   Use a stable hash of `(config.seed, session_index)`:
   ```python
   session_seed = int.from_bytes(
       hashlib.blake2b(f"{config.seed}:{session_index}".encode(), digest_size=8).digest(),
       "big",
   )
   rng = np.random.default_rng(session_seed)
   ```
   (`session_index` is the stable slot identity; the human-facing `session_id` string is not
   an RNG input.)
3. **Path-derived child randomness (recommended).** Derive each node's RNG from its **stable
   graph path** — `session_seed → round r → spawn s → child c → tool-turn t` — by folding the
   path integers into a child BLAKE2b/`SeedSequence.spawn`. This keeps a "single seeded
   pre-order walk" **session-local** *and* makes each node's draw independent of its
   siblings' existence, so adding one early event does not shift every later value. It also
   means the walk order within a session need not be globally fixed, only the per-node seed.

Everywhere the spec says "a single seeded pre-order walk," read it as **session-local**:
one independent seeded walk per `_build_session(N)`, seeded as above.

**Concrete plumbing gotcha:** `sample_from_distribution(config, count, rng=...)`
(`utils/numeric/distribution/utils.py:112`) takes an **optional** `rng` and, when it is
`None`, falls back to an **unseeded** `np.random.default_rng()` (`:138–139`). Every draw in
`_build_session` MUST pass the session-local (or path-derived child) `rng` explicitly —
omitting it silently makes the build non-deterministic and the parent/worker graphs diverge.
The cross-process test below is what catches an accidental `rng=None`.

### 2.4 `build_graph` is NOT on the synthetic path
`build_graph` (`otel_trace_to_replay_graph.py`) is the **offline OTel/Weka
conversion** that turns recorded `RawCall`s into a `ReplayGraph`. The synthetic
generator **produces `ReplayGraph` objects directly** — it authors
`predecessor_event_ids`, `input_segments`, and `expected_output_*` itself, and the
runtime reads those verbatim (`replay_graph_session_datagen.py:1319, :456–461,
:1421`). Consequences:
- The `build_graph` multi-part-tool-response *folding* (which lives in
  `_replay_message_to_dict`) does **not** run on synthetic output. The real runtime
  constraint is invariant #3 above (positional `tool_call_id` rewrite), which holds
  independently of `build_graph`.
- **Causal edges are the authored `predecessor_event_ids`, NOT byte-matching.** The
  verbatim-objective seam (§3/§5) is a *coherence/topic-threading* device only; it
  does not "form the edge." (Even on the OTel path, output-matching requires an
  **assistant**-role message, so a `user`-role objective could never match.)
- `build_graph` may still be used as a **test oracle** if desired, but it is not in
  the generator's data path.

### 2.5 Real-data shapes (verified vs `Exgentic/agent-llm-traces-v2`)
Verified against the real HF parquet dataset (3 independent full-census sampler scans
across all 9 shards). These findings correct assumptions taken from a flattened
derivative file:
- **Role vocabulary is `user` / `assistant` / `tool`** (+ rare `developer`). There is
  **no `system` role** (the system prompt lives in a `gen_ai.system_instructions`
  attribute or the first `user`/`developer` text part) and **no `document` role**
  (a `document` role in any flattened file is an artifact; RAG content rides inside a
  tool result or as user prose).
- **Message shape is governed by the *harness*** — four conventions (see §4.2). The
  design originally modeled only `openai_solo`/`tool_calling`. All four reduce, in the
  replay runtime, to `{role, content, [tool_call_id]}` dicts, so supporting them is an
  **emission choice**, not new runtime code.
- **Tool-call `arguments` are JSON objects** in the source (must be `json.dumps`-ed to
  a string at emit, inv #1); **tool results are doubly/triply JSON-escaped**.
- **The dominant real tool-call message bundles `thinking`+`text`+`tool_call` in ONE
  message.** The runtime drops **`content`** (the `text`) on `tool_calls` messages but keeps
  `reasoning_content` (the `thinking`) — inv #6. Since v1 emits no reasoning, the only piece
  that must move to a **separate** turn is the text/filler — a **runtime-imposed divergence**
  from the real shape that inflates turn count and reshapes the per-turn token histogram.
  Accepted and documented, not a fidelity claim.
- **Parallel tool calls per turn are common** (measured across all shards, n=17,733
  tool-call messages): ~88% carry 1 call, ~8% carry 2, and a long tail runs to 20–58
  calls in one message. By harness the multi-call share is substantial — `claude_code`
  18.3%, `tool_calling` 17.9%, `openai_solo` 3.8%. So `parallel_tool_calls_per_turn` (§8)
  is a **real fidelity target**, not synthetic-only — the real shape is heavy at 1 with an
  ~8–12% shoulder at ≥2 and a light long tail. It is **opt-in** — the user sets
  `parallel_tool_calls_per_turn` to a real-shaped distribution (see the example config in
  §13.5); it is NOT the default, which is **fixed 1** so the
  out-of-the-box generator never knowingly dangles (see C2 for the replay
  caveat).
- **Published-dataset sessions are flat, single-agent, sequential** (0/72 nested in
  Exgentic v2). Multi-agent fan-out does **not** appear in that dataset — but it is a real
  pattern in the wild, confirmed by a collected Claude Code trace (§2.6), so it is a
  synthetic capability the source *dataset* lacks, **not** a shape that never occurs.
- **Known bug** (filed, `docs/superpowers/specs/bug-claude-code-tool-call-id-dropped.md`):
  the OTel converter drops `tool_call_id` for tool results on non-`tool` roles
  (`claude_code`), so `claude_user_role` results carry no id even in real OTel replay
  today — our generator matches that behavior (§4.2).

### 2.6 Validation against a real fan-out trace (collected Claude Code session)
A real Claude Code session, exported as a Jaeger/OTel trace (`gen_ai.*` schema, same as
Exgentic), was analyzed to confirm the multi-agent capability corresponds to something that
actually happens — and that our knobs can produce it. Distilled fingerprint (127 spans; 24
`chat` LLM calls; 34 tool executions; 1 root `claude_code.interaction`):
- **A single autonomous root session** (one task, run end-to-end — `rounds_per_session=1`).
- **A concurrent 3-way sub-agent fan-out:** one `chat` turn emitted **three `Agent` tool
  calls in one assistant message**, spawning three `general-purpose` sub-agents in parallel
  ("Assess documentation / code / performance" of the *same* target file). The sub-agents'
  own `chat` calls are nested one level below (span-tree depth 2: `interaction → … →
  tool.execution → chat`). This is the recursive fan-out Exgentic lacks.
- **Coherent objective threading:** all three sub-agent prompts thread the **same entity**
  (a file path) into distinct verbs — exactly the closed-list Mad-Libs objective seam (§5),
  entity = file path.
- **Long root tool-loop** (~21 root-level tool calls before/around the fan-out) drawing on a
  small *called* set (`Bash` ×18, `Read` ×13, `Agent` ×3) out of a **24-tool advertised
  catalog** — called ⊂ advertised, i.e. catalog inflation (§9).
- **Parallel calls per turn are a genuine mix:** across the 24 turns, 4 turns made 0 calls
  (text answer), 8 made 1, 10 made 2, 2 made 3 — reproducible by setting
  `parallel_tool_calls_per_turn` to the **opt-in** real-shaped distribution (the default is
  fixed 1; see the example config in §13.5).

**How our knobs reproduce it:** `rounds_per_session=1`, `fanout_probability>0` with
`sub_agents_per_spawn≈3`, `max_depth=2`, wide `tool_turns_per_loop`, `tool_definitions_per_agent≈24`,
the opt-in real-shaped `parallel_tool_calls_per_turn`, and a theme whose entity pool is file paths. The
resulting graph has this trace's topology (autonomous root → concurrent 3-agent fan-out →
merge) and load profile (catalog size, tool-loop depth, parallel-call mix).

**One documented divergence:** the real spawn bundles the **three `Agent` calls in one
assistant turn**; our generator emits sub-agent dispatch as **separate single-call turns/
events** (§4.1/C2). Same concurrency, same fan-in load, same merge — but the dispatch turn
count differs. Bundled K-way spawn-in-one-turn is deliberately avoided in v1 (a live model
emitting fewer than K calls would dangle ids → 400); reproducing it byte-for-byte is the
parallel-call count-reconciliation future item (§12).

**Metering artifacts (not modeled, correctly):** many spans show `input_tokens=2` — Anthropic
prompt-cache accounting (the cached prefix is not re-billed), not a real 2-token prompt; and
`claude_code.tool.blocked_on_user` spans are permission gates. Neither is graph structure.

## 3. Architecture: three content layers

Content is produced by a **layered** model. Each turn's text is produced by a
**fixed rule keyed to the turn's role** — there is no turn-by-turn narrative
generation. The generator lays down the entire graph at once (per session).

### Layer 1 — STRUCTURE (always procedural)
Topology, `event_id`s / `call_id`s / `tool_call_id`s, tool-call message shapes,
`input_segments` (shared/output/unique), timing (`t_start/t_end/wait_ms`), and
**bulk tool-definition schemas**. Meaning-free; scales to any count.

### Layer 2 — COHERENCE seam (entity-threaded template)
The **objective line**: when an agent spawns a child, the child's `objective`
argument is emitted **verbatim** as the child's opening `user` message, threading
the goal parent → child → grandchild so the trace reads coherently. This is a
*coherence/topic-threading* device — it does **not** form the DAG edge (edges are
the authored `predecessor_event_ids`, §2.4). The objective is closed-list Mad-Libs
(see §5), NOT a narrative skeleton.

### Layer 3 — FILLER (pads free text to the token budget)
Fills the remaining free-text surfaces to hit config-`Distribution` token counts, via the
bounded iterative `converge_to_exact_length_text` fit. **Two implementation facts force the
approach (both verified against `datagen_utils.py:85–124`):**

1. **Use a best-candidate wrapper, not a plain `try/except`.**
   - *Claim:* "catch the error and accept the closest length" is **not implementable as stated**.
   - *Why:* on non-convergence after 20 iterations the utility `raise`s a `ValueError` carrying
     only the *last measured length* (a number); the generated `text`/`ids` are local and thrown
     away (`:120–124`), so there is no candidate to accept.
   - *Do:* the generator MUST use a **best-candidate wrapper** — call the fit but track, across
     iterations, the `(text, ids)` whose length is closest to target, and return that best
     candidate with metadata `{target_tokens, actual_tokens, exact: bool}` when exact
     convergence fails. (Implement as a thin wrapper that re-runs the same bounded loop while
     remembering the best seen — or extend the utility to return its best attempt instead of
     raising.) A non-`exact` result is accepted (logged at `debug`), not a session failure.
2. **Compute the filler budget *before* fitting; a non-positive budget floors the target.**
   - *Claim:* a turn is fixed content + filler, and the filler budget
     `filler_budget = target_tokens − tokens(objective + marker + required_structure)` can go
     **negative** (`marker` = the `[--- ignore the preceding filler; actual content follows ---]`
     delimiter, ~15 tokens on its own).
   - *Why:* several example minima are 5–10 tokens — **smaller than the marker alone** — so the
     mandatory content already overshoots the target and the fit can *never* converge down (it
     cannot delete the marker), guaranteeing the (1) failure.
   - *Do:* check `filler_budget` up front and, if `≤ 0`, **raise the sampled target to a
     documented floor** = `tokens(fixed content)` (a per-surface `min_turn_tokens` derived from
     the marker+objective cost, so the turn is exactly its mandatory content, no filler) and log
     it — rather than spinning 20 iterations into a `ValueError`. The marker is only emitted when
     `filler_budget > 0`, so a fixed-content-only turn carries no dangling delimiter.

**Filler never emits braces/keys/ids and never occupies a region where a coherence-seam
substring is expected.**

### Per-surface assignment (v1)
| Surface | Producer |
|---|---|
| Topology, ids, tool-call shapes, tool schemas, segments, timing | Structure |
| Bulk tool defs (→ ~22K prompt mode) | Structure: **tool-catalog inflation** (always-on, gated by `tool_definitions_per_agent`) |
| Sub-agent dispatch / objective (the seam) | Coherence template |
| **System prompt** (the shared head, every agent) | **Theme `system_prompt` template** + realistic system-flavored boilerplate padding — **NOT** Shakespeare (§4.2/§7) |
| **Tool-result bodies** | **artifact template** (canonical name; = "shaped synthetic filler", detailed below) |
| User/assistant reasoning & answer prose | **Filler** (Shakespeare prose corpus + marker) |
| Cross-session-shared regions | Byte-identical fixed content: the `system_prompt` head (theme-authored) + any fixed-seed shared filler (see §6) |

The tool-result producer is a **single component** referred to throughout by one
canonical name, **artifact template**. Filler marker convention (user request):
filler prose is capped with `[--- ignore the preceding filler; actual content
follows ---]` to delimit padding from real content.

Note on "shared": the runtime's `InputSegment.type=="shared"` is an **intra-session**
prefix reuse (requires a same-session `source_event_id`). The **cross-session** KV
reuse targeted by §6 is a *different* thing — byte-identical filler content across
sessions — and is not the `InputSegment` "shared" type. Keep the two distinct.

### The artifact template (tool responses) — shaped synthetic filler
The **artifact template** is our synthetic analog of OTel replay's **recorded** tool
response (a real source file, a real log dump). In OTel replay that content is
recorded and replayed verbatim into the next call; here **nothing is recorded, so
the content is manufactured at build time** and frozen into `GraphCall.messages`,
then handled by the **existing** replay injection/substitution paths (no new
replay-time code). It is NOT a real file/log — it is content of the right *shape*
and *size*, with meaningless interior text. Multi-call turns emit K **distinct**
`role:tool` messages, one per `tool_call_id`, count + order matched to the K calls
(invariant #3).

Three properties are reproduced faithfully; one is not:
- **Size (faithful):** each tool's result token count is drawn from a `Distribution`
  (use a long-tail lognormal so some results are ~200 tokens and some ~20k — like a
  real file/log dump). This reproduces the KV-cache/prefill load a big recorded
  artifact imposes on the next call.
- **Shape (faithful):** each tool declares an `artifact_kind`, and a matching
  generator emits that surface form:
  - `code` → indentation, `def`/`function`/braces, comment & import-like lines
  - `logs` → `<ISO-ts> <LEVEL> component=... msg=...` lines, seeded ts/levels
  - `metrics` → the metric/table rows shown in §5
  - `json` → a valid JSON envelope with a large string/array body
- **Downstream load (faithful):** the result is frozen into `GraphCall.messages`
  and fed to the next call via the **same** injection/substitution path as an
  OTel-recorded result.
- **Semantic content (NOT faithful):** the interior bytes (identifiers, log
  messages) are **filler** — Shakespeare corpus slices padded to the budget
  inside the structural frame. The next agent cannot *reason* about them (fine,
  because tool choice is forced, §4).

Genuine real artifacts (real file/log bodies) are the deferred tier: LLM-authored
result templates, or **corpus-harvested** result pools (mine real bodies from
Exgentic and store per-tool). They swap in behind the same interface (see §12).

## 4. How a full session is produced (the walk)

A session has **two node roles** (not a flat "leaf kind"; see §4.2 for why):

- **Root agent** — driven by **principal input**. A *principal input* is either an
  **autonomous task** (the headline case — one task given up front, then the agent runs
  end-to-end with no further external input) or a **human turn** (interactive). Its life
  is a sequence of **rounds**; each round is `[principal input] → [tool-loop of length
  k≥0] → [answer]`, and a round **may spawn sub-agents** (fan-out). `rounds_per_session`
  = N is the **autonomous ↔ interactive axis**: N=1 = autonomous end-to-end run (also
  what exgentic is; §2.5), N≥2 = interactive multi-turn. `tool_turns_per_loop` = k tool-call
  **turns** (sampled per round, **0 allowed** = answer immediately; calls-per-turn is the
  separate `parallel_tool_calls_per_turn` axis).
- **Spawned (sub-)agent** — driven by its **parent**; its life is a **single dispatch**:
  `[task from parent] → [tool-loop of length k≥0] → [answer to parent]`. It **may recurse**
  (spawn its own sub-agents) but is **not** multi-round (no parent↔sub-agent ping-pong).

The dominant, headline workload is **autonomous**: N=1, a long tool-loop, spawning
sub-agents — the spawn/merge happens *within* the single round's tool-loop, and the
final answer consumes the merge (no second round needed). Interactive (N≥2) is the same
structure with the principal injecting again after each answer.

A **session-local** seeded **pre-order walk** (one per `_build_session(N)`, seeded stably
from `(config.seed, session_index)` with per-node path-derived sub-seeds — §2.3a) emits
every turn:
- **round (root)** → emit `[principal input turn]` (objective/seam text) then a tool-loop
  (below); if the round spawns K sub-agents, emit **K separate single-call dispatch events**
  (each: an assistant turn with **one** `dispatch_agent` tool_call, forced single-name so it
  is deterministic and never dangles) + recurse into each sub-agent + a **merge** event that
  depends on all K child terminal events. The merge reconstructs, per child, an
  `[assistant dispatch_agent call (output segment ← dispatch event) , role:"tool" result
  (tool_output segment ← child terminal, §4.1a)]` pair, then emits the round's `[answer]`
  turn. Repeat for N rounds. (The K dispatches are emitted as separate events, NOT one
  assistant turn bearing K parallel `dispatch_agent` calls — that bundled shape would force
  `tool_choice="required"` and risk dangling ids if the live model under-emits, C2. Separate
  single-call events give the same concurrency, since siblings share only the spawn event as
  predecessor and run in parallel, §4.1.)
- **dispatch/sub-agent** → sub-agent runs its single-dispatch tool-loop (turn count from
  `tool_turns_per_loop`, same knob as root rounds), may recurse, returns one answer consumed
  by the parent's merge.
- **tool-loop (k calls)** → for each call: `[assistant reasoning-flavored filler]` +
  `[assistant tool_call (structure + entity args)]` + `[tool result (artifact template)]`;
  the loop **ends with an `[assistant answer]` turn** (matches real sessions, which
  terminate a loop with a text answer, often alongside a final finish/submit call).
  That leading filler is plain **`content`** on its **own** assistant turn — never merged
  onto the tool-call turn, whose `content` would be dropped (invariant #6). It is ordinary
  filler prose, not the `reasoning_content` field (which v1 does not emit; §12).

Producers, by role:
- **user (principal input / sub-agent dispatch prompt)** → **round 1** and every
  sub-agent dispatch use the theme `objective_template` (verbatim copy for the seam);
  **rounds 2..N** (interactive) are built as `[back-reference connective] + [followup
  body]`, where the body is drawn from the theme's `followup_templates` pool
  (cycled/seeded, entity-threaded) and the connective is a short canned lead from
  `followup_connectives` (e.g. *"Following up on the previous result, "*, *"Given that, "*)
  so the turn **reads as reacting to** the prior answer. Fidelity note: this is
  presentational — the connective does not causally derive from round N−1's content (that
  answer is filler); the follow-up is on-topic and plausible, not logically responsive.
  Token/KV behavior is faithful; cross-turn *reasoning* is not (consistent with the
  coherent-seam/filler-body model). Fallbacks: no `followup_templates` → a generic
  "continue" prompt; no `followup_connectives` → a built-in default connective list. Keep
  connectives OUT of any cross-session `shared` region (they repeat — §6/C6).
- **assistant dispatch / tool_call** → structure (shape) + entity-bound args.
- **tool (response)** → **artifact template** (§3), one distinct `role:tool` per
  `tool_call_id`.
- **assistant reasoning-flavored / answer prose** → plain-`content` filler on its own turn
  (never the `reasoning_content` field; §12 / inv #6).
- **child answer delivery** → the parent's **merge** (continuation) event awaits all K child
  terminals as `predecessor_event_ids` and, per child, carries an `output` segment (the
  child's `assistant dispatch_agent` call, `message_count==1`, source = the dispatch event)
  followed by a **`tool_output`** segment (`role:"tool"`, source = the child terminal event)
  that injects the child's **live** answer text while preserving role + `tool_call_id`. A
  plain `output` segment would clobber the `role:"tool"` slot with the child's assistant
  message — see §4.1a for why the new primitive is required and §4.1 for the exact layout.

"The conversation continues" without generation: growing context is real (each
turn's `messages[]` includes prior branch turns as shared/output/unique segments),
but no turn's text depends on *understanding* the previous turn.

### 4.1 Event/segment layout for tool linkage and merges
- **Tool linkage:** event N emits the assistant `tool_call(s)`; the consuming event
  N+1 carries `[output-segment (message_count==1) = that assistant turn]` immediately
  followed by exactly K `role:tool` messages in call order, so the positional
  `tool_call_id` rewrite (invariant #3) lines up.
- **Dispatch + merge fan-in (decomposed; K dispatch nodes + 1 merge node — NOT the same
  node).** The K dispatch events and the 1 merge event are distinct nodes with distinct roles:
  - **K dispatch events**, one per sub-agent. Each is the *predecessor* the child depends on;
    it carries the parent context up to and including the assistant turn with **one**
    `dispatch_agent` tool_call. The child event lists its dispatch event as its predecessor.
  - **1 merge event** — the parent's continuation (its round `[answer]` turn, or its next
    tool call). It lists **all K child terminal events** in `predecessor_event_ids` and awaits
    them (`require_async`, `:456–461`). Its message list reconstructs the K
    `assistant dispatch_agent` → `role:tool result` pairs, one pair per child, then the
    parent's own continuation turn. Each pair is `#role:tool == #tool_calls == 1` (inv #3),
    single-call forced so the `tool_call_id` is static (no positional rewrite dangle, C1/C2).
    The merge is **one `GraphEvent`** (the K pairs live in its `messages`, not as K separate
    events) — so it counts as **1** toward `max_events_per_session`; the per-child cost the
    budget check attributes to a spawn is the K *dispatch* events + K *child* events + this 1
    merge (§8).
  - The child's **assistant** dispatch call in each pair rides an existing `output` segment
    (`message_count==1`, source = the dispatch event) — the branch-1 whole-message
    substitution (`:604`) is correct here because the slot IS an assistant turn.
  - The child's **answer** must land in the following `role:"tool"` slot **with its role and
    `tool_call_id` intact** — which a plain `output` segment cannot do (it would clobber the
    `role:"tool"` slot with the child's `assistant` message → dangling call → 400). This is
    what the new `tool_output` segment is for; see §4.1a for the full rationale and semantics.
  - There is **no** single assistant turn bearing K parallel `dispatch_agent` calls, so the
    "K role:tool must immediately follow one K-call assistant turn" wire constraint does not
    apply — each pair satisfies 1-call/1-result on its own.
  - **Substitution sources must be ancestors.** The merge sources both each dispatch event
    (assistant call) and each child terminal (tool result). Both are ancestors of the merge
    in the DAG, so `require_async` has already resolved them before substitution runs — no
    cycle, no forward reference.
- **Parallelism** is expressed structurally: sibling sub-agents share the spawn event
  as their sole predecessor with **no inter-sibling edges**; the loadgen dispatches
  all ready events and they run concurrently. `t_start_ms`/`t_end_ms` are **cosmetic**
  (DOT labels only); the real fan-in wall-clock is the `require_async` block on the
  slowest child, so the merge event is **pinned to `wait_ms = 0`** (it must not add a
  sampled delay on top of that block — see §8 "Timing"). Leaf turns draw their `wait_ms`
  from `tool_call_latency_sec` (tool-exec / dispatch / round-1 startup) or, for a user's
  rounds 2..N in an interactive session, from `user_think_time_sec` — capped by
  `max_wait_ms`.
- **`predecessor_dependency_types`** is a required field (no default → constructing
  `GraphEvent` without it raises `TypeError`) and must be populated per predecessor from
  valid `DEPENDENCY_TYPE.value` strings: `full_match` for merge/output-injected edges (the
  predecessor's whole output is re-injected), `tool_call_ids_matched` for tool edges,
  `temporal` for timing-only. The merge's **`tool_output` edge** (child terminal → merge) also
  uses **`full_match`** — the child's whole output text is injected as the tool result; since
  the map is DOT-only (below), the label is cosmetic and `full_match` renders correctly.
  **Note:** `output` is an `InputSegment.type`, a *different*
  vocabulary — it is **not** a valid `DEPENDENCY_TYPE` (the six values are `full_match`,
  `tool_call_ids_matched`, `split_parts_matched`, `content_and_split_tools_match`,
  `drop_content_split_parts`, `temporal`; the `DEPENDENCY_TYPE` enum in
  `otel_trace_to_replay_graph.py:446–456`). An `output`-labeled edge falls to the exporter's
  `else` branch and mis-renders as temporal. The map is **DOT-only** (the runtime never reads
  it; the exporter `export_replay_graph_to_dot.py:154` does `.get(pred_id, "temporal")` so an
  empty map does not raise — it just renders every edge as temporal). Note the field is typed
  `Dict[str, str]` (raw strings, not the enum), so the generator writes the `.value` strings
  directly.
- **Round-to-round context growth (interactive N≥2):** round K's principal-input event
  must carry the whole prior conversation, or the "multi-round" session is structurally
  valid but empty (and §11's checks still pass). Its `input_segments` are:
  `[shared(the common prefix of prior turns; source_event_id = an awaited predecessor whose
  INPUT is that prefix), output(message_count==1; source = round K−1's terminal answer
  event), unique(the new follow-up principal turn)]`. **Both** `source_event_id`s MUST also
  appear in `predecessor_event_ids` so substitution runs after `require_async`. Note
  `shared` reads the predecessor's **input** while `output` reads its **answer** — hence
  two segments (mirrors `otel_trace_to_replay_graph.py:781–877`).

### 4.1a Required runtime extension: the `tool_output` input segment
Fan-out is the one shape the current runtime cannot express (§4.1). The `output` segment
substitutes a predecessor's *whole* output message: when the source event has completed it
fetches `get_message_by_event_id` and appends that message verbatim (`:604`). Every completed
event records its output as `{role:"assistant", …}` (`:860/864/914/918`), so an `output`
segment aimed at a child terminal would overwrite the parent's `role:"tool"` placeholder with
an `assistant` message — losing the role and the `tool_call_id`, leaving the `dispatch_agent`
call dangling. The role-preserving content-only branch (`:631–634`) only runs when the source
registered no output message, which never happens for a completed event, so it is unreachable
for this purpose.

**Add a fourth `InputSegment.type`: `"tool_output"`.** Semantics (mirrors the `output`
branch but content-only, role-preserving):
1. Requires exactly one recorded message (`message_count == 1`); that message MUST have
   `role == "tool"` (validate; else log + fall back to recorded, like the `output` guard at
   `:535–542`).
2. Fetch `registry.get_output_by_event_id(source_event_id)` — the child's **text**, not its
   message.
3. Replace **only** `content` on a copy of the recorded message (`dict(msg)`); **preserve
   `role:"tool"` and `tool_call_id`**.
4. If the child produced no output text (unavailable / failed), fall back to the recorded
   placeholder content (do not fabricate).
5. The **existing** positional post-pass (`:517–527`, `:723–739`) rewrites the static
   `tool_call_id` from the immediately-preceding live `assistant dispatch_agent` call — the
   `tool_output` slot participates in that pass exactly as a recorded `role:tool` message
   does today, so no change to the post-pass is needed.

`needs_substitution` (`:476`) must also test `seg.type == "tool_output"` so the substitution
path is entered. This is the **only** runtime code change the whole design requires; it is
additive (a new `elif` branch + one predicate term) and touches no existing segment behavior.
Sourcing rule for the merge: each child pair is `[output(source = dispatch event) ,
tool_output(source = child terminal event)]`; both sources are ancestors of the merge, so
`require_async` has resolved them before substitution (no cycle).

### 4.2 Why node roles, not a `leaf_kind` enum
The design uses two node roles (§4) rather than a `leaf_kind` enum. Multi-round-ness is a
property of **being the root agent** (which receives repeated principal input), not of a
leaf; and multi-round and tool-use are **orthogonal** (a round can be a multi-round
conversation that *also* runs a tool-loop). A `leaf_kind` enum cannot express that
orthogonality, so node roles are the correct factoring.

The two node roles are parameterized by two knobs on the root:
- `rounds_per_session` (N) — number of principal inputs; N=1 = **autonomous** end-to-end
  run, N≥2 = interactive (a human injects follow-ups).
- `tool_turns_per_loop` (k tool-call turns, sampled **per tool-loop**, 0 allowed) — the
  number of tool-call turns for both root rounds and spawned sub-agents (one shared knob);
  calls *within* a turn are `parallel_tool_calls_per_turn` — §8.

The named shapes are just **particular combinations** of these knobs:

| Shape | rounds N | tools/round k | notes |
|---|---|---|---|
| plain answer (`plain_answer`) | 1 | 0 | one shot, no tool call |
| **autonomous tool-loop** (`tool_loop`; exgentic) | 1 | ≥1 | one task → tool-loop → answer |
| **autonomous + fan-out** (headline) | 1 | ≥1 | + round spawns sub-agents |
| plain conversation (`conversation`) | ≥2 | 0 each | no tools (non-agentic / chat) |
| chat-with-tools | ≥2 | varies per round | interactive: ask→tools→answer, repeat |
| any + fan-out | any | any, **+ round spawns sub-agents** | composes with all of the above |

A **single-agent** session = `fanout_probability=0` (no round spawns). A **sub-agent** is
always single-dispatch (one task → tool-loop → one answer, may recurse) — multi-round is
root-only (no real-data analog for conversational sub-agents; §2.5). Chat-with-tools that
*also* spawns fan-outs on some rounds composes naturally.

**Advertise ≠ force.** A round with k=0 answers directly, but its request **still carries the
full `tool_definitions` catalog** (with `expected_output_is_tool_call=false`, so replay does
**not** force `tool_choice`) — this reproduces the dominant real "tools advertised, answered
directly" shape (§13.2). This is unconditional and matches how real agent harnesses build every
request: the tool catalog is attached *before* the model decides whether to call a tool, so a
turn that answers directly still paid the catalog's prompt cost. A **bare** direct answer with
no tools advertised is not a separate knob — it is simply `tool_definitions_per_agent = 0` (the
non-agentic / plain-chat baseline).

**Tool results use the OpenAI convention only.** Results are always emitted as
`{role:"tool", content, tool_call_id}` — the one convention that preserves `tool_call_id`
linkage and forced `tool_choice` cleanly. There is no config option for this. (For
context: the real dataset also has `claude_code` — result on the `user` role, id dropped,
see the bug doc — and `smolagents` — fused `"Observation:"` prose — conventions (§2.5).
We do not reproduce them: we are *generating*, not recording, the runtime normalizes
everything to `{role, content, [tool_call_id]}`, and byte-level wire mimicry is not a
benchmarking goal.)

**System prompt emission site.** `shared_system_prompt_len` (§8) is emitted as a **dedicated
`{role:"system"}` message** at the head of **every agent's first call** (root *and* every
spawned sub-agent — §6, option b), marked a cross-session `shared` region for KV reuse (§6).
A distinct `system` message (rather than a slice of the first `user` turn) keeps the shared
prefix cleanly **distinguishable** — one addressable message in the graph, DOT export, and
debugging — and matches how real agent harnesses and OpenAI-native serving actually send a
system prompt. This **diverges** from the Exgentic role vocabulary (§2.5: the source has no
`system` role — the preamble rides `gen_ai.system_instructions` or the first `user` text
part), but that is the same *generate-cleanly-don't-mimic-the-recorded-dialect* choice made
for the OpenAI-only tool convention (§4.2 above): the runtime normalizes a leading `system`
message with no special-casing (verified — only `role=="tool"` is branched on; `system`
flows through as a plain `{role,content}` dict). The first-`user`-text placement remains a
documented alternative for byte-fidelity to Exgentic's role set.

**System-prompt CONTENT is NOT Shakespeare filler** (unlike every other free-text surface).
A system prompt is semantically load-bearing in real traffic and is the region we cache and
re-send to every agent, so filling it with corpus prose would be wrong and would look absurd
in exports. Its content is instead a **theme-authored `system_prompt` template** (§7) — a
realistic, **session-invariant** agent preamble (role framing + tool-use guidance +
constraints/formatting rules). It MUST stay session-invariant (no per-session entities/
timestamps — those belong in the `user` objective turn) so it is byte-identical across
sessions and caches (§6). If the theme omits `system_prompt`, a built-in **generic agent
system prompt** is used. To hit exactly `shared_system_prompt_len` tokens, the remainder is
padded with **realistic system-prompt-flavored boilerplate** — repeated policy/constraint/
formatting lines and **fixed theme-level** guidance text — NOT Shakespeare, and **NOT** the
agent's sampled tool catalog. Catalog-derived padding would be **invalid**: if
`tool_definitions_per_agent` varies by agent/session, the padding is **not** session-invariant,
so the system prompt would differ per session and never cache. The padding therefore uses only
**fixed, theme-level** boilerplate (independent of the sampled catalog), so it is truly
byte-identical across the cohort. (If a config wants the *advertised catalog* itself inside the
shared prefix — the realistic case — that is handled by the **canonical cohort catalog** rule
in §6, not by folding catalog text into the system message.) The padding is deterministic and
fixed-content; this is the one surface where `converge_to_exact_length_text` fits system-
flavored text, not corpus filler.

### Tool choice is FORCED by default (v1 default; k=0 direct-answer rounds opt out)
The model never *decides* whether a tool is relevant — the generator records
`expected_output_is_tool_call` + `expected_output_tool_names` on each tool turn,
and replay **forces** `tool_choice` to the recorded function(s): a single name →
`{"type":"function","function":{"name":...}}`, K names → `"required"`. Same
mechanism as OTel replay.

**C1 — single-name forcing requires a top-level `name` in `tool_definitions`.** The
replay lookup builds `available = {t["name"] ...}` from the **top-level** key
(`replay_graph_session_datagen.py:367`); it does not read nested `function.name`. So
each `GraphCall.tool_definitions` entry the generator emits MUST carry a top-level
`name`. The server-facing `tools` payload accepts the nested OpenAI shape, so a
missing top-level `name` fails **silently** — `available` is empty, and *every* tool
turn (even single-call) degrades to `tool_choice="required"`, destroying the
single-call determinism below. Implementation: either emit tool defs in flat form
(top-level `name`) or add a build-time step that copies `function.name` up.

Consequences of correct single-name forcing:
- **Deterministic, reproducible** single-call turns — the same graph replays
  identically, so latency/throughput are comparable across runs and servers.
- **No dangling ids** for single-call turns — the forced call is guaranteed.
- **Tool relevance is NOT required.** Tools exist to be *invoked* (token volume,
  tool-call parsing, KV-cache behavior), not *selected*.

The only thing forced-only cannot do is benchmark the model's own tool-**selection**
behavior; that is a distinct model-quality experiment, deferred (see §12).

**C2 — multi-call `"required"` (parallel-in-one-turn) is a real shape, replayed
BEST-EFFORT.** Parallel tool calls are common in the source (~12% of tool messages, up
to ~18% per harness; §2.5), so `parallel_tool_calls_per_turn` models that real
distribution (heavy at 1, ~8–12% at ≥2, light long tail). The replay caveat: a K-call
turn forces `tool_choice="required"`, which guarantees *a* call but does **not** pin
which/how-many. The runtime's positional rewrite (`:723–739`) relinks ids for the first
`min(#role:tool, #live_calls)` messages but does **not** add or trim `role:tool` messages
to match the live count. So if the live model emits fewer than the recorded K (e.g. 1 of
3), the surplus recorded `role:tool` messages keep stale ids → **dangling ids that vLLM
400s**, failing that session. **v1 default avoids this entirely** — `parallel_tool_calls_per_turn`
defaults to **fixed 1** (§8), so the out-of-the-box generator never emits a K>1 turn and never
knowingly dangles. The real-shaped distribution is **opt-in** (set the knob to it; see the
example config in §13.5). When opted in, K>1 is best-effort: drop sessions that 400 on a count mismatch;
the mismatch rate is itself a useful signal about the model under test. Robust count-
reconciliation (trim/pad `role:tool` to the live count) is future work (§12). Note fan-out
across *sub-agents* is unaffected — separate
single-call dispatch events, not K calls in one turn.

**C3 — tool-call turns use their OWN `max_tokens` policy, not `output_tokens_per_turn`.**
The two are different quantities and must not share a knob:
- **Plain-text turns** (reasoning, answers) → `max_tokens` from `output_tokens_per_turn`. This
  is what that distribution is for; it reproduces the "many tiny routing/answer turns" shape
  (§2.3 inv #5).
- **Forced tool-call turns** → `max_tokens` is **derived directly from the known JSON**, not a
  knob: `max_tokens = tokens(expected tool-call JSON) + TOOL_CALL_MARGIN` (a fixed **64-token**
  constant). Because the call is *forced* (deterministic `tool_choice`, our own recorded JSON
  substituted), its output length is essentially fixed by that JSON — there is nothing for a
  larger budget to buy, so no knob is warranted (a distribution's only power here would be
  headroom a forced call doesn't need — cut under the "does it change what a benchmark
  measures?" test). Sizing to the exact JSON + margin guarantees it **never truncates**,
  including multi-call turns (sum the K calls' JSON), closing the failure the old shared-knob
  approach had (a small `output_tokens_per_turn` sample, e.g. 40, would truncate a multi-call
  JSON mid-emit → malformed call → broken session).
- **The old `override_tool_call_max_tokens` boolean is DROPPED on the synthetic path** (§8).
  It was a lose-lose single switch: `True` gave a safe-but-distribution-ignoring
  `max(recorded*4, 4096)` floor; `False` honored the distribution but enabled the truncation
  above. The **derived cap** (`tokens(JSON) + TOOL_CALL_MARGIN`) supersedes both, so the synthetic config does not
  expose the flag.
- **No runtime change, no knob** (verified): `to_request_body` uses the per-event `max_tokens`
  it is handed, which is `event.expected_output_tokens` (`replay_graph_session_datagen.py:1549
  → :1560`). The generator computes `expected_output_tokens` **per event** —
  `sample(output_tokens_per_turn)` for plain-text, `tokens(expected JSON) + TOOL_CALL_MARGIN`
  for tool turns — and the base (with the synthetic config pinning
  `override_tool_call_max_tokens=False`, so the `*4/4096` block at `:350–356` is skipped)
  passes that value through **verbatim**. The whole policy is in what the generator stamps; the
  runtime is untouched and there is no user-facing tool-call-budget knob.

Note: `total_input_tokens` / segment `token_count` are **informational** (DOT +
offline analysis only); the input-token metric is always tokenized live from the
actual messages (`:872/:897`). A K-vs-actual size difference affects only static
annotation, never measured throughput/latency.

## 5. The objective line, unpacked (coherence seam)

Assembled from small closed lists + seeded numbers, all in the theme file:
- `{verb}` — closed verb list: `["Analyze","Assess","Review","Investigate",
  "Diagnose","Inspect","Evaluate"]`, picked by seed (variety at ~zero cost).
- `{subsystem}`/`{label}` — the subtask-tree node's own label (comes from the tree
  we need for topology anyway).
- entities — explicit pools (pick one by seed) **or enumerated ranges**
  (`BP{i}` → `BP0..BP49`; this is how 50+ sub-agents stay coherent — each child
  gets a distinct enumerated entity).
- numbers (`3 → 1180`) — seeded random draws within configured ranges.
- frame — one authored template string per subtask type.

Example rendered objective (emitted verbatim as both parent dispatch arg and child
prompt):
> "Assess Db2 logging and IRLM lock behavior on DBP1 for the commit-latency spike
> that began at 02:14 UTC."

Variety for 1000+ sessions comes from re-binding a handful of authored parts
(`7 verbs × instances × symptoms × enumerated entities × seeded numbers`), not
from authoring 1000 sentences.

## 6. Prefix-sharing (KV-cache) — via the invariant system prompt

There is **no dedicated prefix-sharing knob.** Cross-session KV-cache sharing is achieved
*structurally*, by the parts of the prompt that are byte-identical across sessions by
construction — there is nothing to dial, and no artificial shared-filler padding.

**Two sharing axes, both automatic:**
- **Intra-session (call-to-call).** Within one agent, each call re-includes the prior turns
  (same-session `InputSegment type=="shared"`/`output` refs, §3), so call N+1 is a prefix
  extension of call N. Large, and falls out of the graph structure — not configured.
- **Cross-session (session A vs B).** The shared region is the **head every agent's first call
  begins with: the invariant `{role:"system"}` message (§4.2) + the identical `tools`
  payload.** Both are byte-identical across sessions by construction, so a server's prefix
  cache (vLLM APC / SGLang RadixAttention) hits across sessions with no forcing. Its size — and
  thus the achieved cross-session hit rate — is set by **`shared_system_prompt_len`** (and the
  catalog size); a bigger invariant head = more shared prefix. That single length knob is the
  whole lever (a separate `prefix_sharing_target` fraction was considered and cut — the system
  prompt + tools *is* the realistic shared region; padding Shakespeare into the prefix just to
  hit a fraction measured nothing extra).

**What must hold for the shared head to actually cache (still true, just not knobbed):**
- **Contiguity from token 0.** A cache matches the longest *contiguous* identical run from
  position 0; the shared head must be a true prefix (system message + tools at the front),
  with all per-session content (objective, entities, timestamps, filler) strictly after it.
- **Canonical tool catalog per cohort.** The `tools` payload renders near the front, so for
  cross-session hits the cohort must advertise **one canonical, identically-ordered catalog**
  on the first call. Per-agent catalog *variation* defeats it (sessions diverge inside the tool
  block). If a session's theme fixes the system prompt + catalog, its **cohort** = all sessions
  of that theme (cross-*theme* sharing is not attempted — different heads).
- **Fidelity caveat (chat template).** The generator counts/controls the *emitted* message
  content, not the server's chat-template wrapper (`CustomTokenizer` exposes only
  `count_tokens`, no `apply_chat_template`). The emitted invariant head caches as long as the
  template renders it contiguously from the front — true for the common templates (fixed head,
  then tools, then messages); the template's own fixed head only *adds* shared tokens ahead.
- **Determinism.** The invariant head is config-fixed and independent of `session_index`
  (byte-identical across sessions); per-session content is seeded from the session-local seed
  (§2.3a), never `hash(session_id)`.

**C6 — session-id injection would break it (which is why synthetic mode pins it off).** The
runtime prepends `[SESS:<random>] ` to unique-segment content
(`replay_graph_session_datagen.py:685–704`) when `inject_random_session_id=true` or a session
is a duplicate, to deliberately invalidate KV cache. This interaction is precisely why
synthetic mode **pins `inject_random_session_id=False` and `duplicate_sessions_target=None`**
(§2.2a/§8): both would deliberately invalidate the cross-session KV-cache sharing that is a
core synthetic signal (§6). (Even if injection were on, it would not touch the `{role:"system"}`
head — not a unique segment — so the system-prompt prefix would still cache; but any further
per-session sharing would be destroyed.) With both pinned inert, a clean cross-session KV
benchmark is the default and cannot be silently defeated.

## 7. Theme file format (v1: hand-authored)

**Authority split (resolves §7-vs-§9 tension):** the theme provides only the
**coherence *vocabulary*** — labels, entity pools, per-subtask tool *bindings*, objective
frames, result templates. The **topology (counts and shape) is ALWAYS driven by the §8
Distributions**, never by the theme. So `subtask_tree`'s `spawns` / `spawns_over` / `calls`
are **label/tool-binding pools**, not topology drivers: when a round spawns (per
`fanout_probability`) `sub_agents_per_spawn` children, the walk *draws* that many labels
from `spawns` (enumerating via `spawns_over`, §9); across the `tool_turns_per_loop` tool
turns (each with `parallel_tool_calls_per_turn` calls) it draws tool bindings from `calls`.
The theme never dictates *how many*. (This is
why §9 can say "never relies on the theme *containing* N named subtasks.")

~3–6 hand-authored JSON/YAML files in `assets/` (or a configured path). Shape:

```json
{
  "domain": "db2_latency_incident",
  "verbs": ["Analyze","Assess","Review","Investigate","Diagnose","Inspect","Evaluate"],
  "entities": {
    "db_instance": ["DBP1","DBP2","BPCS_PROD"],
    "symptom": ["commit-latency spike","lock-wait escalation"],
    "bufferpool": {"enumerate": "BP{i}", "range": [0, 49]}
  },
  "subtask_tree": {
    "root": {"label": "the {db_instance} latency incident",
             "spawns": ["logging_irlm","bufferpools","reorg"]},
    "logging_irlm": {"label": "Db2 logging and IRLM lock behavior",
                     "calls": ["query_irlm_waits"]},
    "bufferpools": {"label": "buffer-pool hit ratios",
                    "spawns_over": "bufferpool", "calls": ["get_bp_stats"]}
  },
  "system_prompt": "You are an autonomous site-reliability agent operating on IBM Db2 for z/OS subsystems. You have a set of diagnostic tools; call them to gather evidence before drawing conclusions. Report findings concisely and cite the subsystem and time window you examined.",
  "objective_template": "{verb} {label} on {db_instance} for the {symptom} that began at {spike_time}.",
  "followup_templates": [
    "check {label2} on {db_instance} for the same incident.",
    "what is the impact on {app} if we {remediation}?",
    "summarize the root cause so far."
  ],
  "followup_connectives": ["Following up on that, ", "Given the result, ", "Next, "],
  "tools": {
    "query_irlm_waits": {
      "spec": {"type":"function","function":{
        "name":"query_irlm_waits",
        "description":"Return IRLM lock/latch wait samples for a Db2 subsystem in a time window.",
        "parameters":{"type":"object","properties":{
          "subsystem":{"type":"string"},
          "wait_class":{"type":"string"},
          "window":{"type":"string"}}, "required":["subsystem"]}}},
      "arg_bindings": {"subsystem":"$db_instance","wait_class":"lock_latch"},
      "artifact_kind": "metrics",
      "result_tokens": {"type": "lognormal", "min": 80, "mean": 400, "std_dev": 600, "max": 4000},
      "result_template": "| time | irlm_lock_waits | avg_wait_ms |\n| {t0} | {n0} | {ms0} |\n| {spike_t} | {n_spike} | {ms_spike} |"
    },
    "fetch_log_bundle": {
      "spec": {"type":"function","function":{
        "name":"fetch_log_bundle",
        "description":"Return the active-log write component logs for a Db2 subsystem.",
        "parameters":{"type":"object","properties":{
          "subsystem":{"type":"string"},"window":{"type":"string"}},"required":["subsystem"]}}},
      "arg_bindings": {"subsystem":"$db_instance"},
      "artifact_kind": "logs",
      "result_tokens": {"type": "lognormal", "min": 200, "mean": 3000, "std_dev": 5000, "max": 24000}
    }
  }
}
```

`artifact_kind` selects the shape generator (`code`/`logs`/`metrics`/`json`);
`result_tokens` is the per-tool size distribution. `metrics` uses the row
`result_template`; `logs`/`code` use the shape generator directly. Interior text is
filler; only shape + size are meaningful.

Two authoring notes:
- **`result_tokens` lognormal shape is moment-matched from `mean` + `std_dev`**
  (`utils/numeric/distribution/utils.py:150–163`); `min` shifts the floor, `max` clips. Set `std_dev` explicitly
  and large enough to reach `max` for a real long tail — omitting it inherits the generic
  default `std_dev=200` (`common.py:35`), which collapses a `{mean:3000,max:24000}` spec
  to a near-constant spike (no tail).
- **Tool `spec` is authored in nested OpenAI shape** (`spec.function.name`), but the
  replay forcing lookup reads a **top-level `name`** (`:367`, inv #2/C1). The generator
  MUST copy `function.name` up to a top-level `name` when emitting `GraphCall.tool_definitions`,
  or single-call turns silently degrade to `tool_choice="required"`. The server accepts
  both shapes, so the miss is silent — hence this build-time flatten is mandatory.
- **`followup_templates` / `followup_connectives`** supply interactive round-2..N user text
  (§4). Both are optional: no `followup_templates` → a generic "continue" prompt; no
  `followup_connectives` → a built-in default connective list. Only relevant when
  `rounds_per_session` can exceed 1.
- **`system_prompt`** (optional string) is the theme's agent preamble, emitted as the shared
  `{role:"system"}` head of every agent's first call (§4.2/§6). It MUST be **session-invariant**
  — no `{entity}`/`{time}` interpolation (those go in the objective) — so it stays byte-identical
  across sessions and caches. If shorter than `shared_system_prompt_len`, it is padded to length
  with **fixed theme-level system boilerplate** (repeated policy/formatting/guidance lines),
  **not** Shakespeare and **not** the agent's sampled tool catalog (which would vary per session
  and break invariance — §4.2 correction). If omitted, a built-in generic agent system prompt is
  used. Set `shared_system_prompt_len` to `0` to suppress the system message entirely (the bare
  non-agentic baseline of §4.2).

## 8. Configuration knobs (`SyntheticAgentSessionsConfig`)

**`SyntheticAgentSessionsConfig` MUST subclass `SessionReplayConfig`** (like
`OTelTraceReplayConfig`) and be passed as `replay_config=` into
`ReplayGraphSessionGeneratorBase.__init__` (§2.2). Otherwise the base runtime's read of
`max_wait_ms` silently no-ops (or `AttributeError`s). Of the base's other replay knobs,
synthetic uses only `max_wait_ms`; `inject_random_session_id` and `duplicate_sessions_target`
are **pinned inert and not exposed** (`False` / `None` — §2.2a/§6/C6), and
`override_tool_call_max_tokens` is likewise pinned `False` (C3).

**Knob summary**

| Name | Type | Default | Scope | One-line meaning |
|---|---|---|---|---|
| `num_sessions` | int | REQUIRED | session | How many sessions = the load volume. |
| `rounds_per_session` | Distribution | REQUIRED | round | N principal inputs to the root; N=1 autonomous, N≥2 interactive. |
| `fanout_probability` | float | REQUIRED | round | P(a single agent execution spawns sub-agents), evaluated per execution. |
| `theme_mix` | dict[str, float] | REQUIRED | session | Theme name(s) → weight(s); the workload's domain/content identity. |
| `tool_turns_per_loop` | Optional[Distribution] | fixed 2 | agent-loop | Number of tool-call turns (loop iterations) in a tool-loop (0 allowed). |
| `sub_agents_per_spawn` | Optional[Distribution] | uniform 2–4 | agent-loop | K children drawn when an execution spawns. |
| `max_depth` | int | 2 | session | Hard recursion terminator (depth 0 = root); spawn attempted only when depth < max_depth. |
| `max_events_per_session` | int | 64 | session | The single hard size cap; the walk self-limits to this event budget. |
| `tool_definitions_per_agent` | Optional[Distribution] | fixed 8 | agent-loop | Advertised schema count on every turn (excess over called = catalog inflation). |
| `parallel_tool_calls_per_turn` | Optional[Distribution] | fixed 1 | turn | Number of tool_calls recorded within a single ordinary tool-loop turn (per-turn width). |
| `seed` | int | 42 | session | Base seed for stable per-session RNG derivation (§2.3a). Fixed default (NOT `load.base_seed`, which time-defaults and would break reproducibility); own field, not inherited (`ConversationReplayConfig` naming). |
| `shared_system_prompt_len` | int (tokens) | — | session | Length of the invariant `{role:"system"}` head on every agent's first call; also the cross-session KV-cache-sharing lever (§6). |
| `input_tokens_per_turn` | Distribution | — | turn | Per-turn input token count. |
| `output_tokens_per_turn` | Distribution | — | turn | Per-turn output token count for plain-text turns. |
| `tool_call_latency_sec` | Distribution | — | turn | Machine/agent `wait_ms` gaps: tool execution, sub-agent dispatch, round-1 startup. |
| `user_think_time_sec` | Optional[Distribution] | tool_call_latency_sec | round | Human read-and-type gap before rounds 2..N of an interactive session. |
| `max_model_len` | int | — | session | Fail-fast ceiling: validator raises if system prompt + max input turn would exceed it. |
| `max_wait_ms` | inherited | 15000ms | inherited | Replay cap on every event's `wait_ms`. |
| `duplicate_sessions_target` | inherited | not supported (pinned None) | inherited | Disabled in synthetic mode (§2.2a); num_sessions sets the count. |
| `inject_random_session_id` | inherited | not supported (pinned False) | inherited | Disabled in synthetic mode — would break KV-cache sharing (§6) and determinism. |
| `bad_tool_call_handling` | enum | none | inherited | How a malformed live tool call is handled; `none` = surface the real failure (promoted to SessionReplayConfig, inherited). |

The detailed per-knob prose below is the authority; the table above is a summary.

**Knob scope legend** (the level each knob varies at, for reading the list below):
**session** (whole run — `num_sessions`, `theme_mix`, `seed`, `shared_system_prompt_len`) ·
**round** (one root principal-input cycle — `rounds_per_session`, `fanout_probability`) ·
**agent/loop** (a root round or a sub-agent dispatch — `tool_turns_per_loop`,
`sub_agents_per_spawn`, `tool_definitions_per_agent`) · **turn** (one LLM call —
`parallel_tool_calls_per_turn`, `input_tokens_per_turn`, `output_tokens_per_turn`). The
names do **not** encode the scope; use this legend.

### Inherited from `SessionReplayConfig`
`max_wait_ms` (used as documented). **`inject_random_session_id` and
`duplicate_sessions_target` are pinned inert and NOT exposed** (`False` / `None`): synthetic
mode does not support them — `num_sessions` already sets the session count directly and each
session is distinct + reproducible, so corpus-amplification via duplication is meaningless
here, and session-id injection would deliberately invalidate the cross-session KV-cache
sharing (§6) and determinism (§2.3a) that are core synthetic signals. Full policy in
§2.2a/§6/C6. **`override_tool_call_max_tokens` is pinned to `False` and NOT exposed** — the base
`*4/4096` override (`:350–356`) is bypassed so the per-event `max_tokens` the generator stamps
(from `output_tokens_per_turn` for plain-text, or `tokens(JSON) + TOOL_CALL_MARGIN` for tool
turns) passes through verbatim (C3). `bad_tool_call_handling` is **promoted to
`SessionReplayConfig`** (a small upstream move from `OTelTraceReplayConfig`, where it currently
lives at `replay.py:181`) so both OTel and synthetic **inherit one definition**; this removes
the base's `getattr(self.replay_config, 'bad_tool_call_handling', NONE)` fallback hack
(`:1593` on upstream/main) — the field is now a real inherited attribute, not a workaround.
Note it as a wiring/upstream touch point (§10). **Default = `none`** — matching the OTel default (`replay.py:182`) and
the "don't hide failures" principle: with `none`, a live model that emits malformed tool-call
args produces the real downstream failure, which is a benchmark signal about the model/parser.
`use_recorded` (substitute our clean recorded assistant message at the affected slot) is an
opt-in **resilience** mode — useful for experiments that want the session to survive a
malformed live call, but it **masks** model/parser defects, so it must not be the default.

### Field shapes mirrored from `ConversationReplayConfig` (naming only; NOT inherited —
`ConversationReplayDataGenerator` is a different, non-graph runtime)
`seed` (**fixed default 42**, our own field — `SessionReplayConfig` has no `seed`; do NOT reuse
`load.base_seed`, which defaults to wall-clock time and would make an unset config non-
reproducible, violating §2.3a; it also lives on `LoadConfig`, unreachable from `_build_session`,
and seeds load *timing*, not graph *content* — a distinct concern), `shared_system_prompt_len`
(scalar int tokens — emitted as a dedicated
`{role:"system"}` shared message at the head of **every** agent's first call, root and
sub-agent, §4.2/§6), `input_tokens_per_turn`
(Distribution), `output_tokens_per_turn` (Distribution). `tool_call_latency_sec`
(Distribution) and the new **`user_think_time_sec`** (Distribution, optional) both
populate `GraphEvent.wait_ms` (new mapping — in `conversation_replay` `tool_call_latency_sec`
is an `asyncio.sleep`; here it is the per-event `asyncio.sleep` at
`replay_graph_session_datagen.py:470–473`, applied *after* the event has already blocked
on all predecessors at `:459–461`). See the timing rules directly below for **which**
distribution feeds **which** event kind, and which events are pinned to 0.

#### Timing: how `wait_ms` is assigned per event
Unlike the OTel path — which *measures* `wait_ms = max(0, t_start − last_pred_end)`
from recorded timestamps (`otel_trace_to_replay_graph.py:1089–1095`) — the synthetic
path has no timestamps, so it **samples** the gap. The sample models real wall-clock
that is *not* network: tool execution, agent thinking, a permission gate, user typing.
Two distinct distributions feed it — because the inter-arrival spacing between a
human's successive turns is a different benchmark signal from a tool's execution gap
(it shifts request concurrency and inter-round KV-cache eviction), they must be settable
independently:
- **`tool_call_latency_sec`** — machine/agent gaps: tool execution, sub-agent dispatch,
  the first round's startup.
- **`user_think_time_sec`** (new, optional; defaults to `tool_call_latency_sec` if unset)
  — the human read-and-type gap **before rounds 2..N** of an interactive (N>1) session.
  Ignored when N=1 (no second user turn exists). This is the knob that controls "how long
  the user takes to answer."

Assignment is **by event kind**, not uniform:
- **Root principal turn, round 1** (session's first user message) → one sample from
  `tool_call_latency_sec` (startup gap). Roots with no predecessor still get their sample;
  the OTel "root ⇒ 0" convention does not bind us because there is nothing to measure from.
- **Root principal turn, rounds 2..N** (each subsequent user message, depends on the prior
  round's answer) → one sample from **`user_think_time_sec`** (the user's answer latency).
- **Tool-loop continuation** (turn that follows a tool result) → one sample from
  `tool_call_latency_sec` (tool-exec gap).
- **Sub-agent dispatch** (each of the K single-call dispatch events) → one sample from
  `tool_call_latency_sec`.
- **Fan-in merge** (the parent continuation event awaiting the K child terminals) → **pinned
  to `wait_ms = 0`**. The runtime already blocks on the slowest child via `require_async`
  (`:459–461`); the real wall-clock gap *is* that block. Adding a sampled `wait_ms` on top
  would stack an extra artificial delay that no benchmark signal justifies and that would
  inflate merge latency. This is the one place the value is forced, not drawn.

Every assigned value is still capped at replay by `max_wait_ms` (`min(event.wait_ms,
max_wait_ms)`, `:1320`; default 15000ms, `replay.py:129`), so the sampled distribution
should sit under that ceiling or the tail is silently clipped. `t_start_ms`/`t_end_ms`
remain cosmetic on the synthetic path (DOT labels only) — only `wait_ms` drives replay sleep.

`max_model_len` (int): **fail-fast** —
a `@model_validator` raises if `shared_system_prompt_len` + the max of
`input_tokens_per_turn` would exceed it (the graph runtime does not enforce it, so we
validate at config time rather than silently overrun).

### New agentic — REQUIRED (no default; the config author MUST set these)
These four are *decisions*, not tuning — a silent default would misrepresent the workload,
so a new config does not validate until they are set. (`pydantic` `Field(...)`.)
- `num_sessions: int` — how many sessions = the load volume. No universal right value.
- `rounds_per_session: Distribution` — N principal inputs to the **root** (§4.2). N=1 ⇒
  **autonomous** (headline; exgentic); N≥2 ⇒ interactive. The fundamental shape decision.
- `fanout_probability: float` — P(a single **agent execution** spawns sub-agents), evaluated
  **per execution**: once for **every root round** AND once for **every sub-agent** (at every
  depth). This is what makes recursion expressible — a sub-agent is single-dispatch (no
  rounds), so the probability must be scoped to an *execution*, not a round, for a sub-agent
  to be able to spawn. One probability for both levels is sufficient for v1 (separate
  root/sub-agent probabilities → §12). `0` ⇒ single-agent; >0 ⇒ the headline multi-agent
  capability. **A spawn is only *attempted* when the current depth `< max_depth`** — at the
  cap the probability is forced to 0 (see `max_depth`), which is what bounds the recursion.
  **Note the branching factor:** expected children per node ≈ `fanout_probability ×
  mean(sub_agents_per_spawn)`. If that product is ≥ 1 the tree would grow explosively *were it
  not for the depth cap* — `max_depth` is the hard terminator; `fanout_probability` only tunes
  how densely the depth-bounded tree fills in. (E.g. `p=0.5`, K≈3 ⇒ 1.5 children/node — with
  `max_depth=2` the worst case is `1+3+9=13` agents, not unbounded.)
- `theme_mix: dict[str, float]` — theme name(s) → weight(s). The workload's domain/content
  identity; must be named (no mystery built-in default). **Assignment:** weights are
  normalized to probabilities; each session draws its theme by a **weighted draw from its
  session-local RNG** (§2.3a), so assignment is deterministic per `(config, session_index)`.
  The prefix-sharing **cohort** (§6) is the set of sessions that drew the same theme.

### New agentic — DEFAULTED (tuning within the chosen shape; count-like knobs are
`Optional[Distribution]=None` with a documented per-field fallback — never the generic
`Distribution` `mean=512`)
- `tool_turns_per_loop: Optional[Distribution] = None` → **fallback fixed 2** — the number of
  tool-call **TURNS** (loop iterations) in a tool-loop, NOT the number of tool calls (`0`
  allowed = answer immediately, no tool turn). **Each iteration is exactly one assistant
  tool-call turn**; how many calls are recorded *within* that turn is set separately by
  `parallel_tool_calls_per_turn`. So `tool_turns_per_loop=4` with
  `parallel_tool_calls_per_turn=3` = **4 turns, 12 total calls** (each turn: 1 assistant turn
  with 3 tool_calls → 3 `role:"tool"` results), followed by the loop's terminating `[answer]`
  turn. The two knobs are orthogonal axes: this one = loop **depth** (how many round-trips);
  the other = per-turn **width** (calls per round-trip). Governs **both root rounds and
  spawned sub-agents** (sampled per loop, at every depth) — one knob, so the server sees a
  single aggregate distribution. (A separate root-vs-sub-agent split is deferred; the same
  aggregate load is reachable by widening this one distribution — see §12.)
- `sub_agents_per_spawn: Optional[Distribution] = None` → **fallback uniform 2–4** — K
  children drawn when an execution spawns (the "50 sub-agents" knob).
- `max_depth: int = 2` — **the hard recursion terminator.** Depth 0 = root, depth 1 = its
  sub-agents, … A spawn is attempted only when `depth < max_depth`; at `depth == max_depth`
  the effective spawn probability is **0**, so a depth-`max_depth` sub-agent is always
  single-dispatch (tool-loop, no children). This — not `fanout_probability` — is what
  guarantees every session graph is finite and reproducible; the tree has a fixed ceiling of
  `1 + K + K² + … + K^max_depth` nodes (reached only if every eligible node spawns). `0` ⇒ no
  fan-out ever (equivalent to `fanout_probability=0`). Keep it small (2–3) for realism and to
  keep the node ceiling sane.
- `max_events_per_session: int = 64` — **the single hard size cap: a self-limiting event
  budget for the walk.** The default is deliberately **conservative** — 64 events is a
  single-agent tool-loop or a small fan-out; **large fan-out configs must raise it explicitly**
  (see §13.3, which sets 2048), otherwise their trees truncate at the frontier (below). This is
  intentional: the safe default keeps an unconsidered config bounded and cheap, and opting into
  a big tree is a conscious choice. `max_depth` guarantees recursion *terminates*, but not a
  practical *size* — the tree grows exponentially with fan-out width (`max_depth=2`, 50
  children/spawn, 10 rounds → ≈25,501 agents; `max_depth=3` → ≈1.27M). Rather than estimate a
  worst case and reject the config, the walk **bounds itself as it builds** (this is a LAZY
  generator, §2.3a — a mid-run crash would discard a long benchmark, so we never raise from
  `_build_session`):
  - `_build_session(N)` tracks a **session-local event counter**, starting at 0. It counts
    **events** (what the runtime schedules and what bounds memory), not agents — agents are
    derivable and events are the true cost.
  - **Atomic at each *decision* (the rule that keeps every graph valid):** the budget is
    checked before committing to any sub-structure, never mid-way:
    - **Before a spawn** — if the whole spawn (K dispatch/merge events + the minimum cost of K
      single-dispatch children) won't fit the remaining budget, the node spawns **no** children
      and becomes a plain leaf (its tool-loop + answer). This is the **same no-spawn shape** a
      `fanout_probability` "no" roll produces — an already-valid, already-tested configuration.
    - **Before starting a new root round** (interactive N>1) — if the next round's minimum cost
      won't fit, the walk **stops starting rounds**: the current round finishes cleanly and the
      session ends early with fewer than its sampled N rounds. Same principle — never truncate
      mid-round, only decline to start the next one.
    Never a half-built fan-out or half-built round: you never emit a `dispatch_agent` call whose
    child event won't exist, nor a round without its answer, so no dangling `tool_call_id` (inv
    #3 holds by construction).
  - **Result:** every built graph is `≤ max_events_per_session`, structurally valid, and
    deterministic — the cut lands at the identical node in parent and worker because the counter
    is session-local and the pre-order walk order is fixed (§2.3a). No worst-case arithmetic, no
    config rejection, no skipped session, no mid-run raise.
  - **Truncation is visible, not silent:** if any spawn OR round was budget-blocked in a
    session, log once per session at `info` ("session N: truncated at event budget"); the
    generator also tracks a run-level truncated-session count. This is the signal that a cap is
    clipping the fan-out or rounds the knobs asked for.
  - **Shape caveat (honest):** truncation is **frontier-based, not uniform** — nodes the fixed
    walk reaches *first* get their full fan-out; nodes reached *last* get clipped to leaves. So
    a truncated session is slightly lopsided (early branches bushy, late branches lean). It is
    still a well-formed tree from the same family and faithful for token/KV/concurrency load; it
    is just not a uniformly-shrunk version of the untruncated graph. Set the budget high enough
    that typical sessions don't truncate; use truncation as a hard ceiling, not a shaping tool.
- `tool_definitions_per_agent: Optional[Distribution] = None` → **fallback fixed 8** —
  advertised schema count **on every turn** (the "1000 tool defs" knob; excess beyond
  *called* tools = catalog inflation). The catalog rides on **every** turn, including k=0
  direct-answer turns (§4.2) — matching real agent harnesses, which attach tools before the
  model chooses whether to call one. Set to **0** for the bare non-agentic / plain-chat
  baseline (no tools advertised at all).
- `parallel_tool_calls_per_turn: Optional[Distribution] = None` → **fallback fixed 1**
  (single call per turn — guaranteed clean). The number of tool_calls recorded **within a
  single ordinary tool-loop turn** (per-turn *width*, orthogonal to `tool_turns_per_loop`'s
  *depth*). **Default is 1, deliberately, so the default generator never knowingly produces a
  failure:** a `K>1` turn forces `tool_choice="required"`, which can dangle → server 400 when
  the live model emits fewer than K calls (C2). K=1 forces the exact function and never dangles.
  The **real-shaped** distribution (heavy at 1, ~8–12% shoulder at ≥2, light tail — §2.5) is a
  genuine fidelity target but **opt-in**: set the knob to it (see the example config in §13.5).
  **Applies only to ordinary tool-loop turns, NOT dispatch/spawn turns** — a sub-agent
  dispatch is always single-call (C2); fan-out width is `sub_agents_per_spawn` across separate
  events, a different axis. When the opt-in distribution is used, K>1 turns are replayed
  **best-effort**: mismatch sessions are dropped (C2), and the mismatch rate is itself a signal.
- **Tool-call `max_tokens`: not a knob** — forced tool-call turns derive their cap directly
  from the JSON they will emit (`tokens(expected JSON) + TOOL_CALL_MARGIN`), not from a config
  surface; see C3.
- **Filler source: not a knob.** Free-text filler (§6) is **always** the bundled
  Shakespeare prose corpus (`inference_perf/assets/shakespeare.txt`) padded to budget
  and capped with the `[--- ignore the preceding filler; actual content follows ---]`
  marker — not a knob, because the choice of filler source changes **nothing a benchmark
  measures** (token volume, KV footprint, prefill cost, and prefix-sharing are all
  identical; coherence comes from the seam, not the filler). Prose is strictly the safer
  default (predictable tokenization; random tokens can emit rare/unsplittable tokens that
  skew length↔count convergence). Random-token filler stays available as an internal
  constant for deliberate tokenizer-stress experiments, not a first-class config field.
- **Truncation-likelihood advisory (WARNING, does not raise).** No worst-case validator is
  needed — `max_events_per_session` bounds the graph *as it builds* (above), so no config can
  explode regardless of its knobs. But a config likely to truncate is worth flagging so the
  realized workload doesn't silently diverge from the requested one. A `@model_validator`
  logs a **warning** when the *expected* branching `b = fanout_probability ×
  mean(sub_agents_per_spawn)` is `≥ 1` **and** the rough expected event count at that branching
  approaches `max_events_per_session` — i.e. "with these knobs, sessions will likely hit the
  event budget and truncate; raise `max_events_per_session` or lower fan-out." It never raises:
  a dense tree that truncates is a legitimate (bounded, valid) workload; the warning plus the
  per-session truncation log (above) make it visible.

Argument encoding: tool-call `arguments` are emitted as a **`json.dumps`-ed string**
(inv #1); real source args are JSON objects. Real tool results are additionally JSON-
escaped (single/double); an optional `envelope_encoding` on `artifact_kind` to reproduce
that token inflation is deferred (§12).

## 9. Extreme-knob behavior (layered)
- **1000 tool defs:** the few *called* tools stay theme-coherent; the remaining
  ~988 are procedurally-generated valid schemas (tool-catalog inflation), which is
  exactly how real traces reach the ~22K prompt mode.
- **50+ sub-agents:** entity **enumeration** (`BP0..BP49`) supplies distinct
  coherent objectives; degrades gracefully to index-suffixed entities if the pool
  is smaller than K. Never relies on the theme *containing* N named subtasks.
- **10k agents:** lazy `_build_session`, O(1) text *per node*, per-call token
  budgets, per-session eviction after completion. Caveat: "O(1) per node" bounds
  per-node work, not resident node count — each in-flight session materializes its
  **full** graph (all N events with messages) in both the parent and the owning
  worker until eviction. So a single 10k-agent session = one full graph resident;
  eviction bounds the *concurrent* working set across sessions, not one session's peak.

## 10. Wiring (coordinated edits — no registry in this codebase)
Concrete touch points (verified file:line; line numbers approximate to current tree):
1. `DataGenType.SyntheticAgentSessions = "synthetic_agent_sessions"` (`config/datagen/config.py`).
2. New `SyntheticAgentSessionsConfig(SessionReplayConfig)` model in `config/datagen/replay.py`,
   exported from **both** `config/datagen/__init__.py` **and** top-level
   `config/__init__.py` (real consumers import via the top-level surface).
3. `synthetic_agent_sessions: Optional[SyntheticAgentSessionsConfig]` field on `DataConfig`.
4. Add `DataGenType.SyntheticAgentSessions` to the existing validator tuple in
   `validate_trace_replay_load_type` (`config/config.py:52`) — it already enforces
   `LoadType.TRACE_SESSION_REPLAY`; this is a one-line tuple extension, not a new validator.
5. `SyntheticAgentSessionsDataGenerator` in `datagen/`, exported from `datagen/__init__.py`,
   and **imported in `main.py`** (`main.py:34–45`) or dispatch raises `NameError`.
6. `main.py`: add `SyntheticAgentSessions` to the tokenizer-required set (`~:295`), a presence
   guard, and a dispatch `elif` (`~:365`).
7. `main.py`: extend the `(OTelTraceReplay, WekaTraceReplay)` membership tuples for the
   `mp.Manager` branch (`~:278`, note the `num_workers>0` guard) and the
   `SessionMetricsCollector` branch (`~:380`). Reportgen wiring (`~:389`) needs no edit
   (flows automatically for session generators).
8. **Runtime extension (fan-out only, §4.1a):** add `"tool_output"` to the
   `InputSegment.type` `Literal` (`replay_graph_types.py:61`) and its docstring (`:46–61`);
   add a `tool_output` branch in `_build_messages_with_substitution`
   (`replay_graph_session_datagen.py`, alongside the `output` branch at `:532`) and include
   `seg.type == "tool_output"` in the `needs_substitution` predicate (`:476`). This is the
   sole runtime change; a config that never sets `fanout_probability > 0` does not exercise
   it. Ship it behind the same test in §11 so the OTel/Weka paths are provably untouched.
9. **Promote `bad_tool_call_handling` from `OTelTraceReplayConfig` to `SessionReplayConfig`
   (base)** so it is inherited by both OTel and synthetic configs; remove the
   `getattr(self.replay_config, 'bad_tool_call_handling', NONE)` fallback in the runtime now
   that the attribute always exists. Default stays `none`.

## 11. Testing
- Unit: theme rendering (verbatim objective identity parent↔child), objective
  template Mad-Libs, entity enumeration past pool size, artifact-template output.
- **Graph validity (new validator, NOT `build_graph`):** the generator emits a
  `ReplayGraph` directly, so validate the emitted object with a purpose-built
  checker (build_graph is offline-only, §2.4). Assert: every event has non-empty
  `call.messages` (inv #4); each
  `tool_definitions` entry has a **top-level `name`** (inv #2, C1); `#role:tool ==
  #tool_calls` in order per turn (inv #3); `tool_call_id` linkage present; args are
  `json.loads`-able strings (inv #1); `expected_output_tool_names` /
  `expected_output_is_tool_call` set on tool turns; the **merge** (continuation) event lists
  all K child terminal events in `predecessor_event_ids` and, per child, carries an
  `[output(source = the dispatch event) , tool_output(source = the child terminal)]` pair —
  the `tool_output` slot is `role:"tool"` with a `tool_call_id` (§4.1/§4.1a); every
  substitution `source_event_id` is also an ancestor in `predecessor_event_ids`;
  `predecessor_dependency_types` populated per edge with
  **valid `DEPENDENCY_TYPE` values** (`full_match`/`tool_call_ids_matched`/`temporal` —
  NOT `output`, §4.1); no event references a skipped/empty predecessor.
- **`tool_output` segment (new runtime primitive, §4.1a):** unit-test the substitution branch
  directly — a recorded `{role:"tool", tool_call_id, content}` slot with a `tool_output`
  segment sourcing a completed child event yields a message that **keeps `role:"tool"` and its
  `tool_call_id`** and replaces only `content` with the child's live text (NOT the child's
  `{role:"assistant"}` message — the bug an `output` segment would cause). Assert the guard
  fires (falls back to recorded) when the recorded slot is not `role:"tool"` or
  `message_count != 1`, and when the child output is unavailable. **Regression guard:** an
  OTel/Weka replay graph containing no `tool_output` segment produces byte-identical
  substitution output before and after the runtime change (the branch is inert unless used).
- Token fidelity: **plain-text** per-turn counts converge to `output_tokens_per_turn`; the
  ~22K mode is reachable via tool-catalog inflation.
- **Tool-call max_tokens policy (C3):** assert a forced tool-call turn's emitted
  `expected_output_tokens` = `tokens(expected JSON) + TOOL_CALL_MARGIN` (and the multi-call
  case sums the K calls' JSON), so the forced JSON never truncates; plain-text turns use
  `output_tokens_per_turn` unchanged. Assert no runtime edit is needed: the stamped
  `expected_output_tokens` reaches the request `max_tokens` verbatim
  (`override_tool_call_max_tokens` pinned False, so the `*4/4096` block is skipped).
- **Purity / cross-process determinism (§2.3a) — REQUIRED:** the serialized graph and
  per-session event order for `_build_session(N)` must be **byte-identical** when built
  (i) in a different session-build **order**, (ii) in **independent generator instances**,
  (iii) in **separate spawned processes** (`multiprocessing.spawn`, matching the worker
  path), and (iv) under **different `PYTHONHASHSEED` values** (run the test with
  `PYTHONHASHSEED=0` and `=1`, or two random values, and compare). This is the test that
  catches a stray `hash(...)`, a shared generator-level RNG, or any hidden build-order
  dependence. Compare a stable serialization (event IDs + insertion order + segments +
  messages + tool catalogs), not object identity.
- Determinism: same seed → byte-identical graph.
- **Prefix sharing (§6) — measured:** tokenize two different sessions' **emitted first-call
  payloads** (same theme cohort) and assert the **longest common contiguous token prefix from
  position 0** covers the whole invariant head — the identical `tools` payload + the
  `{role:"system"}` message (length `shared_system_prompt_len`) — and diverges only once
  per-session content (objective/entities) begins. Assert a larger `shared_system_prompt_len`
  → a proportionally longer shared prefix. Assert the shared head holds only **with
  `inject_random_session_id=false`** and non-duplicated sessions (C6). Assert `shared_system_prompt_len=0`
  and a per-agent-varying `tool_definitions_per_agent` → the shared prefix shrinks toward zero
  (nothing invariant to cache). This catches a regression where per-session content leaks into
  the head (which would show a common prefix shorter than the invariant region).
- **Duplication / session-id injection disabled (§2.2a):** setting `duplicate_sessions_target`
  or `inject_random_session_id` has no effect in synthetic mode (rejected-or-pinned-inert) — no
  `_dup{n}` slots are produced and no `[SESS:]` string is injected into `unique` segments.
- **Recursion bounds (§8):** `fanout_probability` fires per execution — assert a **sub-agent
  can itself spawn** (a depth-2 node exists when `max_depth ≥ 2` and `p` high). Assert **no**
  agent exists at depth `> max_depth` for any seed. Assert `max_depth=0` and
  `fanout_probability=0` both yield single-agent sessions.
- **Self-limiting event budget (§8 `max_events_per_session`):** a would-be-explosive config
  (`max_depth=3`, `sub_agents_per_spawn` large, high `p`) **builds successfully** with **every
  session ≤ `max_events_per_session`** — no raise, no skipped session. Assert every truncated
  graph is **structurally valid**: no `dispatch_agent` call lacks its child terminal event
  (inv #3 holds), every event has non-empty messages (inv #4) — i.e. truncation only ever
  turns a would-be-spawning node into a no-spawn leaf, never a half-built fan-out. Assert the
  cut is **deterministic** (same seed → identical truncated graph; and byte-identical across
  processes per §2.3a). Assert a per-session `info` log fires exactly when truncation occurred
  and the run-level truncated-session count matches. Assert a config sized so sessions fit
  never truncates.
- **Filler fitting (§3 Layer 3):** the **best-candidate wrapper** returns a usable
  `(text, ids)` + `{target_tokens, actual_tokens, exact}` even when exact convergence fails —
  assert a forced non-convergent case yields the closest-length text (NOT a raised
  `ValueError`, NOT an empty string) and `exact=False`. Assert the **filler-budget guard**: a
  turn whose target is below `tokens(objective + marker + required_structure)` (e.g. a 5–10
  token output minimum vs the ~15-token marker) raises the target to the fixed-content floor
  and logs it, rather than spinning to a `ValueError`; and the marker is omitted when
  `filler_budget ≤ 0` (no dangling delimiter). Assert `converge_to_exact_length_text` itself
  still raises on genuine tokenizer mismatch (its contract is unchanged; the wrapper is what
  the generator calls).
- **Shape coverage (§2.5/§4.2):** assert the round model emits each shape via its knobs —
  N=1,k=0 with `tool_definitions_per_agent=0` (bare plain answer, no tools advertised),
  N=1,k=0 with `tool_definitions_per_agent>0` (tools-advertised-no-call: catalog present,
  replay does NOT force `tool_choice`), N=1,k≥1 (single tool-loop, ends with an answer turn),
  N≥2,k=0 (plain conversation), and **N≥2 with per-round k varying (chat-with-tools)** —
  and that a spawning round produces sub-agents that are single-dispatch. Tool results
  are always `{role:"tool", content, tool_call_id}` (§4.2).
- **Safe defaults (this review):** with no overrides, `parallel_tool_calls_per_turn`
  resolves to **fixed 1** — assert every ordinary tool turn in a default-config graph has
  exactly one `tool_call` (so no K>1 turn, no dangling-id risk); and `bad_tool_call_handling`
  defaults to **`none`** (assert the field's default, matching OTel). Assert the opt-in
  real-shaped `parallel_tool_calls_per_turn` (the §13.5 example config) does produce K>1 turns.
- **Timing assignment (§8 "Timing"):** every fan-in merge (continuation) event has
  `wait_ms == 0`; tool-loop / dispatch / round-1 turns carry a sampled `wait_ms` matching
  `tool_call_latency_sec`; **rounds 2..N of an N>1 session carry a `wait_ms` matching
  `user_think_time_sec`** (and matching `tool_call_latency_sec` when `user_think_time_sec`
  is unset — the default-to-fallback path); an N=1 session never draws `user_think_time_sec`;
  all assigned values are ≤ `max_wait_ms` (else the tail is clipped at replay, `:1320`).
- Integration: a small generated config replays end-to-end against a mock/local server
  without 400s (exercises single-call forcing + substitution).
- DOT export sanity: `export_replay_graph_to_dot` renders the fan-out/merge with the
  expected roots and edges (note its per-edge `wait_ms` label is read off the successor
  — misleading for multi-predecessor merges, cosmetic only).

## 12. Future work (out of scope for v1, but planned)
- **LLM-authored theme generator:** an offline script that prompts an LLM once per
  domain to emit a theme file **in the exact v1 format**, validated (`json.loads`
  every tool `parameters`, resolve every `spawns`/`calls` reference, confirm every
  objective slot exists in `entities`) and checked in. Pure authoring accelerator —
  no runtime change; the generator consumes hand-authored and LLM-authored themes
  identically. Example theme in §7 is the target output shape.
- Corpus-harvested themes (mine Exgentic for entity vocab, tool catalogs,
  result-body shapes).
- Additional fillers: `domain_jargon_bank`, `self_referential_echo`,
  `markov_ngram` (gated to unique regions).
- Coherent lead sentence on reasoning turns (reuses objective-template machinery).
- **Free tool-choice mode** (`force_tool_choice: false`): let the model choose
  which tool(s) to call, to benchmark real tool-**selection** behavior. Requires
  genuinely coherent goals + tool descriptions + result data (i.e. the
  LLM-authored-content tier) so a real model finds the right tool useful; topology
  becomes non-deterministic. Deferred from v1, which is forced-only (see §4).
- **`thinking` / `reasoning_content` assistant surface** — a distinct reasoning part (real
  data has ~5.9k `thinking` parts). The runtime has **first-class `reasoning_content`
  handling** — it is threaded through `ChatMessage`, accumulated separately in streaming,
  recorded on the output message, and **preserved even on `tool_calls` messages**
  (upstream/main `:1531/:1533`, `:872/:924`), and reasoning tokens are sent to the server for
  KV fidelity. So a synthetic reasoning surface would be *natively* supported (emit
  `reasoning_content` on a turn, distinct from `content`). Still v1-deferred **only** because
  generic filler already carries the token load — NOT because of inv #6 (which governs
  `content`, not `reasoning_content`; reasoning would ride the tool-call turn as
  `reasoning_content`, no separate turn needed). A **clean, low-cost** future add via a
  `thinking_probability` knob that sets `reasoning_content`.
- **`envelope_encoding` on `artifact_kind`** (`none|json_array|double_json`) — reproduce
  the doubly/triply JSON-escaped tool-result strings (§2.5) that inflate result tokens.
- **`artifact_kind: retrieval`** — ranked `{docid, score, snippet}` bodies for RAG
  benchmarks (browsecompplus); `json` approximates it meanwhile. (There is NO `document`
  role — that was a flattened-file artifact, §2.5.)
- **Alternate harness wire conventions** — the non-OpenAI tool-result shapes
  (`claude_code`: result on the `user` role, no id; `smolagents`: `"Observation:"` prose,
  no structured tool parts; §2.5). Not modeled — wire-format mimicry the benchmark does
  not need (§4.2). Only worth revisiting if a consumer specifically needs byte-level
  fidelity to one of those harnesses.
- **`developer` role** option for the rare `tool_calling_with_shortlisting` family.
- **Parallel-call count reconciliation** — trim/pad the successor's `role:tool` messages
  to the live tool-call count so K>1 turns never dangle (removes the best-effort session
  drops of C2). Needs a runtime change to the substitution post-pass (`:723–739`).
- **`sub_agent_tool_turns_per_loop`** — a separate tool-loop distribution for spawned agents,
  to express an asymmetric *shallow orchestrator / deep worker* fan-out directly. Cut from
  v1 (§8): the same aggregate load is reachable by widening the single `tool_turns_per_loop`
  distribution, so the split bought no distinct benchmark signal — only a cleaner authoring
  story. Revisit if a consumer needs to pin root-vs-sub loop depths independently.
- **Optional `shape` preset knob** (§13.0) — a convenience that seeds the orthogonal
  N/`tool_turns_per_loop`/`fanout_probability` knobs for the common shapes, fully overridable
  and composable with fan-out. Net-new pattern (no sibling config uses presets); stated
  defaults + the §13.0 minimal example deliver most of the friendliness without it.

## 13. Example configs (what each simulates)

Illustrative YAML (field names follow §8; `data.type: synthetic_agent_sessions`, `load.type:
trace_session_replay`). These communicate the *intended workloads*, not final schema.

### 13.0 Minimal config — required fields only, everything else defaults (§8)
The four **required** knobs (§8) must be set — they are the workload's shape decisions, so
a config does not validate without them. Everything else defaults (`tool_turns_per_loop≈2`,
`tool_definitions_per_agent≈8`, …; filler is always the Shakespeare corpus). A minimal
runnable config:
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 100                          # required: load volume
    rounds_per_session: {type: fixed, mean: 1} # required: 1 = autonomous
    fanout_probability: 0.0                     # required: 0 = single-agent
    theme_mix: {db2_latency_incident: 1.0}      # required: domain
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 16}]
```
Those four lines force the author to state load / autonomous-vs-interactive /
single-vs-multi-agent / domain explicitly; this exact config is the exgentic single-agent
tool-loop baseline. The examples below override only additional tuning.

*(Optional ergonomic layer, deferred: a `shape` preset — e.g. `autonomous_tool_loop`,
`big_catalog_direct_answer`, `autonomous_fanout`, `multi_round_chat`, `plain_answer` —
that merely SEEDS the orthogonal N/k/fanout knobs, fully overridable and composable with
`fanout_probability`. It is a convenience over the knobs, NOT a closed taxonomy — §4.2
deliberately dropped the closed enum because shapes compose. See §12.)*

### 13.1 Single-agent tool-loop — the exgentic baseline
Simulates the common real shape: one agent, large shared system prompt, a short
forced tool-loop, results on the `tool` role. No fan-out. This is the closest analog
to what `otel_trace_replay` produces from real exgentic traces — a sanity baseline.
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 500
    seed: 42
    fanout_probability: 0.0          # single agent, no sub-agents
    rounds_per_session:  {type: fixed, mean: 1}   # single-shot
    tool_turns_per_loop:  {type: fixed, mean: 2}   # 2 tool-call turns, then an answer
    tool_definitions_per_agent: {type: fixed, mean: 8}
    shared_system_prompt_len: 8000   # big shared prefix (KV-cache reuse)
    input_tokens_per_turn:  {type: lognormal, min: 200, mean: 1500, max: 8000}
    output_tokens_per_turn: {type: lognormal, min: 10,  mean: 120,  max: 800}
    theme_mix: {db2_latency_incident: 1.0}
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 16}]
```

### 13.2 Big tool-catalog, direct-answer — the "22K prompt" mode
Simulates the single most common real shape: many tools **advertised** (huge
`available_tools` block driving a ~22K-token prompt) but the assistant **answers
directly** without calling one. Stresses prefill / long-prompt handling, not tool
execution. The key is a 1-round, k=0 session with a large `tool_definitions_per_agent`: the
catalog is advertised on the direct-answer turn (always, §4.2) but no call is forced.
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 300
    seed: 7
    theme_mix: {ims_command_eval: 1.0}
    fanout_probability: 0.0
    rounds_per_session: {type: fixed, mean: 1}
    tool_turns_per_loop: {type: fixed, mean: 0}    # answer directly, no tool turn
    tool_definitions_per_agent: {type: fixed, mean: 60}   # -> ~22K-token catalog, still advertised on the k=0 turn
    input_tokens_per_turn:  {type: fixed, mean: 22000}
    output_tokens_per_turn: {type: lognormal, min: 5, mean: 40, max: 300}   # tiny answers OK
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 32}]
```
*(A `min` of 5–10 output tokens is below the filler marker's own cost, but answer turns carry
no filler marker, and the §3 filler-budget guard raises any turn's target to its fixed-content
floor rather than failing — so small minima are safe.)*

### 13.3 Autonomous run with recursive fan-out — the headline / synthetic-only capability
Simulates the headline workload no source dataset contains: an **autonomous** agent
(one task, no human follow-ups) that spawns concurrent sub-agents — one recursively
spawns its own, others run tools — merging back into the final answer. Stresses concurrent
in-flight sessions, fan-in substitution, and KV behavior under many parallel branches.
Fan-out is emitted as separate single-call dispatch turns (not K-parallel-in-one-turn;
§2.5/C2). `rounds_per_session: 1` keeps it autonomous; raise it for an interactive run
that spawns fan-outs across several human turns.
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 200
    seed: 1
    rounds_per_session: {type: fixed, mean: 1}   # autonomous: one task, run end-to-end
    fanout_probability: 0.6            # P(EACH execution spawns — root round AND every sub-agent)
    sub_agents_per_spawn: {type: uniform, min: 2, max: 5}   # K, mean 3.5
    max_depth: 3                       # hard recursion cap (terminates recursion)
    max_events_per_session: 2048       # RAISED from the default 64 — this fan-out tree is large
                                       #   (depth 3 × K~3.5 × up to 8 tool turns/agent), so the
                                       #   default would truncate it heavily. Big trees must opt in (§8).
    tool_turns_per_loop:        {type: uniform, min: 0, max: 8}   # root & sub-agents share this (turns, not calls)
    tool_definitions_per_agent: {type: fixed, mean: 12}
    input_tokens_per_turn:  {type: lognormal, min: 300, mean: 2000, max: 12000}
    output_tokens_per_turn: {type: lognormal, min: 20,  mean: 200,  max: 1500}   # plain-text turns
    # (forced tool-call turns size max_tokens from their own JSON automatically — no knob, C3)
    theme_mix: {db2_latency_incident: 0.5, k8s_outage: 0.5}
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 64}]   # many concurrent branches
```

### 13.4 Multi-round chat-with-tools — a human conversation where each turn may use tools
Simulates a real assistant conversation: the human asks, the agent runs a tool-loop and
answers, the human follows up, the agent sometimes answers immediately (k=0) and
sometimes tools again — repeated over several rounds. This is the general single-agent
shape the earlier `leaf_kind` enum could not express.
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 200
    seed: 3
    theme_mix: {db2_latency_incident: 1.0}
    fanout_probability: 0.0                     # single agent, just multi-round
    rounds_per_session: {type: uniform, min: 3, max: 8}   # a real back-and-forth
    tool_turns_per_loop: {type: uniform, min: 0, max: 3}   # some rounds tool, some answer directly
    tool_definitions_per_agent: {type: fixed, mean: 10}
    shared_system_prompt_len: 4000
    input_tokens_per_turn:  {type: lognormal, min: 100, mean: 900, max: 6000}
    output_tokens_per_turn: {type: lognormal, min: 10,  mean: 150, max: 1000}
    tool_call_latency_sec: {type: lognormal, min: 0.2, mean: 1.5, max: 8}   # machine gaps
    user_think_time_sec:   {type: lognormal, min: 2,   mean: 12,  max: 15}  # human read+type before each follow-up
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 32}]
```
*(Set `tool_turns_per_loop` fixed at 0 for a plain no-tools conversation; combine with
`fanout_probability > 0` for a chat where some rounds delegate to sub-agents.
`user_think_time_sec` only matters for N>1 — it spaces out the follow-up turns; omit it
to reuse `tool_call_latency_sec`. Keep its `max` under `max_wait_ms` (default 15000ms)
or the tail is clipped at replay.)*

### 13.5 Opt-in real-shaped parallel calls (best-effort)
The default `parallel_tool_calls_per_turn` is **fixed 1** (never dangles). This example config
sets it to the **real** parallel-call distribution (~8–12% of turns emit ≥2 calls; §2.5) to
benchmark parallel-tool handling faithfully. It is **best-effort**: a K>1 turn 400s if the live model
emits fewer than K calls (C2), so those sessions drop (the mismatch rate is itself a signal
about the model) — do NOT use this as a default/CI workload; use it deliberately when
parallel-call fidelity is the thing under test.
```yaml
data:
  type: synthetic_agent_sessions
  synthetic_agent_sessions:
    num_sessions: 200
    seed: 5
    theme_mix: {db2_latency_incident: 1.0}
    fanout_probability: 0.0
    rounds_per_session: {type: fixed, mean: 1}
    tool_turns_per_loop: {type: uniform, min: 1, max: 4}
    tool_definitions_per_agent: {type: fixed, mean: 12}
    parallel_tool_calls_per_turn: {type: lognormal, min: 1, mean: 1.3, max: 6}  # OPT-IN real shape
    input_tokens_per_turn:  {type: lognormal, min: 200, mean: 1500, max: 8000}
    output_tokens_per_turn: {type: lognormal, min: 20,  mean: 200,  max: 1500}
load:
  type: trace_session_replay
  stages: [{concurrent_sessions: 32}]
```
*(A high count-mismatch drop rate here is a real signal about the model, not a generator bug.
Leave `parallel_tool_calls_per_turn` unset for the safe fixed-1 default.)*

## 14. Viewable graphs (reuse the OTel DOT visualization)

The synthetic generator produces `ReplayGraph` objects — the **same type**
`graph_to_dict` and `visualize_graph` already consume — so visualization comes for
free by reusing the existing path (`export_replay_graph_to_dot.export_to_dot`), no new
rendering code. The synthetic path is in fact simpler than OTel's: there is no trace to
parse; the generator builds the `ReplayGraph`, then dumps/visualizes it directly.

**Proposed dump CLI** — mirror `python -m inference_perf.datagen.otel_trace_to_replay_graph`:
```
python -m inference_perf.datagen.synthetic_agent_sessions \
    --config <synthetic_agent_sessions config.yml>   # reuse the SyntheticAgentSessionsConfig
    --session-index 0                      # which session to materialize (lazy _build_session)
    --output   graph.json                  # graph_to_dict(graph) dumped as JSON
    --vis_output graph.dot                 # export_to_dot(...) -> paste into viz-js.com
    --summary                              # human-readable node/edge/token summary
```
It calls `_build_session(session_index)` to get one `ReplaySession.graph`, then
`graph_to_dict` / `visualize_graph` on it. Because generation is seeded, the dumped
graph is exactly what a run with the same seed replays — making this the primary
**debugging + communication** tool (inspect topology, roots, per-node token counts, and
dependency-edge types before launching a full run).

Caveat inherited from the exporter (§11): its per-edge `wait_ms` label is read off the
successor, so multi-predecessor merge nodes show the same wait on every in-edge —
cosmetic only. A synthetic-specific enrichment (label sub-agent/depth per node) is a
possible later nicety, not required for v1.
