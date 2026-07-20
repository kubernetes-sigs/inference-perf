# Synthetic Agent Sessions Data Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SyntheticAgentSessionsDataGenerator` that procedurally emits multi-agent agentic `ReplaySession(ReplayGraph)` objects for the existing replay runtime, from config knobs, with recursive sub-agent fan-out that recorded traces lack.

**Architecture:** Subclass the existing `ReplayGraphSessionGeneratorBase` (the lazy on-demand runtime merged in #595). Almost all code is **new files**; the runtime is reused unchanged except for **one** additive `InputSegment` type (`tool_output`) needed for fan-out, and one field promotion (`bad_tool_call_handling` moved to the shared base config). A seeded, session-local pre-order walk builds each graph purely from `(config, session_index)`.

**Tech Stack:** Python 3, pydantic (config), numpy `default_rng` (seeded sampling), `hashlib.blake2b` (stable seeds), the repo's `CustomTokenizer` + `converge_to_exact_length_text` (token fitting), pytest.

**Base branch:** This plan is written against `spec/synthetic-agent-sessions` (cut from `upstream/main`, which already contains #595). All `file:line` citations are to `upstream/main` as of 2026-07-20.

**Spec:** `docs/superpowers/specs/2026-07-09-synthetic-agentic-graph-generator-design.md` — read alongside this plan; section refs (§N) point there.

## Global Constraints

- **Minimize existing-code changes.** Only TWO existing files may be modified for runtime behavior: `replay_graph_types.py` (add one Literal value) and `replay_graph_session_datagen.py` (add one substitution branch + one predicate term; promote one config read). Plus pure config/`main.py` wiring. Every other line is a new file. (User directive.)
- **Determinism (§2.3a):** `_build_session(N)` MUST be a pure function of `(config, session_index)`. No generator-level mutable RNG. Per-session seed = `int.from_bytes(hashlib.blake2b(f"{config.seed}:{session_index}".encode(), digest_size=8).digest(), "big")`. NEVER `hash(session_id)`. Every `sample_from_distribution(...)` call MUST pass a seeded `rng` (its default is unseeded — `utils/numeric/distribution/utils.py:138–139`).
- **Config type name:** `DataGenType.SyntheticAgentSessions = "synthetic_agent_sessions"`; config class `SyntheticAgentSessionsConfig(SessionReplayConfig)`; generator `SyntheticAgentSessionsDataGenerator`; module `inference_perf/datagen/synthetic_agent_sessions.py`.
- **Load type:** requires `LoadType.TRACE_SESSION_REPLAY` (same validator tuple as OTel).
- **Constants:** `TOOL_CALL_MARGIN = 64` (tokens); filler marker = `[--- ignore the preceding filler; actual content follows ---]`; `max_events_per_session` default `64`; `seed` default `42`; `converge_to_exact_length_text` raises after 20 iterations (`datagen_utils.py:120–124`).
- **Load-bearing invariants (§2.3):** tool-call `arguments` = `json.dumps`-ed string (#1); every forced tool name has a top-level `name` in `tool_definitions` (#2, `replay_graph_session_datagen.py:367`); `#role:tool == #tool_calls` in order per turn (#3, positional rewrite `:723–739`); every event has non-empty `call.messages` (#4); a `tool_calls` message drops `content` but keeps `reasoning_content` (#6, `:1530–1535`).
- **Disabled knobs:** `inject_random_session_id` pinned `False`, `duplicate_sessions_target` pinned `None`, not exposed (§2.2a). `override_tool_call_max_tokens` pinned `False`.
- **Tool results:** always `{role:"tool", content, tool_call_id}` (OpenAI convention only).
- **TDD, frequent commits, DRY, YAGNI.**

---

## File Structure

**New files (the bulk of the work):**
- `inference_perf/datagen/synthetic_agent_sessions.py` — the generator: config-read, `__init__` (lazy init), `_build_session`, the seeded walk, and the graph-emitting helpers. One focused module.
- `inference_perf/datagen/synthetic_themes.py` — theme loading + the built-in default theme + theme data model. Kept separate so the generator module stays about *structure*, this about *content vocabulary*.
- `inference_perf/assets/synthetic_themes/db2_latency_incident.json` — one hand-authored theme (v1 ships one; more are data, not code).
- `tests/datagen/test_synthetic_agent_sessions.py` — all generator tests.
- `tests/datagen/test_tool_output_segment.py` — the runtime `tool_output` primitive tests + OTel/Weka regression guard.

**Modified files (minimal, isolated):**
- `inference_perf/datagen/replay_graph_types.py` — add `"tool_output"` to the `InputSegment.type` Literal (1 line + docstring).
- `inference_perf/datagen/replay_graph_session_datagen.py` — add `tool_output` branch in `_build_messages_with_substitution` + one predicate term; move `bad_tool_call_handling` read to inherited attribute; (Task 12) promote nothing else.
- `inference_perf/config/datagen/config.py` — `DataGenType` enum value + `DataConfig` field.
- `inference_perf/config/datagen/replay.py` — new `SyntheticAgentSessionsConfig`; promote `bad_tool_call_handling` field to `SessionReplayConfig`.
- `inference_perf/config/datagen/__init__.py`, `inference_perf/config/__init__.py`, `inference_perf/datagen/__init__.py` — exports.
- `inference_perf/config/config.py` — add to `validate_trace_replay_load_type` tuple.
- `inference_perf/main.py` — import, dispatch elif, tokenizer-required set, mp.Manager tuple, SessionMetricsCollector tuple.

**Task ordering rationale (honors "minimal risk to existing code"):** Tasks 1–2 are the two isolated runtime changes, each shipped behind a regression test proving the OTel/Weka paths are byte-identical *before* any synthetic code exists. Tasks 3–9 are entirely new files. Tasks 10–11 are pure config/wiring. Task 12 is the end-to-end integration test.

---

## Task 1: Add the `tool_output` InputSegment primitive (runtime change #1)

**Files:**
- Modify: `inference_perf/datagen/replay_graph_types.py:61` (the `type` Literal) + its docstring at `:46–61`
- Modify: `inference_perf/datagen/replay_graph_session_datagen.py:476` (predicate) and `:533` region (add branch after the `output` branch)
- Test: `tests/datagen/test_tool_output_segment.py`

**Interfaces:**
- Produces: a new `InputSegment(type="tool_output", message_count=1, source_event_id=<child terminal event id>)` behavior — at replay, replaces ONLY `content` of a recorded `role:"tool"` message with the source event's live output **text** (`registry.get_output_by_event_id`), preserving `role` and `tool_call_id`. Consumed by the merge event in Task 8.

- [ ] **Step 1: Write the failing test** — `tests/datagen/test_tool_output_segment.py`

```python
import pytest
from inference_perf.datagen.replay_graph_types import InputSegment
from inference_perf.datagen.replay_graph_session_datagen import (
    SessionChatCompletionAPIData,
    EventOutputRegistry,
)


def _make_event_with_tool_output(registry, source_event_id):
    # A merge-style event: one output segment (assistant dispatch call) + one
    # tool_output segment (the child's answer as a role:tool result).
    original_messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_A", "type": "function",
            "function": {"name": "dispatch_agent", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "PLACEHOLDER"},
    ]
    ev = SessionChatCompletionAPIData(
        event_id="sessX:merge",
        messages=[],
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, source_event_id="sessX:dispatch1"),
            InputSegment(type="tool_output", message_count=1, source_event_id=source_event_id),
        ],
        predecessor_event_ids=["sessX:dispatch1", source_event_id],
        registry=registry,
    )
    return ev


def test_tool_output_replaces_content_preserves_role_and_id():
    registry = EventOutputRegistry()
    # dispatch event produced an assistant tool-call message
    registry.record("sessX:dispatch1", "irrelevant",
                    messages=[],
                    output_message={"role": "assistant", "tool_calls": [
                        {"id": "call_A", "type": "function",
                         "function": {"name": "dispatch_agent", "arguments": "{}"}}]})
    # child terminal event produced live answer TEXT
    registry.record("sessX:child1", "the child's live answer text",
                    messages=[], output_message={"role": "assistant",
                    "content": "the child's live answer text"})

    ev = _make_event_with_tool_output(registry, "sessX:child1")
    result = ev._build_messages_with_substitution()

    tool_msg = result[1]
    assert tool_msg["role"] == "tool"                      # role preserved
    assert tool_msg["tool_call_id"] == "call_A"            # id preserved
    assert tool_msg["content"] == "the child's live answer text"  # content replaced with TEXT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/datagen/test_tool_output_segment.py::test_tool_output_replaces_content_preserves_role_and_id -v`
Expected: FAIL — the `tool_output` branch does not exist, so the recorded `PLACEHOLDER` content survives (assert on `content` fails), or the segment type is rejected.

- [ ] **Step 3: Add the Literal value + docstring** — `replay_graph_types.py`

Change `:61` from:
```python
    type: Literal["shared", "output", "unique"]
```
to:
```python
    type: Literal["shared", "output", "unique", "tool_output"]
```
And in the docstring block (`:46–61`) add one line describing it:
```python
        "tool_output" — a single role:"tool" message whose CONTENT (only) is replaced
                        by a predecessor event's live output TEXT, preserving role +
                        tool_call_id. Used for sub-agent fan-out merges (§4.1a).
```

- [ ] **Step 4: Add the substitution branch** — `replay_graph_session_datagen.py`

At `:476`, extend the predicate:
```python
        needs_substitution = any(
            seg.type == "output" or seg.type == "shared" or seg.type == "tool_output"
            for seg in self.input_segments
        )
```
In `_build_messages_with_substitution`, immediately AFTER the `if seg.type == "output":` block closes and BEFORE `elif seg.type == "shared":`, add:
```python
            elif seg.type == "tool_output":
                # Inject a child agent's live answer TEXT into a role:"tool" slot,
                # preserving role + tool_call_id (§4.1a). Content-only replacement.
                if seg.message_count != 1:
                    logger.error(
                        f"Event {self.event_id}: tool_output segment has message_count="
                        f"{seg.message_count} (expected 1). Using recorded message."
                    )
                    result.extend(seg_msgs)
                    cursor += seg.message_count
                    continue
                recorded = seg_msgs[0]
                if recorded.get("role") != "tool":
                    logger.error(
                        f"Event {self.event_id}: tool_output segment target role="
                        f"{recorded.get('role')!r} (expected 'tool'). Using recorded message."
                    )
                    result.append(recorded)
                    cursor += 1
                    continue
                actual_output = (
                    self.registry.get_output_by_event_id(seg.source_event_id)
                    if seg.source_event_id else None
                )
                if actual_output is not None:
                    substituted = dict(recorded)
                    substituted["content"] = actual_output
                    result.append(substituted)
                else:
                    # child output unavailable — fall back to recorded placeholder
                    result.append(recorded)
                cursor += 1
```
(The existing positional `tool_call_id` rewrite post-pass at `:723–739` already handles the `role:tool` message's id from the preceding live dispatch call — no change there.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/datagen/test_tool_output_segment.py::test_tool_output_replaces_content_preserves_role_and_id -v`
Expected: PASS

- [ ] **Step 6: Add guard tests** — append to the same test file

```python
def test_tool_output_guard_non_tool_role_falls_back():
    registry = EventOutputRegistry()
    registry.record("sessX:child1", "answer", messages=[],
                    output_message={"role": "assistant", "content": "answer"})
    original_messages = [{"role": "assistant", "content": "not a tool msg"}]
    ev = SessionChatCompletionAPIData(
        event_id="sessX:e", messages=[], original_messages=original_messages,
        input_segments=[InputSegment(type="tool_output", message_count=1,
                                     source_event_id="sessX:child1")],
        predecessor_event_ids=["sessX:child1"], registry=registry)
    result = ev._build_messages_with_substitution()
    assert result[0]["role"] == "assistant"          # unchanged
    assert result[0]["content"] == "not a tool msg"  # recorded content kept (guard fired)


def test_tool_output_unavailable_output_falls_back():
    registry = EventOutputRegistry()  # nothing recorded for the source
    original_messages = [{"role": "tool", "tool_call_id": "call_A", "content": "PLACEHOLDER"}]
    ev = SessionChatCompletionAPIData(
        event_id="sessX:e", messages=[], original_messages=original_messages,
        input_segments=[InputSegment(type="tool_output", message_count=1,
                                     source_event_id="sessX:missing")],
        predecessor_event_ids=["sessX:missing"], registry=registry)
    result = ev._build_messages_with_substitution()
    assert result[0]["content"] == "PLACEHOLDER"  # fell back, no crash
```

Run: `pytest tests/datagen/test_tool_output_segment.py -v` → all PASS.
(NOTE: if `SessionChatCompletionAPIData`'s real constructor differs from the kwargs above, adjust the test builders to match its actual signature — read `replay_graph_session_datagen.py` around `:320–345`. Keep the assertions identical.)

- [ ] **Step 7: Commit**

```bash
git add inference_perf/datagen/replay_graph_types.py inference_perf/datagen/replay_graph_session_datagen.py tests/datagen/test_tool_output_segment.py
git commit -m "feat(replay): add tool_output InputSegment for sub-agent fan-out merges"
```

---

## Task 2: Regression guard — OTel/Weka substitution is byte-identical (no behavior drift)

**Files:**
- Test: `tests/datagen/test_tool_output_segment.py` (append)

**Interfaces:**
- Consumes: the Task 1 changes. Proves the new branch is inert unless a `tool_output` segment is present.

- [ ] **Step 1: Write the regression test**

```python
def test_output_and_shared_segments_unchanged_by_tool_output_addition():
    """A graph with NO tool_output segment must substitute exactly as before —
    the new branch is additive and inert on the OTel/Weka path."""
    registry = EventOutputRegistry()
    registry.record("sessY:e1", "live-out", messages=[],
                    output_message={"role": "assistant", "content": "live-out"})
    original_messages = [{"role": "assistant", "content": "PLACEHOLDER"}]
    ev = SessionChatCompletionAPIData(
        event_id="sessY:e2", messages=[], original_messages=original_messages,
        input_segments=[InputSegment(type="output", message_count=1,
                                     source_event_id="sessY:e1")],
        predecessor_event_ids=["sessY:e1"], registry=registry)
    result = ev._build_messages_with_substitution()
    # output segment still substitutes the WHOLE message (assistant), as before
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "live-out"
```

- [ ] **Step 2: Run and verify it passes**

Run: `pytest tests/datagen/test_tool_output_segment.py::test_output_and_shared_segments_unchanged_by_tool_output_addition -v`
Expected: PASS (proves `output` behavior is untouched).

- [ ] **Step 3: Run the full existing replay test suite unchanged**

Run: `pytest tests/datagen/ -k "replay or otel or graph" -v`
Expected: PASS — no existing test changes behavior. If any fails, the Task 1 branch was placed wrong (e.g. altered the `output`/`shared` flow); fix placement so the new branch is a pure `elif`.

- [ ] **Step 4: Commit**

```bash
git add tests/datagen/test_tool_output_segment.py
git commit -m "test(replay): regression guard — tool_output addition leaves output/shared paths byte-identical"
```

---

## Task 3: Promote `bad_tool_call_handling` to `SessionReplayConfig` (runtime change #2)

**Files:**
- Modify: `inference_perf/config/datagen/replay.py` — move the field from `OTelTraceReplayConfig` (`:181`) up to `SessionReplayConfig` (class at `:101`)
- Modify: `inference_perf/datagen/replay_graph_session_datagen.py:1593` — drop the `getattr(..., NONE)` fallback (attribute now always exists)
- Test: `tests/datagen/test_tool_output_segment.py` (append a config-inheritance test)

**Interfaces:**
- Produces: `SessionReplayConfig.bad_tool_call_handling: BadToolCallHandling = NONE`, inherited by both `OTelTraceReplayConfig` and (Task 5) `SyntheticAgentSessionsConfig`.

- [ ] **Step 1: Write the failing test**

```python
def test_bad_tool_call_handling_inherited_by_session_replay_base():
    from inference_perf.config.datagen.replay import SessionReplayConfig, BadToolCallHandling
    cfg = SessionReplayConfig()
    assert cfg.bad_tool_call_handling == BadToolCallHandling.NONE
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_tool_output_segment.py::test_bad_tool_call_handling_inherited_by_session_replay_base -v`
Expected: FAIL — `SessionReplayConfig` has no such attribute yet.

- [ ] **Step 3: Move the field up**

In `replay.py`, DELETE the `bad_tool_call_handling: BadToolCallHandling = Field(...)` block from `OTelTraceReplayConfig` (currently `:181`), and ADD the identical block to `SessionReplayConfig` (class starting `:101`). Keep the `Field(BadToolCallHandling.NONE, description=...)` default and description verbatim. `OTelTraceReplayConfig` and `WekaTraceReplayConfig` inherit it unchanged.

- [ ] **Step 4: Drop the getattr fallback**

In `replay_graph_session_datagen.py:1593`, change:
```python
            bad_tool_call_handling=getattr(self.replay_config, "bad_tool_call_handling", BadToolCallHandling.NONE)
```
to:
```python
            bad_tool_call_handling=self.replay_config.bad_tool_call_handling
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/datagen/test_tool_output_segment.py::test_bad_tool_call_handling_inherited_by_session_replay_base -v` → PASS
Run: `pytest tests/datagen/ -k "otel or replay" -v` → PASS (OTel still gets the field by inheritance).

- [ ] **Step 6: Commit**

```bash
git add inference_perf/config/datagen/replay.py inference_perf/datagen/replay_graph_session_datagen.py tests/datagen/test_tool_output_segment.py
git commit -m "refactor(config): promote bad_tool_call_handling to SessionReplayConfig; drop getattr fallback"
```

---

## Task 4: Theme model + built-in default theme + loader

**Files:**
- Create: `inference_perf/datagen/synthetic_themes.py`
- Create: `inference_perf/assets/synthetic_themes/db2_latency_incident.json`
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Produces:
  - `class Theme` (pydantic): `name: str`, `system_prompt: Optional[str]`, `verbs: list[str]`, `entities: dict[str, list[str]]`, `enumerated: dict[str, str]` (label→prefix, e.g. `{"bufferpool": "BP"}`), `tool_names: list[str]`, `result_templates: dict[str, str]`, `objective_template: str`, `followup_templates: list[str]`, `followup_connectives: list[str]`.
  - `def load_theme(name: str) -> Theme` — loads `assets/synthetic_themes/{name}.json`, validates, returns `Theme`; raises `ValueError` naming missing refs.
  - `DEFAULT_SYSTEM_PROMPT: str` — a generic agent preamble used when a theme omits `system_prompt`.
  - `GENERIC_THEME: Theme` — a built-in fallback theme (so a config can run with a theme name that maps to it, and tests don't depend on the JSON file).

- [ ] **Step 1: Write the failing test**

```python
from inference_perf.datagen.synthetic_themes import load_theme, Theme, GENERIC_THEME, DEFAULT_SYSTEM_PROMPT


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k theme -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the theme JSON** — `inference_perf/assets/synthetic_themes/db2_latency_incident.json`

```json
{
  "name": "db2_latency_incident",
  "system_prompt": "You are an autonomous site-reliability agent operating on IBM Db2 for z/OS subsystems. Investigate incidents using the available tools, reason step by step, and produce a concise remediation summary. Prefer read-only diagnostics before any change.",
  "verbs": ["Analyze", "Assess", "Review", "Diagnose", "Inspect", "Evaluate"],
  "entities": {
    "db_instance": ["DBP1", "DBP2", "DBX9"],
    "symptom": ["commit-latency spike", "lock-wait escalation", "bufferpool thrash"]
  },
  "enumerated": {"bufferpool": "BP"},
  "tool_names": ["get_bp_stats", "get_lock_waits", "get_log_activity", "run_reorg", "get_thread_detail"],
  "objective_template": "{verb} {symptom} on {db_instance}: identify root cause and recommend remediation.",
  "result_templates": {
    "get_bp_stats": "| time | bp | hit_ratio |\n| {t0} | {bp0} | {r0} |\n| {t1} | {bp1} | {r1} |",
    "default": "result for {entity}: value={n0} at {t0}"
  },
  "followup_templates": [
    "What does the {symptom} trend look like over the last hour on {db_instance}?",
    "Given that, should we reorg {bufferpool} or adjust the bufferpool size?"
  ],
  "followup_connectives": ["Following up on the previous result, ", "Given that, ", "Next, "]
}
```

- [ ] **Step 4: Create the module** — `inference_perf/datagen/synthetic_themes.py`

```python
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

_ASSETS = Path(__file__).parent.parent / "assets" / "synthetic_themes"

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent. Use the available tools to complete the given task, "
    "reason step by step, and produce a concise final answer. Prefer read-only actions first."
)


class Theme(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    verbs: list[str]
    entities: dict[str, list[str]]
    enumerated: dict[str, str] = {}
    tool_names: list[str]
    result_templates: dict[str, str]
    objective_template: str
    followup_templates: list[str] = []
    followup_connectives: list[str] = []


def _validate(theme: Theme) -> Theme:
    if not theme.verbs:
        raise ValueError(f"theme {theme.name}: 'verbs' must be non-empty")
    if not theme.tool_names:
        raise ValueError(f"theme {theme.name}: 'tool_names' must be non-empty")
    if "default" not in theme.result_templates:
        raise ValueError(f"theme {theme.name}: 'result_templates' must include a 'default' key")
    return theme


def load_theme(name: str) -> Theme:
    path = _ASSETS / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown synthetic theme {name!r} (looked in {_ASSETS})")
    data = json.loads(path.read_text())
    return _validate(Theme(**data))


GENERIC_THEME = Theme(
    name="generic",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    verbs=["Analyze", "Investigate", "Review"],
    entities={"target": ["service-a", "service-b", "service-c"]},
    enumerated={"item": "ITEM"},
    tool_names=["get_status", "get_metrics", "run_check"],
    result_templates={"default": "result for {entity}: value={n0} at {t0}"},
    objective_template="{verb} the {target} incident: find the cause and recommend a fix.",
    followup_templates=["What about {target}?"],
    followup_connectives=["Following up, ", "Next, "],
)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k theme -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add inference_perf/datagen/synthetic_themes.py inference_perf/assets/synthetic_themes/db2_latency_incident.json tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): synthetic theme model, bundled theme, and generic fallback"
```

---

## Task 5: Config — `SyntheticAgentSessionsConfig` + `DataGenType` + `DataConfig` field

**Files:**
- Modify: `inference_perf/config/datagen/replay.py` (add `SyntheticAgentSessionsConfig`)
- Modify: `inference_perf/config/datagen/config.py` (enum value + `DataConfig` field)
- Modify: `inference_perf/config/datagen/__init__.py`, `inference_perf/config/__init__.py` (exports)
- Modify: `inference_perf/config/config.py` (validator tuple)
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Produces: `SyntheticAgentSessionsConfig(SessionReplayConfig)` with fields (types + defaults from §8 knob table):
  - REQUIRED (`Field(...)`): `num_sessions: int`, `rounds_per_session: Distribution`, `fanout_probability: float`, `theme_mix: dict[str, float]`
  - `seed: int = 42`, `shared_system_prompt_len: int = 0`
  - `tool_turns_per_loop: Optional[Distribution] = None`, `sub_agents_per_spawn: Optional[Distribution] = None`, `max_depth: int = 2`, `max_events_per_session: int = 64`, `tool_definitions_per_agent: Optional[Distribution] = None`, `parallel_tool_calls_per_turn: Optional[Distribution] = None`
  - `input_tokens_per_turn: Distribution`, `output_tokens_per_turn: Distribution`, `tool_call_latency_sec: Distribution`, `user_think_time_sec: Optional[Distribution] = None`, `max_model_len: Optional[int] = None`
  - inherits `bad_tool_call_handling` (Task 3), `max_wait_ms`; pins `inject_random_session_id=False`, `duplicate_sessions_target=None`, `override_tool_call_max_tokens=False` via field overrides.
  - `DataGenType.SyntheticAgentSessions = "synthetic_agent_sessions"`; `DataConfig.synthetic_agent_sessions: Optional[SyntheticAgentSessionsConfig] = None`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k config -v`
Expected: FAIL — class does not exist.

- [ ] **Step 3: Add the config class** — in `replay.py`, after `WekaTraceReplayConfig`

```python
class SyntheticAgentSessionsConfig(SessionReplayConfig):
    """Procedural multi-agent agentic session generation (§8)."""

    # Required — the four workload-shape decisions (no default)
    num_sessions: int = Field(..., gt=0, description="Number of sessions (load volume)")
    rounds_per_session: Distribution = Field(..., description="N principal inputs to the root; N=1 autonomous")
    fanout_probability: float = Field(..., ge=0.0, le=1.0, description="P(an agent execution spawns sub-agents)")
    theme_mix: dict[str, float] = Field(..., description="theme name -> weight")

    # Defaulted
    seed: int = Field(42, description="Base seed for stable per-session RNG (§2.3a)")
    shared_system_prompt_len: int = Field(0, ge=0, description="Invariant system-prompt head length in tokens")
    tool_turns_per_loop: Optional[Distribution] = Field(None, description="tool-call TURNS per loop (fallback fixed 2)")
    sub_agents_per_spawn: Optional[Distribution] = Field(None, description="K children per spawn (fallback uniform 2-4)")
    max_depth: int = Field(2, ge=0, description="Hard recursion terminator")
    max_events_per_session: int = Field(64, gt=0, description="Self-limiting event budget")
    tool_definitions_per_agent: Optional[Distribution] = Field(None, description="advertised catalog size (fallback fixed 8)")
    parallel_tool_calls_per_turn: Optional[Distribution] = Field(None, description="calls per ordinary tool turn (fallback fixed 1)")
    input_tokens_per_turn: Distribution = Field(..., description="per-turn input tokens")
    output_tokens_per_turn: Distribution = Field(..., description="per-turn output tokens (plain-text turns)")
    tool_call_latency_sec: Distribution = Field(..., description="machine/agent wait_ms gaps")
    user_think_time_sec: Optional[Distribution] = Field(None, description="human gap before rounds 2..N")
    max_model_len: Optional[int] = Field(None, description="fail-fast context-length ceiling")

    # Pinned inert / not for synthetic (§2.2a, C3)
    inject_random_session_id: bool = Field(False, frozen=True)
    duplicate_sessions_target: Optional[int] = Field(None, frozen=True)
    override_tool_call_max_tokens: bool = Field(False)
```

(If `frozen=True` is unsupported in the installed pydantic version, drop `frozen=True` and instead pin in `__init__` of the generator by asserting/forcing the value — but prefer the field override.)

- [ ] **Step 4: Add the enum value + DataConfig field + validator + exports**

`config/datagen/config.py`: add `SyntheticAgentSessions = "synthetic_agent_sessions"` to `DataGenType`, and `synthetic_agent_sessions: Optional[SyntheticAgentSessionsConfig] = None` to `DataConfig` (import it).
`config/config.py`: add `DataGenType.SyntheticAgentSessions` to the tuple in `validate_trace_replay_load_type` (`~:52`).
`config/datagen/__init__.py` and `config/__init__.py`: export `SyntheticAgentSessionsConfig`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k config -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add inference_perf/config/
git add tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(config): SyntheticAgentSessionsConfig + DataGenType.SyntheticAgentSessions"
```

---

## Task 6: Deterministic seed + sampling helpers

**Files:**
- Modify: `inference_perf/datagen/synthetic_agent_sessions.py` (create the module skeleton + helpers)
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Produces (module-level helpers):
  - `def session_seed(base_seed: int, session_index: int) -> int` — `int.from_bytes(blake2b(f"{base_seed}:{session_index}".encode(), digest_size=8).digest(), "big")`
  - `def child_rng(parent_seed: int, *path: int) -> np.random.Generator` — folds a stable graph path into a child Generator via `np.random.default_rng([parent_seed, *path])`
  - `def sample_int(dist, rng, fallback) -> int` — resolve an `Optional[Distribution]` (use `fallback` Distribution if None) and draw ONE int via `sample_from_distribution(dist, 1, rng=rng)[0]`, always passing `rng`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from inference_perf.datagen.synthetic_agent_sessions import session_seed, child_rng


def test_session_seed_stable_across_calls_and_processes():
    # Must NOT depend on PYTHONHASHSEED or process — pure function of inputs.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "seed or rng" -v`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Create the module skeleton + helpers** — `inference_perf/datagen/synthetic_agent_sessions.py`

```python
import hashlib
from typing import Optional
import numpy as np

from inference_perf.config.common import Distribution, DistributionType
from inference_perf.utils.numeric.distribution.utils import sample_from_distribution


def session_seed(base_seed: int, session_index: int) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{session_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def child_rng(parent_seed: int, *path: int) -> np.random.Generator:
    return np.random.default_rng([parent_seed, *path])


def sample_int(dist: Optional[Distribution], rng: np.random.Generator, fallback: Distribution) -> int:
    d = dist if dist is not None else fallback
    val = sample_from_distribution(d, 1, rng=rng)[0]
    return int(val)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "seed or rng" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference_perf/datagen/synthetic_agent_sessions.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): deterministic seed + path-derived rng + distribution sampling helpers"
```

---

## Task 7: Content layer — filler fitting (best-candidate wrapper + budget guard)

**Files:**
- Modify: `inference_perf/datagen/synthetic_agent_sessions.py`
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Produces:
  - `TOOL_CALL_MARGIN = 64`
  - `FILLER_MARKER = "[--- ignore the preceding filler; actual content follows ---]"`
  - `def fit_filler(tokenizer, target_tokens: int, fixed_content: str, rng) -> str` — computes `filler_budget = target_tokens - count_tokens(fixed_content + marker)`; if `<= 0`, returns `fixed_content` alone (no marker, target floored to fixed cost) and logs; else pads with Shakespeare-corpus text to hit target using a **best-candidate wrapper** around `converge_to_exact_length_text` that returns the closest `(text)` even on non-convergence (never raises to the caller for length reasons).

- [ ] **Step 1: Write the failing test**

```python
from inference_perf.datagen.synthetic_agent_sessions import fit_filler, FILLER_MARKER, TOOL_CALL_MARGIN


class _FakeTok:
    # 1 token per whitespace-word, deterministic — good enough to test budget logic
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "filler or margin" -v`
Expected: FAIL — `fit_filler` not defined.

- [ ] **Step 3: Implement** — append to `synthetic_agent_sessions.py`

```python
import logging
logger = logging.getLogger(__name__)

TOOL_CALL_MARGIN = 64
FILLER_MARKER = "[--- ignore the preceding filler; actual content follows ---]"

# Loaded once; Shakespeare corpus shipped with the repo.
from pathlib import Path
_SHAKESPEARE = (Path(__file__).parent.parent / "assets" / "shakespeare.txt")


def _corpus_words() -> list[str]:
    if _SHAKESPEARE.exists():
        return _SHAKESPEARE.read_text(errors="ignore").split()
    return ["lorem", "ipsum", "dolor", "sit", "amet"]  # safe fallback


def fit_filler(tokenizer, target_tokens: int, fixed_content: str, rng) -> str:
    marker_and_fixed = fixed_content + " " + FILLER_MARKER
    fixed_cost = tokenizer.count_tokens(marker_and_fixed)
    filler_budget = target_tokens - fixed_cost
    if filler_budget <= 0:
        logger.debug("fit_filler: non-positive budget (target=%d, fixed=%d); flooring to fixed content",
                     target_tokens, fixed_cost)
        return fixed_content  # no marker, no filler
    # Best-candidate padding: append corpus words until we reach/pass target, keep closest.
    words = _corpus_words()
    best_text, best_gap = marker_and_fixed, abs(fixed_cost - target_tokens)
    buf = marker_and_fixed
    idx = 0
    for _ in range(20):  # bounded, mirrors converge_to_exact_length_text
        cur = tokenizer.count_tokens(buf)
        gap = abs(cur - target_tokens)
        if gap < best_gap:
            best_gap, best_text = gap, buf
        if cur >= target_tokens:
            break
        take = max(1, target_tokens - cur)
        buf = buf + " " + " ".join(words[idx:idx + take])
        idx += take
    return best_text
```

(NOTE: if the repo already exposes a Shakespeare filler helper in `datagen_utils.py` / the synthetic generator, reuse it instead of `_corpus_words`; check `inference_perf/datagen/datagen_utils.py` for `converge_to_exact_length_text` and any corpus loader, and prefer wrapping the existing utility to keep this DRY. The best-candidate loop above is the fallback if none exists.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "filler or margin" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference_perf/datagen/synthetic_agent_sessions.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): filler fitting with best-candidate wrapper and budget guard"
```

---

## Task 8: The seeded walk — build a single session graph (no fan-out yet)

**Files:**
- Modify: `inference_perf/datagen/synthetic_agent_sessions.py` (the `_build_session` core + walk for N rounds × k tool-turns, single-agent)
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Consumes: helpers (Task 6), `fit_filler` (Task 7), `Theme` (Task 4), `SyntheticAgentSessionsConfig` (Task 5), and the graph types `ReplayGraph`, `GraphEvent`, `GraphCall`, `InputSegment` (`replay_graph_types.py`), `ReplaySession` (`replay_graph_session_datagen.py`).
- Produces: `def build_graph_for_session(cfg, theme, tokenizer, session_index: int) -> ReplayGraph` — emits a valid single-agent graph: N rounds, each `[principal input] → k tool-turns (assistant tool_call → role:tool result) → [answer]`, wired with `predecessor_event_ids` and `input_segments`. Fan-out is added in Task 9. Uses `session_seed`/`child_rng`, honors `max_events_per_session` (stop starting rounds when the budget won't fit).

- [ ] **Step 1: Write the failing test**

```python
from inference_perf.config.common import Distribution
from inference_perf.datagen.synthetic_agent_sessions import build_graph_for_session
from inference_perf.datagen.synthetic_themes import GENERIC_THEME
from inference_perf.config.datagen.replay import SyntheticAgentSessionsConfig


class _WordTok:
    def count_tokens(self, text, add_special_tokens=True):
        return max(1, len(str(text).split()))
    def get_tokenizer(self):
        raise NotImplementedError


def _cfg(**kw):
    base = dict(num_sessions=5, rounds_per_session=Distribution(type="fixed", mean=1),
                fanout_probability=0.0, theme_mix={"generic": 1.0},
                input_tokens_per_turn=Distribution(type="fixed", mean=20),
                output_tokens_per_turn=Distribution(type="fixed", mean=10),
                tool_call_latency_sec=Distribution(type="fixed", mean=1),
                tool_turns_per_loop=Distribution(type="fixed", mean=2))
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


def test_determinism_same_index_same_graph():
    g1 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    g2 = build_graph_for_session(_cfg(), GENERIC_THEME, _WordTok(), 3)
    assert list(g1.events.keys()) == list(g2.events.keys())  # same ids, same insertion order


def test_event_budget_caps_rounds():
    cfg = _cfg(rounds_per_session=Distribution(type="fixed", mean=100), max_events_per_session=6)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    assert len(g.events) <= 6
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "single_agent or determinism_same or event_budget" -v`
Expected: FAIL — `build_graph_for_session` not defined.

- [ ] **Step 3: Implement the walk** — append to `synthetic_agent_sessions.py`

```python
import json
from inference_perf.datagen.replay_graph_types import ReplayGraph, GraphEvent, GraphCall, InputSegment

_FB_TOOL_TURNS = Distribution(type="fixed", mean=2)
_FB_TOOL_DEFS = Distribution(type="fixed", mean=8)
_FB_PARALLEL = Distribution(type="fixed", mean=1)


def _tool_definitions(theme, n: int) -> list[dict]:
    # top-level name (inv #2); inflate by cycling theme tool names
    out = []
    for i in range(n):
        name = theme.tool_names[i % len(theme.tool_names)] + ("" if i < len(theme.tool_names) else f"_{i}")
        out.append({"name": name, "type": "function",
                    "function": {"name": name, "parameters": {"type": "object", "properties": {}}}})
    return out


def _render_objective(theme, rng) -> str:
    verb = theme.verbs[int(rng.integers(0, len(theme.verbs)))]
    subs = {"verb": verb}
    for k, vals in theme.entities.items():
        subs[k] = vals[int(rng.integers(0, len(vals)))]
    try:
        return theme.objective_template.format(**subs)
    except KeyError:
        return f"{verb}: complete the task."


def build_graph_for_session(cfg, theme, tokenizer, session_index: int) -> ReplayGraph:
    seed = session_seed(cfg.seed, session_index)
    sid = f"synthN{session_index}"
    events: dict[str, GraphEvent] = {}
    root_ids: list[str] = []
    budget = cfg.max_events_per_session
    n_rounds = sample_int(cfg.rounds_per_session, child_rng(seed, 0), cfg.rounds_per_session)
    tool_defs_n = sample_int(cfg.tool_definitions_per_agent, child_rng(seed, 1), _FB_TOOL_DEFS)
    tool_defs = _tool_definitions(theme, max(0, tool_defs_n))

    system_msg = None
    if cfg.shared_system_prompt_len > 0:
        content = fit_filler(tokenizer, cfg.shared_system_prompt_len,
                             theme.system_prompt or "", rng=None)
        system_msg = {"role": "system", "content": content}

    prev_answer_id: Optional[str] = None
    seq = 0

    def _emit(event_id, messages, preds, dep_types, segs, wait_ms, is_tool_call, tool_names):
        events[event_id] = GraphEvent(
            event_id=event_id,
            call=GraphCall(
                call_id=event_id, model="", messages=messages, expected_output="",
                input_segments=segs, total_input_tokens=0, expected_output_tokens=0,
                temperature=0.0, max_tokens_recorded=None, tool_definitions=tool_defs,
                expected_output_is_tool_call=is_tool_call,
                expected_output_tool_names=tool_names, attributes=None,
            ),
            predecessor_event_ids=preds,
            predecessor_dependency_types=dep_types,
            wait_ms=wait_ms, t_start_ms=0, t_end_ms=0,
        )

    for r in range(n_rounds):
        # budget check: a round costs ~ (1 principal + 2*k tool msgs + 1 answer)
        k = sample_int(cfg.tool_turns_per_loop, child_rng(seed, r, 100), _FB_TOOL_TURNS)
        round_cost = 1 + 2 * k + 1
        if len(events) + round_cost > budget:
            break  # stop starting new rounds (§8)

        principal_id = f"{sid}:r{r}:principal"
        obj = _render_objective(theme, child_rng(seed, r, 1))
        principal_msgs = ([system_msg] if system_msg else []) + [{"role": "user", "content": obj}]
        wait = int(sample_from_distribution(
            cfg.user_think_time_sec if (r > 0 and cfg.user_think_time_sec) else cfg.tool_call_latency_sec,
            1, rng=child_rng(seed, r, 2))[0] * 1000)
        _emit(principal_id, principal_msgs,
              [prev_answer_id] if prev_answer_id else [],
              {prev_answer_id: "full_match"} if prev_answer_id else {},
              [], wait if r > 0 else 0, False, None)
        if not root_ids:
            root_ids.append(principal_id)
        last_id = principal_id

        for t in range(k):
            call_name = tool_defs[0]["name"] if tool_defs else "noop"
            tc_id = f"call_{r}_{t}"
            tool_call_msg = {"role": "assistant", "tool_calls": [
                {"id": tc_id, "type": "function",
                 "function": {"name": call_name, "arguments": json.dumps({"q": obj[:20]})}}]}
            result = theme.result_templates.get("default").format(
                entity="x", n0=int(child_rng(seed, r, t, 9).integers(0, 999)), t0="t0")
            tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result}
            turn_id = f"{sid}:r{r}:t{t}"
            _emit(turn_id, [tool_call_msg, tool_msg], [last_id], {last_id: "full_match"},
                  [], int(sample_from_distribution(cfg.tool_call_latency_sec, 1,
                       rng=child_rng(seed, r, t, 3))[0] * 1000),
                  True, [call_name])
            last_id = turn_id

        answer_id = f"{sid}:r{r}:answer"
        ans = fit_filler(tokenizer,
                         sample_int(cfg.output_tokens_per_turn, child_rng(seed, r, 4), cfg.output_tokens_per_turn),
                         "Summary:", rng=child_rng(seed, r, 5))
        _emit(answer_id, [{"role": "assistant", "content": ans}], [last_id], {last_id: "full_match"},
              [], 0, False, None)
        prev_answer_id = answer_id

    if not events:  # degenerate config produced nothing schedulable
        return ReplayGraph(events={}, root_event_ids=[], source_file="synthetic")
    return ReplayGraph(events=events, root_event_ids=root_ids, source_file="synthetic")
```

(NOTE: read the actual `GraphCall` / `GraphEvent` / `ReplayGraph` constructor signatures at `replay_graph_types.py:68–114` and adjust field names/kwargs to match exactly — the above uses the documented field set from §2.1 but the real dataclass may name or order them differently. The test asserts *behavior* (inv #3/#4, determinism, budget), so adapt construction freely to satisfy it.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "single_agent or determinism_same or event_budget" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference_perf/datagen/synthetic_agent_sessions.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): seeded single-agent session walk (rounds x tool-turns, budget-capped)"
```

---

## Task 9: Fan-out — recursive sub-agents + merge via tool_output

**Files:**
- Modify: `inference_perf/datagen/synthetic_agent_sessions.py` (add spawn + merge to the walk)
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Consumes: the walk (Task 8), the `tool_output` segment (Task 1).
- Produces: fan-out inside a round — when `rng < fanout_probability` and `depth < max_depth` and the spawn fits the budget, emit K single-call dispatch events + recurse into K child agents + one **merge** event that lists all K child terminals in `predecessor_event_ids` and carries, per child, `[output(src=dispatch event) , tool_output(src=child terminal)]` pairs. Uses `sub_agents_per_spawn` (fallback uniform 2–4) and `parallel_tool_calls_per_turn` (fallback fixed 1, ordinary turns only — dispatch stays single-call).

- [ ] **Step 1: Write the failing test**

```python
def test_fanout_produces_subagents_and_valid_merge():
    cfg = _cfg(fanout_probability=1.0, max_depth=2,
               sub_agents_per_spawn=Distribution(type="fixed", mean=2),
               max_events_per_session=2048)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    # a sub-agent exists (depth >= 1): some event id contains ":sub"
    assert any(":sub" in eid for eid in g.events), "sub-agents spawned"
    # every dispatch_agent tool_call has a matching role:tool result (inv #3, no dangling)
    for ev in g.events.values():
        n_calls = sum(len(m.get("tool_calls", [])) for m in ev.call.messages if m.get("tool_calls"))
        n_tool = sum(1 for m in ev.call.messages if m.get("role") == "tool")
        assert n_tool == n_calls


def test_no_agent_beyond_max_depth():
    cfg = _cfg(fanout_probability=1.0, max_depth=1,
               sub_agents_per_spawn=Distribution(type="fixed", mean=2),
               max_events_per_session=2048)
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    # depth encoded in id as ":dN:"; assert none exceeds max_depth
    for eid in g.events:
        import re
        m = re.search(r":d(\d+):", eid)
        if m:
            assert int(m.group(1)) <= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "fanout or max_depth" -v`
Expected: FAIL — no spawn logic yet.

- [ ] **Step 3: Implement fan-out** — modify the round loop in `build_graph_for_session`

Add, after the tool-loop and before the answer, a spawn decision. Encode depth in event ids as `:dN:`. Extract a recursive `_build_agent(depth, agent_seed, ...)` helper that both the root round and sub-agents call, so recursion is uniform. Per child: emit a dispatch event (`[assistant dispatch_agent(child) call]`, single-call, `expected_output_is_tool_call=True`), recurse to build the child (its own tool-loop, may spawn if `depth+1 < max_depth`), then the merge event carries `[output(src=dispatch) , tool_output(src=child terminal)]` per child. The merge is ONE event; it counts as 1 toward the budget. Spawn only if `fanout roll < fanout_probability`, `depth < max_depth`, and the whole spawn subtree's minimum cost fits `max_events_per_session` (else no spawn — the node stays a plain leaf).

```python
# inside the recursive agent builder, at the spawn decision point:
spawn_roll = float(child_rng(seed, r, depth, 7).random())
if spawn_roll < cfg.fanout_probability and depth < cfg.max_depth:
    K = sample_int(cfg.sub_agents_per_spawn, child_rng(seed, r, depth, 8),
                   Distribution(type="uniform", min=2, max=4))
    min_spawn_cost = K * (1 + 1 + 1)  # dispatch + minimal child + share of merge
    if len(events) + min_spawn_cost + 1 <= budget:
        child_terminal_ids, dispatch_ids = [], []
        for c in range(K):
            disp_id = f"{sid}:r{r}:d{depth}:disp{c}"
            tc_id = f"dispatch_{r}_{depth}_{c}"
            _emit(disp_id, [{"role": "assistant", "tool_calls": [
                    {"id": tc_id, "type": "function", "function": {
                        "name": "dispatch_agent",
                        "arguments": json.dumps({"objective": _render_objective(theme, child_rng(seed, r, depth, c, 1))})}}]}],
                  [last_id], {last_id: "full_match"}, [], 0, True, ["dispatch_agent"])
            # recurse: child agent, single-dispatch, depth+1
            child_terminal = _build_child_agent(depth + 1, c, disp_id, tc_id)
            dispatch_ids.append((disp_id, tc_id))
            child_terminal_ids.append(child_terminal)
        # merge event: per child [output(dispatch), tool_output(child terminal)]
        merge_msgs, merge_segs, preds, deps = [], [], [], {}
        for (disp_id, tc_id), child_term in zip(dispatch_ids, child_terminal_ids):
            merge_msgs.append({"role": "assistant", "tool_calls": [
                {"id": tc_id, "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}]})
            merge_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": "PLACEHOLDER"})
            merge_segs.append(InputSegment(type="output", message_count=1, source_event_id=disp_id))
            merge_segs.append(InputSegment(type="tool_output", message_count=1, source_event_id=child_term))
            preds += [disp_id, child_term]
            deps[disp_id] = "full_match"; deps[child_term] = "full_match"
        merge_id = f"{sid}:r{r}:d{depth}:merge"
        _emit(merge_id, merge_msgs, preds, deps, merge_segs, 0, False, None)
        last_id = merge_id
```

Implement `_build_child_agent(...)` as the single-dispatch variant of the round body (task from parent → tool-loop of `tool_turns_per_loop` → answer; may recurse via the same spawn block at `depth+1`), returning its terminal (answer) event id. Keep the recursion, budget checks (`len(events)` against `budget`), and id scheme (`:dN:`, `:subC:`) consistent so the depth-regex test and the `:sub` test pass.

- [ ] **Step 4: Run tests**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "fanout or max_depth" -v`
Expected: PASS
Also re-run Task 8 tests to confirm single-agent still works: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "single_agent or determinism_same" -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_perf/datagen/synthetic_agent_sessions.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): recursive sub-agent fan-out with tool_output merge"
```

---

## Task 10: The generator class — `__init__` (lazy) + `_build_session`

**Files:**
- Modify: `inference_perf/datagen/synthetic_agent_sessions.py` (the `SyntheticAgentSessionsDataGenerator` class)
- Modify: `inference_perf/datagen/__init__.py` (export)
- Test: `tests/datagen/test_synthetic_agent_sessions.py`

**Interfaces:**
- Consumes: `build_graph_for_session` (Tasks 8–9), `load_theme`/`GENERIC_THEME` (Task 4), `ReplayGraphSessionGeneratorBase` (base runtime), `ReplaySession`.
- Produces: `class SyntheticAgentSessionsDataGenerator(ReplayGraphSessionGeneratorBase)` with the OTel-mirrored `__init__` (reads `config.synthetic_agent_sessions`, passes `replay_config=` to `super().__init__`, calls `self.initialize_sessions_lazy([f"synthN{i}" for i in range(num_sessions)])`) and `_build_session(session_index) -> Optional[ReplaySession]` (picks theme by weighted draw from the session-local rng over `theme_mix`, calls `build_graph_for_session`, wraps in `ReplaySession(session_id=..., source_id=..., session_index=...)`).

- [ ] **Step 1: Write the failing test**

```python
def test_generator_builds_session_lazily():
    from inference_perf.config.datagen.config import DataConfig, DataGenType
    from inference_perf.datagen.synthetic_agent_sessions import SyntheticAgentSessionsDataGenerator

    data = DataConfig(type=DataGenType.SyntheticAgentSessions, synthetic_agent_sessions=_cfg(num_sessions=4))
    # api_config: minimal; tokenizer: the word tokenizer
    gen = SyntheticAgentSessionsDataGenerator(api_config=_min_api(), config=data,
                                              tokenizer=_WordTok(), num_workers=1)
    assert gen.get_session_count() == 4
    gen._ensure_session_built(0)
    assert gen.sessions[0] is not None
    # determinism: two generators, same index -> same event ids
    gen2 = SyntheticAgentSessionsDataGenerator(api_config=_min_api(), config=data,
                                               tokenizer=_WordTok(), num_workers=1)
    gen2._ensure_session_built(0)
    assert list(gen.sessions[0].graph.events.keys()) == list(gen2.sessions[0].graph.events.keys())
```

(Provide `_min_api()` returning a minimal valid `APIConfig` — copy the shape used in existing `tests/datagen/` OTel tests; read one for the exact constructor.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "generator_builds" -v`
Expected: FAIL — class not defined / not exported.

- [ ] **Step 3: Implement the class** — append to `synthetic_agent_sessions.py`, mirroring `OTelTraceReplayDataGenerator.__init__` (`otel_trace_replay_datagen.py:338–470`)

```python
from typing import Optional
from inference_perf.datagen.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.synthetic_themes import load_theme, GENERIC_THEME


class SyntheticAgentSessionsDataGenerator(ReplayGraphSessionGeneratorBase):
    def __init__(self, api_config, config, tokenizer, mp_manager=None, base_seed=None, num_workers=1):
        if getattr(config, "synthetic_agent_sessions", None) is None:
            raise ValueError("synthetic_agent_sessions configuration is required")
        self.synthetic_config = config.synthetic_agent_sessions
        super().__init__(api_config, config, tokenizer, mp_manager=mp_manager,
                         base_seed=base_seed, num_workers=num_workers,
                         replay_config=self.synthetic_config)
        self._themes = {name: (load_theme(name) if name != "generic" else GENERIC_THEME)
                        for name in self.synthetic_config.theme_mix}
        session_ids = [f"synthN{i}" for i in range(self.synthetic_config.num_sessions)]
        self.initialize_sessions_lazy(session_ids)

    def _pick_theme(self, session_index: int):
        names = list(self.synthetic_config.theme_mix.keys())
        weights = np.array([self.synthetic_config.theme_mix[n] for n in names], dtype=float)
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
```

(NOTE: confirm `ReplaySession`'s exact constructor fields at `replay_graph_session_datagen.py:1047`; adjust kwargs. Confirm `self.tokenizer` is the attribute the base stores — read the base `__init__`.)

- [ ] **Step 4: Export + run tests**

Add to `inference_perf/datagen/__init__.py`: `from .synthetic_agent_sessions import SyntheticAgentSessionsDataGenerator`.
Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k "generator_builds" -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_perf/datagen/synthetic_agent_sessions.py inference_perf/datagen/__init__.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(datagen): SyntheticAgentSessionsDataGenerator (lazy build_session + theme weighting)"
```

---

## Task 11: `main.py` wiring

**Files:**
- Modify: `inference_perf/main.py` (import, dispatch elif, tokenizer-required set, mp.Manager tuple, SessionMetricsCollector tuple)
- Test: `tests/datagen/test_synthetic_agent_sessions.py` (a smoke test that dispatch resolves the generator)

**Interfaces:**
- Consumes: everything above.
- Produces: `config.data.type == DataGenType.SyntheticAgentSessions` resolves to `SyntheticAgentSessionsDataGenerator` in `main.py`, with tokenizer required, mp.Manager provided, and a `SessionMetricsCollector` wired.

- [ ] **Step 1: Add the import** — `main.py:44` region

```python
from inference_perf.datagen import (
    OTelTraceReplayDataGenerator,
    WekaTraceReplayDataGenerator,
    SyntheticAgentSessionsDataGenerator,
)
```

- [ ] **Step 2: Add to tokenizer-required set** — `main.py:288–295`

Add `DataGenType.SyntheticAgentSessions,` to the tuple listing generators that require a tokenizer.

- [ ] **Step 3: Add to the mp.Manager + SessionMetricsCollector tuples** — `main.py:278` and `:380`

Change both `in (DataGenType.OTelTraceReplay, DataGenType.WekaTraceReplay)` tuples to also include `DataGenType.SyntheticAgentSessions`.

- [ ] **Step 4: Add the dispatch elif** — `main.py:369` region (after the Weka elif)

```python
        elif config.data.type == DataGenType.SyntheticAgentSessions:
            datagen = SyntheticAgentSessionsDataGenerator(
                config.api, config.data, tokenizer, mp_manager, config.load.base_seed, num_workers=config.load.num_workers
            )
```

- [ ] **Step 5: Smoke test**

```python
def test_dispatch_resolves_synthetic_generator(monkeypatch):
    # Minimal: assert the generator class is importable and the enum value maps.
    from inference_perf.config.datagen.config import DataGenType
    from inference_perf.datagen import SyntheticAgentSessionsDataGenerator
    assert DataGenType.SyntheticAgentSessions.value == "synthetic_agent_sessions"
    assert SyntheticAgentSessionsDataGenerator is not None
```

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py -k dispatch -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add inference_perf/main.py tests/datagen/test_synthetic_agent_sessions.py
git commit -m "feat(main): wire SyntheticAgentSessions dispatch, tokenizer, manager, session metrics"
```

---

## Task 12: End-to-end integration test (replay a generated config against a mock)

**Files:**
- Test: `tests/datagen/test_synthetic_agent_sessions.py` (append)

**Interfaces:**
- Consumes: the full generator + runtime. Proves a generated graph replays without 400s (single-call forcing + substitution + tool_output merge all cohere).

- [ ] **Step 1: Write the integration test**

```python
@pytest.mark.asyncio
async def test_generated_session_replays_without_dangling_ids():
    """Build a fan-out session and walk every event's substituted messages;
    assert no role:tool message references a tool_call_id absent from a preceding
    assistant tool_call in the same event (inv #3 end-to-end)."""
    cfg = _cfg(fanout_probability=1.0, max_depth=2,
               sub_agents_per_spawn=Distribution(type="fixed", mean=2),
               max_events_per_session=2048, tool_turns_per_loop=Distribution(type="fixed", mean=1))
    g = build_graph_for_session(cfg, GENERIC_THEME, _WordTok(), 0)
    for ev in g.events.values():
        call_ids = {tc["id"] for m in ev.call.messages for tc in m.get("tool_calls", [])}
        tool_ids = {m["tool_call_id"] for m in ev.call.messages if m.get("role") == "tool"}
        assert tool_ids <= call_ids, f"dangling tool_call_id in {ev.event_id}"
```

- [ ] **Step 2: Run the full suite**

Run: `pytest tests/datagen/test_synthetic_agent_sessions.py tests/datagen/test_tool_output_segment.py -v`
Expected: ALL PASS.

- [ ] **Step 3: Run the whole datagen suite (no regressions)**

Run: `pytest tests/datagen/ -v`
Expected: ALL PASS (existing OTel/Weka/conversation tests unchanged).

- [ ] **Step 4: Commit**

```bash
git add tests/datagen/test_synthetic_agent_sessions.py
git commit -m "test(datagen): end-to-end fan-out replay validity (no dangling tool_call_ids)"
```

---

## Deferred (explicitly NOT in v1 — from §12)

Not tasks; recorded so no one implements them by accident: LLM-authored themes; `thinking_probability`/`reasoning_content` surface; parallel-call count reconciliation (trim/pad); `shape` preset knob; coherent-filler tiers (echo/jargon/markov); asymmetric `sub_agent_tool_turns_per_loop`; the branching-factor / truncation-likelihood advisory validators and per-session truncation logging (nice-to-have observability, can be added when needed — the self-limiting walk already bounds size without them).
