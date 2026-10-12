# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceLab Trace Replay Data Generator.

Replays UW TraceLab coding-agent traces (Claude Code / Codex) where each
JSONL row is one LLM invocation with serving-relevant token accounting::

    input_tokens_total = prefix_tokens + newly_append_tokens

The public trace is sanitized: raw prompts, completions, and tool payloads
are removed, but per-round token counts, tool-call metadata (names,
latencies), event timing, and round ordering are preserved. See
https://tracelab.cs.washington.edu and https://arxiv.org/abs/2606.30560.

Replay model:
- Rows are grouped by ``session_id`` and ordered by ``round_index``.
- Each round becomes one LLM call in a linear session chain. The prompt text
  is synthetically reconstructed so the full message list tokenizes to the
  recorded ``input_tokens_total`` (deterministic corpus slices), and the
  expected output is sized to the recorded ``output_tokens``.
- Inter-round spacing comes from ``timing_events`` timestamps when present
  (the gap between the previous round's last event and this round's first
  event is the recorded think time), falling back to the round's tool
  latency (sum of ``tools[].tool_wall_latency_ms``) otherwise. Gaps are
  capped by ``trace_idle_gap_cap_seconds``.
"""

import gzip
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from inference_perf.config import APIConfig, DataConfig
from inference_perf.datagen.replay.otel_trace_to_replay_graph import RawCall, build_graph
from inference_perf.datagen.replay.otel_trace_utils import _compile_filter
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    ReplaySession,
    ReplayGraphSessionGeneratorBase,
)
from inference_perf.datagen.replay.replay_graph_types import ReplayMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

TRACE_FILE_SUFFIXES = {".json", ".jsonl", ".jsonl.gz", ".gz"}

DEFAULT_HF_DATASET = "UW-SyFI/TraceLab"


class FilterExpressionError(ValueError):
    """Raised when a user-supplied filter expression fails to evaluate."""


class TraceLabToolCall(BaseModel):
    model_config = {"extra": "ignore"}

    tool_name: Optional[str] = None
    tool_wall_latency_ms: Optional[float] = None
    tool_internal_latency_ms: Optional[float] = None


class TraceLabTimingEvent(BaseModel):
    model_config = {"extra": "ignore"}

    event_type: Optional[str] = None
    timestamp: Optional[str] = None


class TraceLabRound(BaseModel):
    """One TraceLab JSONL row (one LLM invocation). Extra fields ignored."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    session_id: str
    round_id: Optional[str] = None
    round_index: int
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens_total: int
    prefix_tokens: int = 0
    newly_append_tokens: int = 0
    output_tokens: int
    tools: List[TraceLabToolCall] = Field(default_factory=list)
    timing_events: List[TraceLabTimingEvent] = Field(default_factory=list)


def _parse_ts(ts: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 timestamp to epoch seconds, None when unparseable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _round_time_bounds(rnd: TraceLabRound) -> Tuple[Optional[float], Optional[float]]:
    """First/last event timestamps of a round, (None, None) when absent."""
    stamps = [_parse_ts(e.timestamp) for e in rnd.timing_events or []]
    present = [s for s in stamps if s is not None]
    if not present:
        return (None, None)
    return (min(present), max(present))


def _read_jsonl_rows(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """Read TraceLab rows from .json / .jsonl / .jsonl.gz files.

    Returns (line_number, row) tuples so validation errors can name the
    file and line. JSON files report line 0.
    """
    suffixes = "".join(path.suffixes)
    opener: Callable[..., Any]
    if suffixes.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    rows: List[Tuple[int, Dict[str, Any]]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return rows
        f.seek(0)
        if path.suffix == ".json" and not str(path).endswith(".jsonl"):
            blob = json.load(f)
            if isinstance(blob, list):
                return [(0, r) for r in blob if isinstance(r, dict)]
            if isinstance(blob, dict):
                return [(0, blob)]
            return rows
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                rows.append((line_no, json.loads(line)))
    return rows


class TraceLabTraceReplayDataGenerator(ReplayGraphSessionGeneratorBase):
    """Data generator that replays TraceLab coding-agent traces."""

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        if not hasattr(config, "tracelab_trace_replay") or config.tracelab_trace_replay is None:
            raise ValueError("tracelab_trace_replay configuration is required for TraceLabTraceReplayDataGenerator")

        self.tracelab_config = config.tracelab_trace_replay

        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            replay_config=self.tracelab_config,
        )

        self.mp_manager = mp_manager
        self.num_workers = max(1, num_workers)
        self.base_seed = base_seed if base_seed is not None else 42

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for TraceLabTraceReplayDataGenerator")

        if self.config and self.config.corpus_file_path:
            corpus_path = Path(self.config.corpus_file_path)
        else:
            corpus_path = Path(__file__).resolve().parents[2] / "assets" / "shakespeare.txt"
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Prompt corpus file not found: {corpus_path}")
        corpus_text = corpus_path.read_text(encoding="utf-8")
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        self._tokenized_corpus = self.tokenizer.get_tokenizer().encode(base_prompt + corpus_text)
        self._corpus_size = len(self._tokenized_corpus)
        if self._corpus_size == 0:
            raise ValueError(f"Prompt corpus is empty: {corpus_path}")

        sessions_by_id = self._load_tracelab_sessions()
        sessions = self._build_sessions(sessions_by_id)
        self.initialize_sessions(sessions)

    @staticmethod
    def _augment_for_filter(session_id: str, rounds: List[TraceLabRound]) -> Dict[str, Any]:
        per_round = [(r.input_tokens_total or 0) + (r.output_tokens or 0) for r in rounds]
        providers = sorted({r.provider for r in rounds if r.provider})
        models = sorted({r.model for r in rounds if r.model})
        return {
            "session_id": session_id,
            "provider": providers[0] if len(providers) == 1 else providers,
            "providers": providers,
            "models": models,
            "max_tokens": max(per_round, default=0),
            "total_tokens": sum(per_round),
            "num_rounds": len(rounds),
            "num_turns": len(rounds),
        }

    def _keep_session(self, augmented: Dict[str, Any], filter_func: Optional[Any]) -> bool:
        if filter_func is None:
            return True
        try:
            return bool(filter_func(augmented))
        except Exception as e:
            raise FilterExpressionError(f"Filter expression failed on session {augmented.get('session_id')!r}: {e!r}") from e

    def _load_tracelab_sessions(self) -> Dict[str, List[TraceLabRound]]:
        """Load rows from all configured sources, grouped by session."""
        filter_func = _compile_filter(self.tracelab_config.filter)
        if filter_func:
            logger.info(f"Using filter expression: {self.tracelab_config.filter}")

        raw_rows: List[Tuple[str, int, Dict[str, Any]]] = []
        cfg = self.tracelab_config

        if cfg.trace_directory:
            trace_dir = Path(cfg.trace_directory)
            if not trace_dir.exists() or not trace_dir.is_dir():
                raise ValueError(f"Trace directory does not exist or is not a directory: {trace_dir}")
            files = sorted(p for p in trace_dir.iterdir() if p.is_file() and "".join(p.suffixes) in TRACE_FILE_SUFFIXES)
            if not files:
                raise ValueError(f"No TraceLab trace files found in {trace_dir}")
            for f in files:
                try:
                    raw_rows.extend((str(f), line_no, blob) for line_no, blob in _read_jsonl_rows(f))
                except Exception as e:
                    logger.error(f"Failed to load trace file {f.name}: {e}")
                    if not cfg.skip_invalid_files:
                        raise
        elif cfg.trace_files:
            for path in cfg.trace_files:
                f = Path(path)
                if not f.is_file():
                    raise ValueError(f"Trace file does not exist: {path}")
                try:
                    raw_rows.extend((str(f), line_no, blob) for line_no, blob in _read_jsonl_rows(f))
                except Exception as e:
                    logger.error(f"Failed to load trace file {f.name}: {e}")
                    if not cfg.skip_invalid_files:
                        raise
        elif cfg.hf_dataset_path:
            hf_rows = self._load_hf_rows(cfg.hf_dataset_path, cfg.num_dataset_entries, cfg.skip_invalid_files)
            raw_rows.extend((f"hf:{cfg.hf_dataset_path}", line_no, blob) for line_no, blob in enumerate(hf_rows, start=1))
        else:
            raise ValueError("Either trace_directory, trace_files, or hf_dataset_path must be provided")

        grouped: Dict[str, List[TraceLabRound]] = {}
        n_invalid = 0
        for source, line_no, blob in raw_rows:
            try:
                rnd = TraceLabRound.model_validate(blob)
            except Exception as e:
                logger.error(f"Failed to validate TraceLab row {source}:{line_no}: {e}")
                if not cfg.skip_invalid_files:
                    raise ValueError(f"Invalid TraceLab row at {source}:{line_no}: {e}") from e
                n_invalid += 1
                continue
            # session_id is required now (no default), so validation ensures it exists.
            key = rnd.session_id  # session_id already includes provider prefix in real data
            grouped.setdefault(key, []).append(rnd)

        # Order rounds and apply the session filter.
        kept: Dict[str, List[TraceLabRound]] = {}
        n_filtered = 0
        for session_id, rounds in grouped.items():
            rounds.sort(key=lambda r: (r.round_index, r.round_id or ""))
            # Re-assign dense round indices so downstream timing is stable even
            # when the trace carries sparse/duplicate indices.
            for i, r in enumerate(rounds):
                r.round_index = i
            augmented = self._augment_for_filter(session_id, rounds)
            try:
                if not self._keep_session(augmented, filter_func):
                    n_filtered += 1
                    continue
            except FilterExpressionError:
                raise
            except Exception as e:
                logger.error(f"Failed to filter session {session_id}: {e}")
                if not cfg.skip_invalid_files:
                    raise
                continue
            kept[session_id] = rounds

        if filter_func:
            logger.info(f"Filter applied: {len(kept)} sessions kept, {n_filtered} rejected of {len(grouped)} scanned")
        if n_invalid:
            logger.warning(f"Skipped {n_invalid} invalid rows (skip_invalid_files=True)")
        if not kept:
            raise ValueError("No valid TraceLab sessions found")
        logger.info(f"Loaded {len(kept)} TraceLab sessions ({sum(len(v) for v in kept.values())} rounds)")
        return kept

    def _load_hf_rows(self, hf_path: Any, limit: int, skip_invalid: bool) -> List[Dict[str, Any]]:
        """Load TraceLab rounds from HF and join tools/timing_events tables.

        The UW-SyFI/TraceLab dataset has multiple configs:
        - "rounds" (default): one row per LLM invocation
        - "tools": one row per tool call, with round_pk foreign key
        - "timing_events": one row per timing event, with round_pk foreign key

        We load the rounds table and join tools/timing_events by round_pk.
        The limit applies to SESSIONS, not rows.
        """
        from datasets import load_dataset

        if isinstance(hf_path, dict):
            path = hf_path.get("path", DEFAULT_HF_DATASET)
            kwargs = {k: v for k, v in hf_path.items() if k != "path"}
        else:
            path = hf_path or DEFAULT_HF_DATASET
            kwargs = {}

        logger.info(f"Downloading TraceLab traces from Hugging Face dataset: {path}")

        # Load rounds table (default config)
        try:
            ds_rounds = load_dataset(path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset {path}: {e}") from e

        if hasattr(ds_rounds, "keys"):
            split = kwargs.get("split", "train")
            ds_rounds = ds_rounds[split] if split in ds_rounds.keys() else ds_rounds[list(ds_rounds.keys())[0]]

        # Load tools table if available
        tools_by_round: Dict[str, List[Dict[str, Any]]] = {}
        try:
            ds_tools = load_dataset(path, name="tools", **kwargs)
            if hasattr(ds_tools, "keys"):
                split = kwargs.get("split", "train")
                ds_tools = ds_tools[split] if split in ds_tools.keys() else ds_tools[list(ds_tools.keys())[0]]
            for t in ds_tools:
                if isinstance(t, dict):
                    rpk = t.get("round_pk")
                    if rpk:
                        tools_by_round.setdefault(rpk, []).append(
                            {
                                "tool_name": t.get("tool_name"),
                                "tool_wall_latency_ms": t.get("tool_wall_latency_ms"),
                                "tool_internal_latency_ms": t.get("tool_internal_latency_ms"),
                            }
                        )
        except Exception as e:
            logger.warning(f"Could not load tools table from {path}: {e}")

        # Load timing_events table if available
        timing_by_round: Dict[str, List[Dict[str, Any]]] = {}
        try:
            ds_timing = load_dataset(path, name="timing_events", **kwargs)
            if hasattr(ds_timing, "keys"):
                split = kwargs.get("split", "train")
                ds_timing = ds_timing[split] if split in ds_timing.keys() else ds_timing[list(ds_timing.keys())[0]]
            for te in ds_timing:
                if isinstance(te, dict):
                    rpk = te.get("round_pk")
                    if rpk:
                        timing_by_round.setdefault(rpk, []).append(te)
        except Exception as e:
            logger.warning(f"Could not load timing_events table from {path}: {e}")

        # Build rows by joining tools/timing, stopping at session limit.
        rows: List[Dict[str, Any]] = []
        session_ids_seen: Set[str] = set()

        for rnd in ds_rounds:
            if not isinstance(rnd, dict):
                if not skip_invalid:
                    raise ValueError(f"Unexpected row type {type(rnd)} in Hugging Face dataset {path}")
                continue

            session_id = rnd.get("session_id", "")
            if session_id and session_id not in session_ids_seen:
                session_ids_seen.add(session_id)
                if len(session_ids_seen) > limit:
                    break

            # Join tools and timing_events by round_pk
            rpk = rnd.get("round_pk", "")
            if rpk and rpk in tools_by_round:
                rnd["tools"] = tools_by_round[rpk]
            if rpk and rpk in timing_by_round:
                rnd["timing_events"] = timing_by_round[rpk]

            rows.append(dict(rnd))

        logger.info(f"Loaded {len(rows)} rounds across {len(session_ids_seen)} sessions from HF")
        return rows

    def _stable_offset(self, session_id: str, round_index: int, salt: int) -> int:
        """Return a deterministic offset for corpus slicing.

        Uses MD5 of session_id + round_index + salt for cross-process stability.
        """
        h = hashlib.md5(f"{session_id}:{round_index}:{salt}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    def _build_model_map(self, models: List[str]) -> Dict[str, str]:
        configured = self.tracelab_config.model_mapping or {}
        if self.tracelab_config.use_static_model:
            return {m: self.tracelab_config.static_model_name for m in set(models)}
        return configured

    def _text_for_tokens(self, n_tokens: int, offset: int) -> str:
        """Decode a deterministic corpus slice of ~n_tokens tokens to text."""
        if n_tokens <= 0:
            return ""
        assert self.tokenizer is not None
        start = offset % self._corpus_size
        tokens = self._tokenized_corpus[start : start + n_tokens]
        if len(tokens) < n_tokens:
            tokens = tokens + self._tokenized_corpus[: n_tokens - len(tokens)]
        decoded = self.tokenizer.get_tokenizer().decode(list(tokens))
        return decoded if isinstance(decoded, str) else " ".join(decoded)

    def _round_gap_ms(self, prev: TraceLabRound, curr: TraceLabRound) -> Optional[int]:
        """Recorded think time before ``curr``: its first event minus the
        previous round's last event, in milliseconds.

        Returns None when either round lacks usable ``timing_events``
        timestamps so the caller can fall back to tool latency.
        """
        _, prev_end = _round_time_bounds(prev)
        curr_start, _ = _round_time_bounds(curr)
        if prev_end is None or curr_start is None:
            return None
        return max(int((curr_start - prev_end) * 1000.0), 0)

    @staticmethod
    def _tool_latency_ms(rnd: TraceLabRound) -> int:
        total = 0
        for t in rnd.tools or []:
            lat = t.tool_wall_latency_ms if t.tool_wall_latency_ms is not None else t.tool_internal_latency_ms
            if lat:
                total += max(int(lat), 0)
        return total

    def _reconstruct_raw_calls(self, session_id: str, rounds: List[TraceLabRound]) -> List[RawCall]:
        model_map = self._build_model_map([r.model or "" for r in rounds if r.model])
        cap_ms = int(self.tracelab_config.trace_idle_gap_cap_seconds * 1000.0)

        calls: List[RawCall] = []
        history: List[ReplayMessage] = []
        t_ms = 0

        for i, rnd in enumerate(rounds):
            in_tokens = max(int(rnd.input_tokens_total or 0), 0)
            out_tokens = max(int(rnd.output_tokens or 0), 0)

            # Spacing before this round: the recorded think time from
            # timing_events when present, else the previous round's tool
            # latency (those tools ran after that round's output).
            if i > 0 and not self.tracelab_config.ignore_trace_delays:
                gap = self._round_gap_ms(rounds[i - 1], rnd)
                if gap is None:
                    gap = self._tool_latency_ms(rounds[i - 1])
                t_ms += min(gap, cap_ms)

            # Determine how many tokens the current history has.
            history_tokens = 0
            if self.tokenizer and history:
                full_history_text = "".join(m.text for m in history)
                history_tokens = len(self.tokenizer.get_tokenizer().encode(full_history_text))

            # The user message should make total tokens ≈ in_tokens.
            need_user_tokens = max(in_tokens - history_tokens, 1)
            user_text = self._text_for_tokens(
                need_user_tokens,
                offset=self._stable_offset(session_id, i, 7919),
            )

            messages = list(history) + [ReplayMessage(role="user", text=user_text or " ")]

            # Verify the built prompt tokenizes to approximately input_tokens_total.
            if self.tokenizer:
                built_prompt = "".join(m.text for m in messages)
                built_tokens = len(self.tokenizer.get_tokenizer().encode(built_prompt))
                if abs(built_tokens - in_tokens) > max(4, in_tokens * 0.1):
                    logger.warning(
                        f"Session {session_id} round {i}: built prompt {built_tokens} tokens "
                        f"vs recorded {in_tokens} (history={history_tokens}, need_user={need_user_tokens})"
                    )

            out_text = self._text_for_tokens(
                out_tokens,
                offset=self._stable_offset(session_id, i, 104729),
            )
            out_message = ReplayMessage(role="assistant", text=out_text or " ")
            model = model_map.get(rnd.model or "", rnd.model or "tracelab-model")

            calls.append(
                RawCall(
                    call_id=f"round_{i}",
                    trace_id=session_id,
                    t_start_ms=t_ms,
                    t_end_ms=t_ms,
                    model=model,
                    messages=messages,
                    out_message=out_message,
                    prompt_tokens=in_tokens,
                    completion_tokens=out_tokens,
                    temperature=0.0,
                    max_tokens_recorded=out_tokens,
                )
            )

            # Grow transcript for next round.
            history = history + [ReplayMessage(role="user", text=user_text or " "), out_message]

        return calls

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        """Make a session id safe for qualified event ids.

        The replay runtime splits qualified event ids on the first ':'
        to recover the session, so a raw TraceLab id like 'claude:<uuid>'
        would corrupt event resolution. Only the runtime id is rewritten;
        the original stays in ``source_id``.
        """
        return session_id.replace(":", "_")

    def _build_sessions(self, sessions_by_id: Dict[str, List[TraceLabRound]]) -> List[ReplaySession]:
        sessions: List[ReplaySession] = []
        for trace_index, session_id in enumerate(sorted(sessions_by_id.keys())):
            rounds = sessions_by_id[session_id]
            try:
                raw_calls = self._reconstruct_raw_calls(session_id, rounds)
                if not raw_calls:
                    continue
                graph = build_graph(raw_calls, source_file=f"tracelab_trace_{session_id}")
                sessions.append(
                    ReplaySession(
                        session_id=f"tracelabtrace{trace_index}_{self._safe_session_id(session_id)}",
                        source_id=session_id,
                        session_index=trace_index,
                        graph=graph,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to process TraceLab session {session_id}: {e}")
                if not self.tracelab_config.skip_invalid_files:
                    raise
        if not sessions:
            raise ValueError("No valid TraceLab replay sessions built")
        random.seed(self.base_seed)
        random.shuffle(sessions)
        logger.info(f"Built {len(sessions)} TraceLab replay sessions")
        return sessions
