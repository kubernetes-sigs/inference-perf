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
- Prompt text is built in token space: each round keeps the token ids it sent,
  and the next round reuses the leading ``prefix_tokens`` of them verbatim.
  Encoding/decoding that shared run happens without adding special tokens, so
  a BOS-adding tokenizer cannot shift every later token by one position per
  round and destroy server-side prefix-cache block alignment.
- Inter-round spacing comes from ``timing_events``: a round is submitted when
  its last input event (``user_message``/``tool_result``) arrives and completes
  at its last model-output event (``reasoning``/``text``/``tool_call``).
  ``usage_report`` rows are written after the tool ran, so they are excluded
  from both bounds. Rounds without usable timestamps fall back to the previous
  round's tool wall time (overlapping parallel tool intervals are merged, not
  summed). Gaps are capped by ``trace_idle_gap_cap_seconds``, or replayed
  verbatim when that cap is disabled.
"""

import gzip
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from inference_perf.config import APIConfig, DataConfig, TraceLabTraceReplayConfig
from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
    DEPENDENCY_TYPE,
    RawCall,
    build_graph,
    find_predecessors_by_text_matching,
)
from inference_perf.datagen.replay.otel_trace_utils import _compile_filter
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    ReplaySession,
    ReplayGraphSessionGeneratorBase,
)
from inference_perf.datagen.replay.replay_graph_types import ComplexReplayMessage, ReplayMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

# Matched with str.endswith so a dotted stem (tracelab_v0.0.2.jsonl) is still
# recognised: joining Path.suffixes yields ".0.0.2.jsonl" for that name.
TRACE_FILE_SUFFIXES = (".jsonl", ".jsonl.gz", ".json", ".json.gz")

# TraceLab event types (see the dataset's README): the LLM request is submitted
# when the last input event arrives and finishes at the last model-output event.
INPUT_EVENT_TYPES = frozenset({"user_message", "tool_result"})
OUTPUT_EVENT_TYPES = frozenset({"reasoning", "text", "tool_call"})

# How many tokens the shared prefix may be trimmed by when decode→encode does
# not reproduce it. Keeps prefix reuse O(1) re-encodes instead of O(prefix_len).
_MAX_SHARED_PREFIX_TRIM = 8

DEFAULT_HF_DATASET = "UW-SyFI/TraceLab"


class FilterExpressionError(ValueError):
    """Raised when a user-supplied filter expression fails to evaluate."""


class TraceLabToolCall(BaseModel):
    model_config = {"extra": "ignore"}

    tool_name: Optional[str] = None
    tool_wall_latency_ms: Optional[float] = None
    tool_internal_latency_ms: Optional[float] = None
    # Wall-clock window of the tool execution. Used to merge overlapping
    # intervals when parallel tools ran concurrently.
    emitted_at: Optional[str] = None
    result_at: Optional[str] = None


class TraceLabTimingEvent(BaseModel):
    model_config = {"extra": "ignore"}

    event_type: Optional[str] = None
    timestamp: Optional[str] = None


class TraceLabRound(BaseModel):
    """One TraceLab JSONL row (one LLM invocation). Extra fields ignored."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    session_id: str
    round_id: Optional[str] = None
    round_pk: Optional[Union[int, str]] = None  # HF join key for tool_calls/timing_events
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
    """Recorded (submission, completion) time of a round.

    The LLM request for a round is submitted when its last *input* event
    arrives: ``user_message`` on a fresh turn, or ``tool_result`` once the last
    of several parallel tools has returned. Falling back to ``min`` of the
    model-output events keeps rounds that recorded no input event usable.

    The round completes at its last *model-output* event. ``usage_report`` rows
    (Codex writes them after the tool call has finished) describe the response
    that was already produced, so including them would push ``prev_end`` past
    the tool execution and collapse the measured think time to zero.

    Returns ``(None, None)`` when the round carries no usable event.
    """
    events = [(e.event_type or "", _parse_ts(e.timestamp)) for e in rnd.timing_events or []]
    timed = [(event_type, stamp) for event_type, stamp in events if stamp is not None]
    inputs = [stamp for event_type, stamp in timed if event_type in INPUT_EVENT_TYPES]
    outputs = [stamp for event_type, stamp in timed if event_type in OUTPUT_EVENT_TYPES]
    if not inputs and not outputs:
        return (None, None)
    start = max(inputs) if inputs else min(outputs)
    end = max(outputs) if outputs else max(inputs)
    return (start, end)


def _iter_jsonl_rows(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield TraceLab rows from .json / .jsonl / .jsonl.gz files lazily.

    Yields (line_number, row) tuples so validation errors can name the file
    and line. JSON files report line 0. Lazily so callers can stop reading
    once the requested number of sessions has been collected.
    """
    name = path.name
    opener: Callable[..., Any]
    if name.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        # A .json.gz file is still a JSON array; Path.suffix is ".gz" for it, so
        # the extension test has to look at the whole name.
        if name.endswith((".json", ".json.gz")):
            blob = json.load(f)
            if isinstance(blob, list):
                for r in blob:
                    if isinstance(r, dict):
                        yield 0, r
            elif isinstance(blob, dict):
                yield 0, blob
            return
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


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
        # Tokenize without special tokens: the corpus is sliced and wrapped
        # inside prompts, so an embedded BOS/EOS would land mid-prompt.
        self._tokenized_corpus = self.tokenizer.get_tokenizer().encode(base_prompt + corpus_text, add_special_tokens=False)
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
        """Load sessions, stopping once ``num_dataset_entries`` sessions are kept.

        Both local files and the Hugging Face ``rounds`` table are streamed, so
        requesting a small number of sessions never materializes the full
        corpus. Rows for one session must be contiguous in the source; TraceLab
        JSONL exports and the HF ``rounds`` table both group rows by session.
        """
        cfg = self.tracelab_config
        filter_func = _compile_filter(cfg.filter)
        if filter_func:
            logger.info(f"Using filter expression: {cfg.filter}")

        blocks: Iterator[Tuple[str, List[TraceLabRound], int]]
        if cfg.hf_dataset_path:
            blocks = self._iter_session_blocks(self._iter_hf_rounds(cfg.hf_dataset_path))
        else:
            blocks = self._iter_session_blocks(self._iter_local_rows(cfg))

        kept: Dict[str, List[TraceLabRound]] = {}
        n_filtered = 0
        n_invalid = 0
        limit = cfg.num_dataset_entries

        for session_id, rounds, invalid in blocks:
            n_invalid += invalid
            if not rounds:
                continue
            if not self._keep_session(self._augment_for_filter(session_id, rounds), filter_func):
                n_filtered += 1
                continue
            kept[session_id] = rounds
            if len(kept) >= limit:
                logger.info(f"Stopped reading TraceLab input after {len(kept)} sessions (num_dataset_entries={limit})")
                break

        # Join the HF aux tables only after selection so their rows are indexed
        # for the kept rounds, not for the whole dataset.
        if cfg.hf_dataset_path and kept:
            self._attach_hf_aux_tables(kept, cfg.hf_dataset_path)

        if filter_func:
            logger.info(f"Filter applied: {len(kept)} sessions kept, {n_filtered} rejected")
        if n_invalid:
            logger.warning(f"Skipped {n_invalid} invalid rows (skip_invalid_files=True)")
        if not kept:
            raise ValueError("No valid TraceLab sessions found")
        logger.info(f"Loaded {len(kept)} TraceLab sessions ({sum(len(v) for v in kept.values())} rounds)")
        return kept

    def _iter_local_rows(self, cfg: TraceLabTraceReplayConfig) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
        """Yield rows from local trace files lazily so callers can stop early."""
        if cfg.trace_directory:
            trace_dir = Path(cfg.trace_directory)
            if not trace_dir.exists() or not trace_dir.is_dir():
                raise ValueError(f"Trace directory does not exist or is not a directory: {trace_dir}")
            files = sorted(p for p in trace_dir.iterdir() if p.is_file() and p.name.endswith(TRACE_FILE_SUFFIXES))
            if not files:
                raise ValueError(f"No TraceLab trace files found in {trace_dir}")
        else:
            files = []
            for path in cfg.trace_files or []:
                f = Path(path)
                if not f.is_file():
                    raise ValueError(f"Trace file does not exist: {path}")
                files.append(f)

        for f in files:
            try:
                for line_no, blob in _iter_jsonl_rows(f):
                    yield str(f), line_no, blob
            except Exception as e:
                logger.error(f"Failed to load trace file {f.name}: {e}")
                if not cfg.skip_invalid_files:
                    raise

    def _iter_session_blocks(
        self, rows: Iterator[Tuple[str, int, Dict[str, Any]]]
    ) -> Iterator[Tuple[str, List[TraceLabRound], int]]:
        """Group contiguous rows into validated, ordered sessions.

        Yields ``(session_id, rounds, invalid_rows)``. A session is yielded as
        soon as the next session starts, so the consumer can stop reading
        without waiting for the rest of the corpus.

        Contiguity is what makes that early stop safe, so a session_id that
        reappears after another session is a hard error instead of silently
        replacing (and dropping) the rounds already collected for it.
        """
        cfg = self.tracelab_config
        current_id: Optional[str] = None
        rounds: List[TraceLabRound] = []
        n_invalid = 0
        seen: Set[str] = set()

        for source, line_no, blob in rows:
            try:
                rnd = TraceLabRound.model_validate(blob)
            except Exception as e:
                logger.error(f"Failed to validate TraceLab row {source}:{line_no}: {e}")
                if not cfg.skip_invalid_files:
                    raise ValueError(f"Invalid TraceLab row at {source}:{line_no}: {e}") from e
                n_invalid += 1
                continue

            if rnd.session_id != current_id:
                if current_id is not None:
                    yield current_id, self._finalize_block(rounds), n_invalid
                if rnd.session_id in seen:
                    raise ValueError(
                        f"TraceLab session {rnd.session_id!r} reappears at {source}:{line_no} after another "
                        "session. Streaming early-stop requires every row of a session to be contiguous; "
                        "sort the JSONL by session_id, or export one file per session."
                    )
                seen.add(rnd.session_id)
                current_id = rnd.session_id
                rounds = []
                n_invalid = 0
            rounds.append(rnd)

        if current_id is not None:
            yield current_id, self._finalize_block(rounds), n_invalid

    @staticmethod
    def _finalize_block(rounds: List[TraceLabRound]) -> List[TraceLabRound]:
        """Order a session's rounds and renumber them densely.

        Real traces carry occasional duplicate ``(session_id, round_index)``
        pairs and rows whose ``round_index`` disagrees with their timestamps.
        When every round has a submission timestamp, the timestamps decide
        (tie-breaking on round_index/round_id); otherwise round_index does.
        """
        if not rounds:
            return rounds
        stamped: List[Tuple[Optional[float], TraceLabRound]] = [(_round_time_bounds(r)[0], r) for r in rounds]

        def stamped_key(item: Tuple[Optional[float], TraceLabRound]) -> Tuple[float, int, str]:
            start, rnd = item
            assert start is not None
            return (start, rnd.round_index, rnd.round_id or "")

        if all(start is not None for start, _ in stamped):
            ordered = [rnd for _, rnd in sorted(stamped, key=stamped_key)]
        else:
            ordered = sorted(rounds, key=lambda r: (r.round_index, r.round_id or ""))
        # Re-assign dense round indices so downstream timing is stable even
        # when the trace carries sparse/duplicate indices.
        for i, r in enumerate(ordered):
            r.round_index = i
        return ordered

    @staticmethod
    def _hf_source(hf_path: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(hf_path, dict):
            return hf_path.get("path", DEFAULT_HF_DATASET), {k: v for k, v in hf_path.items() if k != "path"}
        return hf_path or DEFAULT_HF_DATASET, {}

    @staticmethod
    def _as_split(ds: Any, kwargs: Dict[str, Any]) -> Any:
        if hasattr(ds, "keys"):
            split = kwargs.get("split", "train")
            keys = list(ds.keys())
            return ds[split] if split in keys else ds[keys[0]]
        return ds

    def _load_hf_table(self, path: str, kwargs: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Load one TraceLab table, streaming by default.

        Without ``streaming=True`` datasets downloads and Arrow-builds the whole
        table (~665k rounds, ~744k tool_calls, ~2.69M timing_events) before the
        first row is yielded, which defeats the session cap and the early stop.
        Callers can opt back into full local caching with ``streaming=False`` in
        ``hf_dataset_path``.
        """
        from datasets import load_dataset

        call_kwargs = dict(kwargs)
        streaming = call_kwargs.pop("streaming", True)
        if name is not None:
            call_kwargs["name"] = name
        return self._as_split(load_dataset(path, streaming=streaming, **call_kwargs), call_kwargs)

    def _iter_hf_rounds(self, hf_path: Any) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
        """Yield the HF ``rounds`` table row by row, without loading aux tables.

        The UW-SyFI/TraceLab dataset has multiple configs: the default
        ("rounds") table has one row per LLM invocation, "tool_calls" one row
        per tool call, and "timing_events" one row per timing event, both with
        a ``round_pk`` foreign key into rounds.
        """
        path, kwargs = self._hf_source(hf_path)
        logger.info(f"Downloading TraceLab traces from Hugging Face dataset: {path}")
        try:
            ds_rounds = self._load_hf_table(path, kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset {path}: {e}") from e

        source = f"hf:{path}"
        for idx, rnd in enumerate(ds_rounds, start=1):
            if not isinstance(rnd, dict):
                if not self.tracelab_config.skip_invalid_files:
                    raise ValueError(f"Unexpected row type {type(rnd)} in Hugging Face dataset {path}")
                continue
            yield source, idx, rnd

    def _attach_hf_aux_tables(self, kept: Dict[str, List[TraceLabRound]], hf_path: Any) -> None:
        """Join tool_calls/timing_events onto the kept rounds only.

        Indexing just the kept ``round_pk`` values avoids building dictionaries
        over the full aux tables (hundreds of thousands to millions of rows).
        """
        round_pks = {str(r.round_pk) for rounds in kept.values() for r in rounds if r.round_pk is not None}
        if not round_pks:
            return

        path, kwargs = self._hf_source(hf_path)

        def _aux_rows(name: str) -> Iterator[Dict[str, Any]]:
            try:
                table = self._load_hf_table(path, kwargs, name=name)
            except Exception as e:
                msg = (
                    f"Could not load '{name}' table from {path}: {e}. "
                    "Rounds will replay without it (tool-latency waits or "
                    "timestamp gaps from that table will be 0)."
                )
                if not self.tracelab_config.skip_invalid_files:
                    raise ValueError(msg) from e
                logger.warning(msg)
                return
            for row in table:
                if isinstance(row, dict):
                    yield row

        tools_by_round: Dict[str, List[TraceLabToolCall]] = {}
        for t in _aux_rows("tool_calls"):
            rpk = t.get("round_pk")
            if rpk is not None and str(rpk) in round_pks:
                tools_by_round.setdefault(str(rpk), []).append(
                    TraceLabToolCall.model_validate(
                        {
                            "tool_name": t.get("tool_name"),
                            "tool_wall_latency_ms": t.get("tool_wall_latency_ms"),
                            "tool_internal_latency_ms": t.get("tool_internal_latency_ms"),
                            "emitted_at": t.get("emitted_at"),
                            "result_at": t.get("result_at"),
                        }
                    )
                )

        timing_by_round: Dict[str, List[TraceLabTimingEvent]] = {}
        for te in _aux_rows("timing_events"):
            rpk = te.get("round_pk")
            if rpk is not None and str(rpk) in round_pks:
                timing_by_round.setdefault(str(rpk), []).append(
                    TraceLabTimingEvent.model_validate({"event_type": te.get("event_type"), "timestamp": te.get("timestamp")})
                )

        for rounds in kept.values():
            for r in rounds:
                if r.round_pk is None:
                    continue
                key = str(r.round_pk)
                if key in tools_by_round:
                    r.tools = tools_by_round[key]
                if key in timing_by_round:
                    r.timing_events = timing_by_round[key]

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

    def _encode_text(self, text: str) -> List[int]:
        """Tokenize prompt text the way the server will see it.

        Special tokens are never added: prompt text is concatenated across
        rounds, so a per-call ``<bos>`` would shift every later token one
        position to the right on each round and misalign every fixed-size
        KV-cache block from block 0 onward.
        """
        assert self.tokenizer is not None
        return list(self.tokenizer.get_tokenizer().encode(text, add_special_tokens=False))

    def _decode_ids(self, ids: List[int]) -> str:
        """Decode token ids back to wire text, dropping any special tokens."""
        assert self.tokenizer is not None
        decoded = self.tokenizer.get_tokenizer().decode(list(ids), skip_special_tokens=True)
        return decoded if isinstance(decoded, str) else " ".join(decoded)

    def _text_for_tokens(self, n_tokens: int, offset: int) -> str:
        """Decode a deterministic corpus slice of exactly ``n_tokens`` tokens.

        The corpus is repeated as many times as needed instead of wrapping at
        most once, so a small custom ``corpus_file_path`` still fills a long
        TraceLab prompt rather than silently under-sizing it.
        """
        if n_tokens <= 0:
            return ""
        corpus = self._tokenized_corpus
        size = len(corpus)
        ids: List[int] = []
        idx = offset % size
        while len(ids) < n_tokens:
            take = min(n_tokens - len(ids), size - idx)
            ids.extend(corpus[idx : idx + take])
            idx = (idx + take) % size
        return self._decode_ids(ids)

    def _prefix_text(self, prev_ids: List[int], requested: int) -> Tuple[str, List[int]]:
        """Decode the leading ``requested`` tokens of the previous prompt.

        Decode→encode is not idempotent at BPE boundaries, so the shared run is
        trimmed until re-encoding the decoded text reproduces the previous ids
        exactly. Both the text and those ids are returned so the caller can
        check that the new prompt really starts with them — that prefix is what
        the server's prefix cache compares.

        Backing off a few tokens covers the usual mismatch (the cut lands inside
        a byte fragment or a merged token). The trim is capped: hunting further
        would cost O(prefix_len) full re-encodes, so a run that never round-trips
        shares nothing and lets the round be a plain cache miss.
        """
        n = min(requested, len(prev_ids))
        for _ in range(_MAX_SHARED_PREFIX_TRIM):
            if n <= 0:
                break
            text = self._decode_ids(prev_ids[:n])
            ids = self._encode_text(text)
            if ids == prev_ids[:n]:
                return text, ids
            n -= 1
        return "", []

    def _fit_tail(
        self,
        session_id: str,
        round_index: int,
        shared_text: str,
        shared_count: int,
        target_total: int,
    ) -> Tuple[str, List[int]]:
        """Append a fresh corpus tail so the whole prompt has ``target_total`` tokens.

        Decode/re-encode is not idempotent at BPE boundaries either, so the tail
        length is corrected from the measured error. The loop is bounded to
        three measurements; the previous version tried up to 33 candidates and
        re-tokenized the entire prompt each time, which cost O(rounds ×
        context_len) on long sessions.
        """
        need = max(target_total - shared_count, 0)
        base = self._stable_offset(session_id, round_index, 7919)
        best_text = ""
        best_ids: List[int] = []
        best_diff: Optional[int] = None
        for attempt in range(3):
            prompt_text = shared_text + self._text_for_tokens(need, offset=base + attempt * 131071)
            prompt_ids = self._encode_text(prompt_text)
            diff = target_total - len(prompt_ids)
            if best_diff is None or abs(diff) < best_diff:
                best_text, best_ids, best_diff = prompt_text, prompt_ids, abs(diff)
            if diff == 0:
                break
            need = max(need + diff, 0)
        return best_text, best_ids

    def _build_prompt(
        self,
        session_id: str,
        round_index: int,
        prev_ids: List[int],
        prefix_tokens: int,
        target_total: int,
    ) -> Tuple[str, List[int], int]:
        """Build one round's prompt, sharing the recorded prefix with the last one.

        Returns ``(text, ids, shared_count)``. ``ids`` is what the tokenizer
        produces for ``text``, so ``shared_count`` is the number of leading
        tokens round i shares with round i-1, i.e. the server-side
        prefix-cache hit width.
        """
        requested = min(max(prefix_tokens, 0), len(prev_ids), max(target_total, 0))
        attempt = 0
        while requested > 0 and attempt < 3:
            shared_text, shared_ids = self._prefix_text(prev_ids, requested)
            if not shared_ids:
                break
            prompt_text, prompt_ids = self._fit_tail(session_id, round_index, shared_text, len(shared_ids), target_total)
            if prompt_ids[: len(shared_ids)] == shared_ids:
                return prompt_text, prompt_ids, len(shared_ids)
            # The seam merged the last shared token with the first new one;
            # share a little less and retry.
            requested = min(len(shared_ids) - 1, requested - 1)
            attempt += 1
        prompt_text, prompt_ids = self._fit_tail(session_id, round_index, "", 0, target_total)
        return prompt_text, prompt_ids, 0

    def _round_gap_ms(self, prev: TraceLabRound, curr: TraceLabRound) -> Optional[int]:
        """Recorded think time before ``curr``: its submission time minus the
        previous round's completion time, in milliseconds.

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
        """Wall time the round's tools occupied before the next round started.

        Parallel tools run concurrently, so overlapping ``[emitted_at,
        result_at]`` intervals are merged instead of summed: N concurrent 5s
        tools cost 5s, not 5×N. Tools that lack both timestamps fall back to
        their recorded latency, which is added on top of the merged total.
        """
        intervals: List[Tuple[float, float]] = []
        serial_ms = 0.0
        for t in rnd.tools or []:
            emitted = _parse_ts(t.emitted_at)
            result = _parse_ts(t.result_at)
            if emitted is not None and result is not None and result >= emitted:
                intervals.append((emitted, result))
                continue
            lat = t.tool_wall_latency_ms if t.tool_wall_latency_ms is not None else t.tool_internal_latency_ms
            if lat:
                serial_ms += max(float(lat), 0.0)

        merged_ms = 0.0
        if intervals:
            intervals.sort()
            cur_start, cur_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= cur_end:
                    cur_end = max(cur_end, end)
                else:
                    merged_ms += (cur_end - cur_start) * 1000.0
                    cur_start, cur_end = start, end
            merged_ms += (cur_end - cur_start) * 1000.0
        return max(int(merged_ms + serial_ms), 0)

    def _reconstruct_raw_calls(self, session_id: str, rounds: List[TraceLabRound]) -> List[RawCall]:
        model_map = self._build_model_map([r.model or "" for r in rounds if r.model])
        cap_seconds = self.tracelab_config.trace_idle_gap_cap_seconds
        cap_ms: Optional[int] = int(cap_seconds * 1000.0) if cap_seconds is not None and cap_seconds > 0 else None

        calls: List[RawCall] = []
        # Token ids of the prompt this session sent last round. The next round
        # reuses a leading run of them (``prefix_tokens``) and drops the rest
        # (compaction). Keeping ids instead of re-encoding text avoids the
        # per-round BOS that used to shift every later token by one position.
        prev_ids: List[int] = []
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
                t_ms += gap if cap_ms is None else min(gap, cap_ms)

            prompt_text, prompt_ids, shared = self._build_prompt(
                session_id=session_id,
                round_index=i,
                prev_ids=prev_ids,
                prefix_tokens=int(rnd.prefix_tokens or 0),
                target_total=in_tokens,
            )
            if not prompt_text:
                # Empty prompt (recorded zero input tokens): send a single
                # space so the request still carries one user message.
                prompt_text = " "
                prompt_ids = self._encode_text(prompt_text)
            messages = [ReplayMessage(role="user", text=prompt_text)]

            # Verify the built prompt tokenizes to the recorded size.
            if len(prompt_ids) != in_tokens:
                logger.warning(
                    f"Session {session_id} round {i}: built prompt {len(prompt_ids)} tokens "
                    f"vs recorded {in_tokens} (shared={shared})"
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
                    # A recorded zero would reach the client as max_tokens=0
                    # and be replaced by the client-level default; use this
                    # replay's own default instead.
                    max_tokens_recorded=out_tokens or self.tracelab_config.default_max_tokens,
                )
            )

            prev_ids = prompt_ids

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
                graph = build_graph(
                    raw_calls,
                    source_file=f"tracelab_trace_{session_id}",
                    predecessor_finder=_find_tracelab_predecessors,
                )
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
        # Shuffle with a local RNG so importing/instantiating this generator
        # never mutates the process-global random state.
        random.Random(self.base_seed).shuffle(sessions)
        logger.info(f"Built {len(sessions)} TraceLab replay sessions")
        return sessions


def _find_tracelab_predecessors(
    calls: List[RawCall],
) -> Tuple[List[Dict[int, DEPENDENCY_TYPE]], Dict[Tuple[str, str], List[int]]]:
    """Linear, semantics-identical predecessor finder for TraceLab calls.

    Each reconstructed round is a single user message, so the generic
    text-matching finder can never find a causal edge (it only looks for a
    predecessor's output inside a successor's *assistant* messages) and always
    falls back to the closest earlier call that already finished. That fallback
    is ``i - 1`` here because calls are emitted in order with non-overlapping
    timestamps, so the identical edges can be produced in O(n) instead of
    O(n²) string scans per session.

    Falls back to the generic finder if the call shape ever stops matching
    those assumptions.
    """
    if any(isinstance(c.out_message, ComplexReplayMessage) for c in calls):
        return find_predecessors_by_text_matching(calls)
    if any(any(message.role == "assistant" for message in c.messages) for c in calls):
        return find_predecessors_by_text_matching(calls)

    predecessor_indices: List[Dict[int, DEPENDENCY_TYPE]] = [{} for _ in range(len(calls))]
    for i in range(1, len(calls)):
        for j in range(i - 1, -1, -1):
            if calls[i].t_start_ms >= calls[j].t_end_ms:
                predecessor_indices[i][j] = DEPENDENCY_TYPE.TEMPORAL
                break
    return predecessor_indices, {}
