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
latencies), and round ordering are preserved. See
https://tracelab.cs.washington.edu and https://arxiv.org/abs/2606.30560.

Replay model (v1):
- Rows are grouped by ``session_id`` and ordered by ``round_index``.
- Each round becomes one LLM call in a linear session chain. The prompt text
  is synthetically reconstructed to match the recorded input token count
  (deterministic corpus slices, so prefix-cache structure grows like the
  original), and the expected output is sized to the recorded
  ``output_tokens``.
- Inter-round tool latency (sum of ``tools[].tool_wall_latency_ms``) is
  preserved as event spacing, capped by ``trace_idle_gap_cap_seconds``.
"""

import gzip
import json
import logging
import random
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


class TraceLabRound(BaseModel):
    """One TraceLab JSONL row (one LLM invocation). Extra fields ignored."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    session_id: str = Field(default="")
    round_id: Optional[str] = None
    round_index: int = Field(default=0)
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens_total: int = Field(default=0)
    prefix_tokens: int = Field(default=0)
    newly_append_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    tools: List[TraceLabToolCall] = Field(default_factory=list)


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """Read TraceLab rows from .json / .jsonl / .jsonl.gz files."""
    suffixes = "".join(path.suffixes)
    opener: Callable[..., Any]
    if suffixes.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    rows: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return rows
        f.seek(0)
        if path.suffix == ".json" and not str(path).endswith(".jsonl"):
            blob = json.load(f)
            if isinstance(blob, list):
                return [r for r in blob if isinstance(r, dict)]
            if isinstance(blob, dict):
                return [blob]
            return rows
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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

        raw_rows: List[Dict[str, Any]] = []
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
                    raw_rows.extend(_read_jsonl_rows(f))
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
                    raw_rows.extend(_read_jsonl_rows(f))
                except Exception as e:
                    logger.error(f"Failed to load trace file {f.name}: {e}")
                    if not cfg.skip_invalid_files:
                        raise
        elif cfg.hf_dataset_path:
            raw_rows = self._load_hf_rows(cfg.hf_dataset_path, cfg.num_dataset_entries, cfg.skip_invalid_files)
        else:
            raise ValueError("Either trace_directory, trace_files, or hf_dataset_path must be provided")

        grouped: Dict[str, List[TraceLabRound]] = {}
        n_invalid = 0
        for idx, blob in enumerate(raw_rows):
            try:
                rnd = TraceLabRound.model_validate(blob)
            except Exception as e:
                logger.error(f"Failed to validate TraceLab row {idx}: {e}")
                if not cfg.skip_invalid_files:
                    raise
                n_invalid += 1
                continue
            if not rnd.session_id:
                rnd.session_id = f"unknown_session_{idx}"
            key = f"{rnd.provider}:{rnd.session_id}" if rnd.provider else rnd.session_id
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
            logger.info(f"Skipped {n_invalid} invalid rows")
        if not kept:
            raise ValueError("No valid TraceLab sessions found")
        logger.info(f"Loaded {len(kept)} TraceLab sessions ({sum(len(v) for v in kept.values())} rounds)")
        return kept

    def _load_hf_rows(self, hf_path: Any, limit: int, skip_invalid: bool) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        if isinstance(hf_path, dict):
            path = hf_path.get("path", DEFAULT_HF_DATASET)
            kwargs = {k: v for k, v in hf_path.items() if k != "path"}
        else:
            path = hf_path or DEFAULT_HF_DATASET
            kwargs = {}
        logger.info(f"Downloading TraceLab traces from Hugging Face dataset: {path}")
        try:
            ds = load_dataset(path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset {path}: {e}") from e
        # load_dataset returns a DatasetDict when no split is pinned; prefer train.
        if hasattr(ds, "keys"):
            split = kwargs.get("split", "train")
            dataset = ds[split] if split in ds.keys() else ds[list(ds.keys())[0]]
        else:
            dataset = ds
        rows: List[Dict[str, Any]] = []
        for row in dataset:
            if len(rows) >= limit:
                break
            if isinstance(row, dict):
                rows.append(dict(row))
            elif not skip_invalid:
                raise ValueError(f"Unexpected row type {type(row)} in Hugging Face dataset {path}")
        return rows

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

    def _reconstruct_raw_calls(self, session_id: str, rounds: List[TraceLabRound]) -> List[RawCall]:
        model_map = self._build_model_map([r.model or "" for r in rounds if r.model])
        cap_ms = int(self.tracelab_config.trace_idle_gap_cap_seconds * 1000.0)

        calls: List[RawCall] = []
        history: List[ReplayMessage] = []
        t_ms = 0
        for i, rnd in enumerate(rounds):
            in_tokens = max(int(rnd.input_tokens_total or 0), 0)
            out_tokens = max(int(rnd.output_tokens or 0), 0)
            # History already carries the grown transcript; size only the new
            # append so the total tracks the recorded input length. Fall back
            # to the full input length for the first round or when the trace
            # only reports totals.
            append_tokens = int(rnd.newly_append_tokens or 0)
            if i == 0 or append_tokens <= 0 or append_tokens > in_tokens:
                new_tokens = in_tokens
            else:
                new_tokens = append_tokens
            user_text = self._text_for_tokens(new_tokens, offset=(hash(session_id) % 10_000) + i * 7919)
            messages = list(history) + [ReplayMessage(role="user", text=user_text or " ")]
            out_text = self._text_for_tokens(out_tokens, offset=(hash(session_id) % 10_000) + i * 104729 + 13)
            out_message = ReplayMessage(role="assistant", text=out_text or " ")
            model = model_map.get(rnd.model or "", rnd.model or "tracelab-model")

            tool_wait_ms = 0
            for t in rnd.tools or []:
                lat = t.tool_wall_latency_ms if t.tool_wall_latency_ms is not None else t.tool_internal_latency_ms
                if lat:
                    tool_wait_ms += max(int(lat), 0)
            if self.tracelab_config.ignore_trace_delays:
                wait_ms = 0
            else:
                wait_ms = min(tool_wait_ms, cap_ms) if tool_wait_ms else 0
            t_ms += wait_ms

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
            # Grow the transcript so build_graph infers the linear chain via
            # output->input text matching and the KV cache sees growth.
            history = history + [ReplayMessage(role="user", text=user_text or " "), out_message]
        return calls

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
                        session_id=f"tracelabtrace{trace_index}_{session_id}",
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
