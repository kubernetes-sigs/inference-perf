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
"""TraceLab replay against llm-d-inference-sim.

Pins what a replayed agent session has to reproduce: one request per
recorded round, sent in round order with the recorded gap after the
previous round finished, each prompt tokenizing to the recorded input
length, and each response the recorded output length.

Two sessions of three rounds. Session A grows: each round's prompt covers
the whole previous context. Session B compacts on its last round: the
input shrinks. Each round starts THINK_SEC after the previous round of
its session ended (via timing_events timestamps).

If `llm-d-inference-sim` is not on PATH these tests skip automatically.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from inference_perf.config import CustomTokenizerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from utils.accuracy import assert_successful_run, request_body, server_completion_tokens
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# (round_index, input_tokens_total, prefix_tokens, output_tokens)
SESSIONS: Dict[str, List[Tuple[int, int, int, int]]] = {
    "claude:A": [(0, 48, 0, 6), (1, 96, 48, 8), (2, 160, 96, 10)],
    "claude:B": [(0, 80, 0, 5), (1, 128, 80, 12), (2, 64, 16, 20)],
}
THINK_SEC = 0.5
ROUND_SEC = 0.4  # recorded duration of a round, first event to last

# How much later than the recorded gap a round may start. The session
# runtime waits on the predecessor's completion in another process, so a
# couple of hundred milliseconds of skew is normal; starting early is not.
LATE_TOLERANCE_SEC = 0.3
EARLY_TOLERANCE_SEC = 0.05


def fork_start_method_executable() -> List[str]:
    """inference-perf under a start method that can hand the session generator
    to its workers. Python 3.14 defaults to forkserver on Linux, which pickles
    the worker arguments, and the session generators hold the multiprocessing
    manager, which cannot be pickled; the tool itself only forces fork on
    macOS. CI's Python defaults to fork, so this matters for local runs."""
    return [
        sys.executable,
        "-c",
        "import multiprocessing as mp, sys; mp.set_start_method('fork', force=True); "
        "from inference_perf.main import main_cli; sys.exit(main_cli())",
    ]


def write_tracelab_trace(path: Path) -> Path:
    rows = []
    for session_offset, (session_id, rounds) in enumerate(SESSIONS.items()):
        clock = 10.0 + 0.1 * session_offset
        for index, total, prefix, out in rounds:
            start, end = clock, clock + ROUND_SEC
            clock = end + THINK_SEC
            rows.append(
                {
                    "session_id": session_id,
                    "round_id": f"r{index + 1}",
                    "round_index": index,
                    "provider": session_id.split(":")[0],
                    "model": "claude-opus-4-8",
                    "input_tokens_total": total,
                    "prefix_tokens": prefix,
                    "newly_append_tokens": total - prefix,
                    "output_tokens": out,
                    "timing_events": [
                        {"event_type": "user_message", "timestamp": f"2026-06-01T10:00:{start:06.3f}Z"},
                        {"event_type": "text", "timestamp": f"2026-06-01T10:00:{end:06.3f}Z"},
                    ],
                    "tools": [],
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def round_index_of(entry: Dict[str, Any]) -> int:
    """Per-request report carries the raw graph event id event_NNN_round_i."""
    event_id = entry["info"]["graph_event_id"]
    assert "_round_" in event_id, f"unexpected graph event id: {event_id}"
    return int(event_id.rsplit("_round_", 1)[1])


def prompt_text(entry: Dict[str, Any]) -> str:
    """Concatenated contents of the sent chat messages (growing transcript)."""
    messages = request_body(entry)["messages"]
    assert messages and all(m["role"] in ("system", "user", "assistant", "tool") for m in messages), (
        f"unexpected roles: {[m.get('role') for m in messages]}"
    )
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append("".join(p.get("text", "") for p in content if isinstance(p, dict)))
    return "".join(parts)


def assert_replays_the_sessions(entries: List[Dict[str, Any]], tokenizer: CustomTokenizer) -> None:
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        session_id = entry.get("session_id")
        assert session_id, "per-request entry is missing session_id"
        entry["_round"] = round_index_of(entry)
        by_session[session_id].append(entry)
    assert len(by_session) == len(SESSIONS), f"sessions in report: {sorted(by_session)}"

    # Runtime session ids are tracelabtrace{idx}_{safe}: match candidates to the
    # spec by their sorted prompt-token sequences (A grows, B compacts).
    def prompt_sequence(candidate: List[Dict[str, Any]]) -> List[int]:
        ordered = sorted(candidate, key=lambda e: e["_round"])
        return [tokenizer.count_tokens(prompt_text(e)) for e in ordered]

    unmatched = list(by_session.values())
    for session_id, rounds in SESSIONS.items():
        want = [total for _, total, _, _ in rounds]
        match = None
        for candidate in unmatched:
            if len(candidate) == len(rounds) and prompt_sequence(candidate) == want:
                match = candidate
                break
        assert match is not None, (
            f"no session with prompt sizes {want} for {session_id}; got {[prompt_sequence(c) for c in unmatched]}"
        )
        unmatched.remove(match)

        got = sorted(match, key=lambda e: e["start_time"])
        assert [e["_round"] for e in got] == [r[0] for r in rounds], f"{session_id} rounds ran out of order"

        previous_end = None
        for entry, (index, total, _prefix, out) in zip(got, rounds, strict=True):
            body = request_body(entry)
            got_in = tokenizer.count_tokens(prompt_text(entry))
            assert got_in == total, f"{session_id} round {index}: prompt of {got_in} tokens, trace recorded {total}"
            assert body["max_tokens"] == out, (
                f"{session_id} round {index}: asked for {body['max_tokens']}, trace recorded {out}"
            )
            assert server_completion_tokens(entry) == out, (
                f"{session_id} round {index}: server produced {server_completion_tokens(entry)}, expected {out}"
            )
            if previous_end is not None:
                gap = entry["start_time"] - previous_end
                assert gap >= THINK_SEC - EARLY_TOLERANCE_SEC, (
                    f"{session_id} round {index} started {gap:.3f}s after the previous round ended, recorded {THINK_SEC}s"
                )
                assert gap <= THINK_SEC + LATE_TOLERANCE_SEC, (
                    f"{session_id} round {index} started {gap:.3f}s after the previous round ended, recorded {THINK_SEC}s"
                )
            previous_end = entry["end_time"]


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_tracelab_session_replay(tmp_path: Path) -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    trace = write_tracelab_trace(tmp_path / "syfi_coding_trace.jsonl")
    total_rounds = sum(len(rounds) for rounds in SESSIONS.values())

    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "tracelab_trace_replay",
                    "tracelab_trace_replay": {
                        "trace_files": [str(trace)],
                        "use_static_model": True,
                        "static_model_name": TEST_MODEL_NAME,
                    },
                },
                "load": {"type": "trace_session_replay", "num_workers": 2, "stages": [{"concurrent_sessions": 2}]},
                "api": {"type": "chat", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {"summary": True, "per_stage": True, "per_request": True},
                    "session_lifecycle": {"summary": True, "per_stage": True, "per_session": True},
                },
            },
            executable=fork_start_method_executable(),
        )

    entries = assert_successful_run(result, expected_requests=total_rounds)
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(model_path)))
    assert_replays_the_sessions(entries, tokenizer)

    sessions = result.reports.get("summary_session_lifecycle_metrics.json") if result.reports else None
    assert sessions, f"missing session summary in {sorted(result.reports or [])}"
    assert sessions["num_sessions"] == len(SESSIONS), sessions
    assert sessions["num_sessions_succeeded"] == len(SESSIONS), sessions
