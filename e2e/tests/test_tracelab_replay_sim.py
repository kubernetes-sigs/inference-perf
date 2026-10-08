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
"""TraceLab replay against llm-d-inference-sim: the session runtime driven
by a workload arrangement.

Pins what a replayed agent session has to reproduce: one request per
recorded round, sent in round order with the recorded gap after the
previous round finished, each prompt tokenizing to the recorded length,
each response the recorded length, and each round's prompt starting with
exactly the blocks the previous round's prompt had where the trace says
the engine served them from cache.
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
BLOCK_SIZE = 16

# Two sessions of three rounds. Session A grows: each round's prefix is the
# whole previous prompt, so every block is reused. Session B compacts on its
# last round: the prefix drops to one block, so only that block is reused.
# Each round starts 0.5 s after the previous round of its session ended.
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
                    "round_index": index,
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


def session_and_round(entry: Dict[str, Any]) -> Tuple[str, int]:
    """The per-request report carries the graph event id, event_NNN_<node id>
    with ':' replaced by '_', and a TraceLab node id is <session>:<round>."""
    event_id = entry["info"]["graph_event_id"]
    node_id = event_id.split("_", 2)[2]
    session, round_index = node_id.rsplit("_", 1)
    return session.replace("_", ":", 1), int(round_index)


def prompt_content(entry: Dict[str, Any]) -> str:
    messages = request_body(entry)["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user", f"expected one user message, got {messages}"
    return str(messages[0]["content"])


def assert_shared_prefix(tokenizer: CustomTokenizer, previous: str, current: str, prefix_tokens: int) -> None:
    """The current prompt's first prefix_tokens ids equal the previous
    prompt's, and the block after them differs. Decoding and re-encoding
    can merge one token pair at the boundary, so one id of slack is allowed
    there and reported when it is needed."""
    hf = tokenizer.get_tokenizer()
    prev_ids = list(hf.encode(previous, add_special_tokens=False))
    cur_ids = list(hf.encode(current, add_special_tokens=False))
    shared = prefix_tokens
    if prev_ids[:shared] != cur_ids[:shared]:
        assert prev_ids[: shared - 1] == cur_ids[: shared - 1], (
            f"prompts diverge before the {shared} shared tokens: {prev_ids[:shared]} vs {cur_ids[:shared]}"
        )
        shared -= 1
        print(f"boundary token differs after {shared} shared ids (tokenizer merge at the block edge)")
    after = slice(shared, shared + BLOCK_SIZE)
    if len(prev_ids) > shared and len(cur_ids) > shared:
        assert prev_ids[after] != cur_ids[after], f"the block after the {shared} shared tokens is still shared"


def assert_replays_the_sessions(entries: List[Dict[str, Any]], tokenizer: CustomTokenizer) -> None:
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        session, round_index = session_and_round(entry)
        entry["_round"] = round_index
        by_session[session].append(entry)
    assert sorted(by_session) == sorted(SESSIONS), f"sessions in report: {sorted(by_session)}"

    for session_id, rounds in SESSIONS.items():
        got = sorted(by_session[session_id], key=lambda e: e["start_time"])
        assert [e["_round"] for e in got] == [r[0] for r in rounds], f"{session_id} rounds ran out of order"

        previous_prompt = ""
        previous_end = None
        for entry, (index, total, prefix, out) in zip(got, rounds, strict=True):
            body = request_body(entry)
            content = prompt_content(entry)
            got_in = tokenizer.count_tokens(content)
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
                assert_shared_prefix(tokenizer, previous_prompt, content, prefix)
            previous_prompt = content
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
                    "type": "workload_replay",
                    "workload": {"format": "TraceLab", "file": str(trace), "block_size": BLOCK_SIZE},
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
