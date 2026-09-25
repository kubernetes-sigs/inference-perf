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
"""Weka trace replay against llm-d-inference-sim.

Pins what a replayed Weka trace has to reproduce, whichever code path
replays it: one request per recorded call, each request's own text (the
system and user messages, which substitution never touches) tokenizing to
the recorded input length less the recorded assistant blocks, each request
asking for and getting the recorded output length, a later round's prompt
starting with the earlier round's, every recorded assistant turn replaced
by the live reply it stands for, subagent calls spawned from and joined back
into their parent, and the recorded gap between a call and its predecessor
honoured.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from inference_perf.config import CustomTokenizerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from utils.accuracy import assert_successful_run, request_body, response_metrics, server_completion_tokens
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

BLOCK_SIZE = 8
API_TIME_SEC = 0.3

# How far a call's send time may drift from where its predecessor's
# completion plus the recorded gap puts it. Session events pass through a
# worker process and a completion queue, so a couple of hundred milliseconds
# is normal; a whole second is not.
GAP_TOLERANCE_SEC = 0.3


def call(t: float, n_in: int, n_out: int, hash_ids: List[int]) -> Dict[str, Any]:
    return {"t": t, "type": "n", "model": "m", "in": n_in, "out": n_out, "hash_ids": hash_ids, "api_time": API_TIME_SEC}


# Two sessions, seven calls, every call with its own output length so the
# report can be matched back to the trace by max_tokens.
#
# Session A: three rounds of one growing conversation (no system prompt).
# Session B: a system prompt (tool_tokens + system_tokens = two blocks), a
# subagent spawned after the first round that makes two calls of its own,
# and a second parent round that joins the subagent's result back in.
TRACE_A = {
    "id": "A",
    "models": ["m"],
    "block_size": BLOCK_SIZE,
    "tool_tokens": 0,
    "system_tokens": 0,
    "requests": [
        call(0.0, 24, 4, [1, 2, 3]),
        call(1.0, 40, 6, [1, 2, 3, 4, 5]),
        call(2.5, 56, 8, [1, 2, 3, 4, 5, 6, 7]),
    ],
}
TRACE_B = {
    "id": "B",
    "models": ["m"],
    "block_size": BLOCK_SIZE,
    "tool_tokens": 8,
    "system_tokens": 8,
    "requests": [
        call(0.0, 32, 5, [10, 11, 12, 13]),
        {
            "t": 0.8,
            "type": "subagent",
            "agent_id": "sa1",
            "subagent_type": "worker",
            "duration_ms": 900,
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [call(0.8, 16, 2, [20, 21]), call(1.3, 24, 3, [20, 21, 22])],
        },
        call(2.0, 48, 7, [10, 11, 12, 13, 14, 15]),
    ],
}

# What each call must look like on the wire, keyed by its recorded output
# length: the recorded input length, the roles of its messages, and how
# many of those tokens are the request's own (system and user) text. The
# recorded assistant turns are one block each and get replaced by the live
# reply, so they are excluded from the exact count.
EXPECTED: Dict[int, Tuple[int, List[str]]] = {
    4: (24, ["user"]),
    6: (40, ["user", "assistant", "user"]),
    8: (56, ["user", "assistant", "user", "assistant", "user"]),
    5: (32, ["system", "user"]),
    2: (16, ["user"]),
    3: (24, ["user", "assistant"]),
    7: (48, ["system", "user", "assistant", "user"]),
}

# Dependency edges as (call, predecessor, recorded gap in seconds): the gap
# is the call's t minus the predecessor's t plus api_time. Within a session
# they chain the rounds; across the subagent they are the spawn (2 after 5)
# and the join (7 after 3).
EDGES: List[Tuple[int, int, float]] = [
    (6, 4, 0.7),
    (8, 6, 1.2),
    (2, 5, 0.5),
    (3, 2, 0.2),
    (7, 3, 0.4),
]

# A later call's own messages start with an earlier call's: (call, earlier).
PREFIXES: List[Tuple[int, int]] = [(6, 4), (8, 6), (3, 2), (7, 5)]


def write_traces(tmp_path: Path) -> List[str]:
    paths = []
    for trace in (TRACE_A, TRACE_B):
        path = tmp_path / f"{trace['id']}.json"
        path.write_text(json.dumps(trace))
        paths.append(str(path))
    return paths


def weka_trace_config(trace_files: List[str]) -> Dict[str, Any]:
    """The pre-record-layer spelling: the Weka generator and the session runner."""
    return {
        "data": {
            "type": "weka_trace_replay",
            "weka_trace_replay": {
                "trace_files": trace_files,
                "default_block_size": BLOCK_SIZE,
                "datagen_workers": 1,
                "use_static_model": True,
                "static_model_name": TEST_MODEL_NAME,
            },
        },
        # Session replay always runs through worker processes. Under Python
        # 3.14, whose default start method on Linux is forkserver, the session
        # generators cannot be pickled to the child and the run dies before
        # the first request; run this under an interpreter that forks (3.13
        # or earlier). That is a pre-existing problem shared with the OTel
        # path, not this test's.
        "load": {"type": "trace_session_replay", "stages": [{"concurrent_sessions": 2}], "num_workers": 2},
    }


def own_messages(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """The request's system and user messages: the text the trace fixed."""
    return [m for m in request_body(entry)["messages"] if m["role"] != "assistant"]


def reply_text(entry: Dict[str, Any]) -> str:
    """The reply the client received, from the raw streamed chunks it kept."""
    text = ""
    for chunk in response_metrics(entry)["response_chunks"]:
        payload = chunk.strip()
        if payload.startswith("data:"):
            payload = payload[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        for choice in json.loads(payload).get("choices", []):
            text += choice.get("delta", {}).get("content") or ""
    return text


def by_output_length(entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Match each report entry to its recorded call by the max_tokens it asked for."""
    matched: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        n_out = request_body(entry)["max_tokens"]
        assert n_out in EXPECTED, f"request asked for {n_out} tokens, which no recorded call does"
        assert n_out not in matched, f"two requests asked for {n_out} tokens; the trace has one such call"
        matched[n_out] = entry
    assert sorted(matched) == sorted(EXPECTED), f"calls replayed: {sorted(matched)}, recorded: {sorted(EXPECTED)}"
    return matched


def assert_replays_the_trace(entries: List[Dict[str, Any]], tokenizer: CustomTokenizer) -> None:
    calls = by_output_length(entries)

    # Lengths and roles per call. Prompt text is measured with the
    # benchmark's own tokenizer: the sim estimates prompt_tokens instead of
    # tokenizing, so its usage numbers say nothing about what was sent.
    for n_out, (n_in, roles) in EXPECTED.items():
        entry = calls[n_out]
        messages = request_body(entry)["messages"]
        assert [m["role"] for m in messages] == roles, f"call {n_out}: roles {[m['role'] for m in messages]}, expected {roles}"
        own_tokens = sum(tokenizer.count_tokens(m["content"], add_special_tokens=False) for m in own_messages(entry))
        expected_own = n_in - BLOCK_SIZE * roles.count("assistant")
        assert own_tokens == expected_own, (
            f"call {n_out}: own text is {own_tokens} tokens, expected {expected_own} of the recorded {n_in}"
        )
        assert server_completion_tokens(entry) == n_out, f"call {n_out}: server produced {server_completion_tokens(entry)}"

    # A later round carries the earlier round's text in front of its own.
    for later, earlier in PREFIXES:
        head = own_messages(calls[later])[: len(own_messages(calls[earlier]))]
        assert head == own_messages(calls[earlier]), f"call {later} does not start with call {earlier}'s messages"

    # Every recorded assistant turn is the live reply of the call it followed.
    for later, earlier in PREFIXES:
        assistant = [m["content"] for m in request_body(calls[later])["messages"] if m["role"] == "assistant"]
        if assistant:
            assert assistant[-1] == reply_text(calls[earlier]), (
                f"call {later} carries recorded text where call {earlier}'s live reply belongs"
            )

    # Timing: a call leaves the recorded gap after its predecessor finishes.
    for n_out, pred, gap in EDGES:
        got = calls[n_out]["start_time"] - calls[pred]["end_time"]
        assert abs(got - gap) <= GAP_TOLERANCE_SEC, (
            f"call {n_out} left {got:.3f}s after call {pred} finished, recorded gap {gap:.3f}s"
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize("make_config", [pytest.param(weka_trace_config, id="weka_trace_replay")])
async def test_weka_trace_replay(tmp_path: Path, make_config: Any) -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    trace_files = write_traces(tmp_path)

    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            {
                **make_config(trace_files),
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
            }
        )

    entries = assert_successful_run(result, expected_requests=len(EXPECTED))
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(model_path)))
    assert_replays_the_trace(entries, tokenizer)

    # Two sessions, both completed: three events in A, four in B.
    assert result.reports is not None
    summary = result.reports["summary_session_lifecycle_metrics.json"]
    assert summary["num_sessions"] == 2 and summary["num_sessions_succeeded"] == 2, summary
    sessions = result.reports["per_session_lifecycle_metrics.json"]
    assert sorted(s["num_events"] for s in sessions) == [3, 4], sessions
