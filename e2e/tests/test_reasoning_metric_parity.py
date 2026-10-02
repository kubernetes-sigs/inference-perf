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
"""Do inference-perf and guidellm report the same latency numbers for a reasoning model?

Both tools are pointed at ``FakeOpenAIServer``, the same scripted fake the
integration tier uses (``tests/required/integration/fake_openai_server.py``).
It streams a fixed script (reasoning tokens, a pause, content tokens) and
stamps when each request arrived and when each token was sent. Those stamps are the expected values. Each tool is checked
against the server's timeline, then the two tools against each other, so a
failure says which tool moved.

Which report field is compared with which:

    this test     inference-perf (seconds)       guidellm (milliseconds)
    ttft          time_to_first_token            time_to_first_token_ms
    ttfo          time_to_first_output_token     time_to_first_output_token_ms
    per_token     time_per_output_token          inter_token_latency_ms
    output tokens output_tokens                  output_tokens

The names do not line up for ``per_token``. inference-perf's
time_per_output_token and guidellm's inter_token_latency_ms are both
(last token - first token) / (tokens - 1). guidellm's time_per_output_token_ms
is (last token - request start) / tokens, a different quantity, and is not
compared here.

Locally the guidellm tests skip when guidellm is not installed (see
``e2e/utils/guidellm.py``). CI installs the pinned guidellm and sets
GUIDELLM_REQUIRED=1, which turns that skip into a failure. The inference-perf
test always runs.
"""

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

from utils.accuracy import SUMMARY_REPORT, assert_successful_run, client_output_tokens
from utils.benchmark import run_benchmark_minimal
from utils.guidellm import GuidellmUnavailable, check_version, find_guidellm_bin, guidellm_required, run_guidellm
from utils.testdata import extract_tarball

# The fake server lives with the integration tests that also use it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "required" / "integration"))
from fake_openai_server import FakeOpenAIServer, ServedStream, StreamEvent  # noqa: E402

MODEL_NAME = "reasoning-model"
TOKENIZER_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# inference-perf sends RATE * DURATION requests; guidellm is given the same
# number of prompts.
RATE = 2
DURATION = 3
N_REQUESTS = RATE * DURATION

# How far a reported mean may sit from the expected one, in seconds. Sized
# against the scripts below: counting from the wrong token moves ttft or ttfo
# by at least 0.78s, and dropping reasoning tokens moves per_token by 0.03s.
FIRST_TOKEN_TOLERANCE = 0.06
PER_TOKEN_TOLERANCE = 0.008

# Seconds the server waits: before the first token, between tokens on one
# channel, and between the last reasoning token and the first content token.
FIRST_TOKEN_DELAY = 0.10
INTER_TOKEN_DELAY = 0.04
ANSWER_DELAY = 0.50

# Text the scripts are cut from, one token per event.
REASONING_CORPUS = "The user wants the sum of two small numbers, so first recall that addition is commutative."
CONTENT_CORPUS = "The answer is four, because two plus two makes four whichever order they are added in."


# One script: `reasoning_tokens` reasoning events, then `content_tokens`
# content events. `reasoning_key` is the delta field the reasoning text is
# sent under.
@dataclass(frozen=True)
class ReasoningCase:
    reasoning_tokens: int
    content_tokens: int
    reasoning_key: str = "reasoning_content"

    @property
    def output_tokens(self) -> int:
        return self.reasoning_tokens + self.content_tokens


CASES: Dict[str, ReasoningCase] = {
    # 8 reasoning tokens 40ms apart, a 500ms pause, 8 content tokens 40ms apart.
    "reasoning_then_content": ReasoningCase(reasoning_tokens=8, content_tokens=8),
    # The output budget ran out mid-reasoning: no content token ever arrives.
    "reasoning_only": ReasoningCase(reasoning_tokens=8, content_tokens=0),
    # A model that does not reason.
    "content_only": ReasoningCase(reasoning_tokens=0, content_tokens=8),
    # Same stream as the first case, with reasoning under the other field name
    # vLLM has used (delta.reasoning instead of delta.reasoning_content).
    "reasoning_field_named_reasoning": ReasoningCase(reasoning_tokens=8, content_tokens=8, reasoning_key="reasoning"),
}

# Run this whole file on one pytest-xdist worker. Each tool's result is cached
# in this process, and the timing checks should not share a CPU with unrelated
# tests running at the same time.
pytestmark = pytest.mark.xdist_group(name="reasoning-metric-parity")


# The four numbers compared, as means over the requests of one run. Times are
# seconds. ttfo is None when no request received a content token.
@dataclass(frozen=True)
class Metrics:
    ttft: float
    ttfo: Optional[float]
    per_token: float
    output_tokens: List[int]


# One tool's run against one case: what the tool reported, and what the server
# recorded for those same requests. `summary` is inference-perf's summary
# report, empty for guidellm.
@dataclass(frozen=True)
class ToolRun:
    reported: Metrics
    expected: Metrics
    summary: Dict[str, Any]


# Mean of the values, or None if every value is None. [0.5, 0.7] gives 0.6.
# [None, None] gives None. A mix of None and numbers is an error: within one
# run every request gets the same stream.
def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    assert len(present) == len(values), f"some requests have a value and some do not: {values}"
    return statistics.mean(present)


# Cuts the first n tokens of `corpus` into n strings of one token each, using
# the tokenizer's own character offsets. ("The answer is", 3) gives
# ["The", " answer", " is"]. Fails if any piece does not re-encode to exactly
# one token, so a bad corpus breaks here and not in a metric assertion.
def _one_token_chunks(tokenizer: Any, corpus: str, n: int) -> List[str]:
    if n == 0:
        return []
    offsets = tokenizer(corpus, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    assert len(offsets) >= n, f"corpus has {len(offsets)} tokens, case needs {n}"
    starts = [offset[0] for offset in offsets[:n]] + [offsets[n - 1][1]]
    chunks = [corpus[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)]

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    assert count("".join(chunks)) == n and all(count(chunk) == 1 for chunk in chunks), chunks
    return chunks


# Builds the server script for a case. 2 reasoning + 2 content tokens gives
# four events with delays 0.10 (first token), 0.04, 0.50 (the pause before
# the answer), 0.04.
def _script(tokenizer: Any, case: ReasoningCase) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    for channel, corpus, n in (
        ("reasoning", REASONING_CORPUS, case.reasoning_tokens),
        ("content", CONTENT_CORPUS, case.content_tokens),
    ):
        for i, text in enumerate(_one_token_chunks(tokenizer, corpus, n)):
            if not events:
                delay = FIRST_TOKEN_DELAY
            elif i == 0:
                delay = ANSWER_DELAY
            else:
                delay = INTER_TOKEN_DELAY
            events.append(StreamEvent(channel, text, delay))
    return events


# Turns the server's per-request stamps into the expected Metrics: ttft is
# arrival to first token on either channel, ttfo is arrival to first content
# token, per_token is (last token - first token) / (tokens - 1). For the 8+8
# case this is roughly ttft 0.10, ttfo 0.88, per_token 0.071,
# output_tokens [16] * N_REQUESTS.
def _expected(case: ReasoningCase, served: List[ServedStream]) -> Metrics:
    assert len(served) == N_REQUESTS, f"server saw {len(served)} chat requests, expected {N_REQUESTS}"
    ttft, ttfo, per_token = [], [], []
    for r in served:
        sends = r.reasoning_send_times + r.content_send_times
        ttft.append(sends[0] - r.arrival_time)
        ttfo.append(r.content_send_times[0] - r.arrival_time if r.content_send_times else None)
        per_token.append((sends[-1] - sends[0]) / (len(sends) - 1))
    return Metrics(
        ttft=statistics.mean(ttft),
        ttfo=_mean_or_none(ttfo),
        per_token=statistics.mean(per_token),
        output_tokens=[case.output_tokens] * N_REQUESTS,
    )


# Runs inference-perf against `server` and reads the Metrics from its summary
# report (means) and per-request report (token counts).
async def _run_inference_perf(server: FakeOpenAIServer, tokenizer_path: str) -> Tuple[Metrics, Dict[str, Any]]:
    result = await run_benchmark_minimal(
        {
            "data": {"type": "mock"},
            "load": {"type": "constant", "stages": [{"rate": RATE, "duration": DURATION}], "num_workers": 1},
            "api": {"type": "chat", "streaming": True},
            "server": {"type": "vllm", "model_name": MODEL_NAME, "base_url": server.base_url, "ignore_eos": True},
            "tokenizer": {"pretrained_model_name_or_path": tokenizer_path},
            "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
        },
        timeout_sec=120,
    )
    entries = assert_successful_run(result, N_REQUESTS)
    assert result.reports is not None
    summary = result.reports[SUMMARY_REPORT]["successes"]
    latency = summary["latency"]
    ttfo = latency["time_to_first_output_token"]
    metrics = Metrics(
        ttft=latency["time_to_first_token"]["mean"],
        ttfo=None if ttfo is None else ttfo["mean"],
        per_token=latency["time_per_output_token"]["mean"],
        output_tokens=[client_output_tokens(entry) for entry in entries],
    )
    return metrics, summary


# Runs guidellm against `server` and reads the Metrics from its per-request
# report entries, converting milliseconds to seconds. If guidellm is not
# installed, skips the calling test, or fails it when GUIDELLM_REQUIRED=1.
async def _run_guidellm(server: FakeOpenAIServer) -> Metrics:
    try:
        guidellm_bin = find_guidellm_bin()
    except GuidellmUnavailable as e:
        if guidellm_required():
            pytest.fail(f"GUIDELLM_REQUIRED is set and {e}")
        pytest.skip(str(e))
    await check_version(guidellm_bin)
    result = await run_guidellm(
        guidellm_bin=guidellm_bin,
        base_url=server.base_url,
        model=MODEL_NAME,
        prompts=[f"What is {i} plus {i}?" for i in range(N_REQUESTS)],
        timeout_sec=120,
    )
    assert result.success, f"guidellm failed (rc={result.return_code}, timed_out={result.timed_out}):\n{result.stdout}"
    assert len(result.requests) == N_REQUESTS, f"guidellm reported {len(result.requests)} successful requests"

    def seconds(field: str) -> List[Optional[float]]:
        return [None if r[field] is None else r[field] / 1000 for r in result.requests]

    ttft = _mean_or_none(seconds("time_to_first_token_ms"))
    per_token = _mean_or_none(seconds("inter_token_latency_ms"))
    assert ttft is not None and per_token is not None, f"guidellm reported no ttft or itl: {result.requests[0]}"
    return Metrics(
        ttft=ttft,
        ttfo=_mean_or_none(seconds("time_to_first_output_token_ms")),
        per_token=per_token,
        output_tokens=[r["output_tokens"] for r in result.requests],
    )


_RUNS: Dict[Tuple[str, str], ToolRun] = {}


# Runs `tool` ("inference-perf" or "guidellm") against a fresh server serving
# CASES[case_id], once. Later calls with the same arguments return the first
# result, so three tests reading one run cost one run.
async def _tool_run(tool: str, case_id: str) -> ToolRun:
    key = (tool, case_id)
    if key not in _RUNS:
        case = CASES[case_id]
        tokenizer_path = str(extract_tarball(TOKENIZER_TARBALL))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        fake = FakeOpenAIServer(
            _script(tokenizer, case),
            completion_tokens=case.output_tokens,
            reasoning_key=case.reasoning_key,
            model=MODEL_NAME,
        )
        async with fake as server:
            summary: Dict[str, Any] = {}
            if tool == "inference-perf":
                reported, summary = await _run_inference_perf(server, tokenizer_path)
            else:
                reported = await _run_guidellm(server)
        _RUNS[key] = ToolRun(reported=reported, expected=_expected(case, server.served), summary=summary)
    return _RUNS[key]


# Fails unless `got` and `want` agree: identical token counts, ttft and ttfo
# within FIRST_TOKEN_TOLERANCE, per_token within PER_TOKEN_TOLERANCE, and
# ttfo missing on both sides or neither. `got_name` and `want_name` label the
# two sides in the failure message, e.g. "guidellm" and "server timeline".
def _assert_metrics_agree(got: Metrics, want: Metrics, *, got_name: str, want_name: str) -> None:
    where = f"{got_name} vs {want_name}"
    assert got.output_tokens == want.output_tokens, f"{where}: output tokens {got.output_tokens} != {want.output_tokens}"
    assert got.ttft == pytest.approx(want.ttft, abs=FIRST_TOKEN_TOLERANCE), (
        f"{where}: ttft {got.ttft:.4f}s != {want.ttft:.4f}s"
    )
    assert (got.ttfo is None) == (want.ttfo is None), f"{where}: ttfo {got.ttfo} vs {want.ttfo}, only one is missing"
    if want.ttfo is not None:
        assert got.ttfo == pytest.approx(want.ttfo, abs=FIRST_TOKEN_TOLERANCE), (
            f"{where}: ttfo {got.ttfo:.4f}s != {want.ttfo:.4f}s"
        )
    assert got.per_token == pytest.approx(want.per_token, abs=PER_TOKEN_TOLERANCE), (
        f"{where}: per_token {got.per_token * 1000:.2f}ms != {want.per_token * 1000:.2f}ms"
    )


# Guard on the tolerances, no server involved. For the 8+8 script, a tool that
# anchored ttft to the first content token would be off by the whole reasoning
# phase (7 * 40ms + 500ms = 0.78s), and one that dropped reasoning tokens from
# per_token would report 40ms instead of (7*40 + 500 + 7*40) / 15 = 70.7ms.
# Both errors must be several times larger than the tolerance, or the tests
# below could pass with the wrong definition.
def test_tolerances_can_tell_the_definitions_apart() -> None:
    case = CASES["reasoning_then_content"]
    reasoning_phase = (case.reasoning_tokens - 1) * INTER_TOKEN_DELAY + ANSWER_DELAY
    content_phase = (case.content_tokens - 1) * INTER_TOKEN_DELAY
    all_tokens_itl = (reasoning_phase + content_phase) / (case.output_tokens - 1)
    content_only_itl = INTER_TOKEN_DELAY

    assert reasoning_phase > 4 * FIRST_TOKEN_TOLERANCE
    assert all_tokens_itl - content_only_itl > 3 * PER_TOKEN_TOLERANCE


# For each case, inference-perf's summary report must match what the server
# recorded for the same 6 requests. For the 8+8 case that means ttft near
# 0.10s, ttfo near 0.88s, time_per_output_token and inter_token_latency both
# near 70.7ms, and 16 output tokens per request. With no reasoning, ttfo
# equals ttft. With no content, ttfo is absent.
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(CASES))
async def test_inference_perf_matches_server_timeline(case_id: str) -> None:
    run = await _tool_run("inference-perf", case_id)
    _assert_metrics_agree(run.reported, run.expected, got_name="inference-perf", want_name="server timeline")

    # One token per event, so the mean gap between tokens is the same number
    # as time_per_output_token.
    itl = run.summary["latency"]["inter_token_latency"]["mean"]
    assert itl == pytest.approx(run.expected.per_token, abs=PER_TOKEN_TOLERANCE)
    assert run.summary["token_count_mismatches"] == 0

    if CASES[case_id].reasoning_tokens == 0:
        assert run.reported.ttfo == pytest.approx(run.reported.ttft, abs=1e-3)


# The same check for guidellm: its per-request numbers, averaged over 6
# requests, must match what the server recorded for them.
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(CASES))
async def test_guidellm_matches_server_timeline(case_id: str) -> None:
    run = await _tool_run("guidellm", case_id)
    _assert_metrics_agree(run.reported, run.expected, got_name="guidellm", want_name="server timeline")

    if CASES[case_id].reasoning_tokens == 0:
        assert run.reported.ttfo == pytest.approx(run.reported.ttft, abs=1e-3)


# The two tools' reports, side by side. Same script served to both, so ttft,
# ttfo, per_token and output token counts must agree within the same
# tolerances used against the server.
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(CASES))
async def test_inference_perf_and_guidellm_agree(case_id: str) -> None:
    inference_perf = await _tool_run("inference-perf", case_id)
    guidellm = await _tool_run("guidellm", case_id)
    _assert_metrics_agree(inference_perf.reported, guidellm.reported, got_name="inference-perf", want_name="guidellm")
