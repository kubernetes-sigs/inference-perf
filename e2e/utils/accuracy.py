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
"""Shared accuracy/consistency assertions over benchmark reports.

Helpers operate on entries of ``per_request_lifecycle_metrics.json`` (dicts,
as parsed by ``run_benchmark_minimal``). The golden accuracy e2e (#631) calls
them with ``tolerance=0`` and exact expectations; the universal invariant
harness (#630) is meant to reuse them with non-zero tolerances on runs where
ground truth is unknown.

Every helper starts from the same rule: an assertion that can pass on an
empty or failed run is worthless, so callers go through
``assert_successful_run`` first, which refuses vacuous reports.
"""

import json
from typing import Any, Dict, List, Optional

from utils.benchmark import BenchmarkResult

PER_REQUEST_REPORT = "per_request_lifecycle_metrics.json"
SUMMARY_REPORT = "summary_lifecycle_metrics.json"


def assert_successful_run(result: BenchmarkResult, expected_requests: int) -> List[Dict[str, Any]]:
    """Refuse vacuous runs: the benchmark must have exited cleanly and produced
    exactly ``expected_requests`` successful, error-free request entries.

    Returns the per-request entries so callers can chain assertions.
    """
    assert result.success, f"benchmark process failed (rc={result.return_code}, timed_out={result.timed_out})"
    assert result.reports, "benchmark produced no reports"
    assert PER_REQUEST_REPORT in result.reports, f"missing {PER_REQUEST_REPORT} in {sorted(result.reports)}"

    entries = result.reports[PER_REQUEST_REPORT]
    assert len(entries) == expected_requests, f"expected {expected_requests} requests in report, got {len(entries)}"
    errored = [e for e in entries if e.get("error")]
    assert not errored, f"{len(errored)}/{len(entries)} requests errored, first: {errored[0]['error']}"

    summary = result.reports.get(SUMMARY_REPORT)
    assert summary, f"missing {SUMMARY_REPORT}"
    assert summary["successes"]["count"] == expected_requests, (
        f"summary reports {summary['successes']['count']} successes, expected {expected_requests}"
    )
    return entries


def response_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = (entry.get("info") or {}).get("response_metrics")
    assert metrics, f"request entry has no response_metrics: {entry.get('request', '')[:200]}"
    return metrics


def client_output_tokens(entry: Dict[str, Any]) -> int:
    """Client-derived output length (re-tokenization of the received text)."""
    tokens = response_metrics(entry).get("output_tokens")
    assert tokens is not None, "response_metrics has no output_tokens"
    return int(tokens)


def server_completion_tokens(entry: Dict[str, Any]) -> Optional[int]:
    """Server-reported usage.completion_tokens, None if the server sent none."""
    usage = response_metrics(entry).get("server_usage")
    if not usage or usage.get("completion_tokens") is None:
        return None
    return int(usage["completion_tokens"])


def request_body(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON request body the client actually sent."""
    return json.loads(entry["request"])


def assert_output_token_accounting(
    entry: Dict[str, Any],
    *,
    expected: Optional[int] = None,
    tolerance: int = 0,
) -> None:
    """Client-derived output_len must agree with the server-reported
    completion_tokens within ``tolerance`` tokens (0 = exactly), and with
    ``expected`` (ground truth) when the caller knows it.
    """
    client = client_output_tokens(entry)
    server = server_completion_tokens(entry)
    assert server is not None, "server reported no completion_tokens; cannot check token accounting"
    assert abs(client - server) <= tolerance, (
        f"client output_len {client} vs server completion_tokens {server} exceeds tolerance {tolerance}"
    )
    if expected is not None:
        assert server == expected, f"server completion_tokens {server} != expected {expected}"
        assert abs(client - expected) <= tolerance, (
            f"client output_len {client} vs expected {expected} exceeds tolerance {tolerance}"
        )


def chunk_times(entry: Dict[str, Any]) -> List[float]:
    times = response_metrics(entry).get("chunk_times")
    assert times is not None, "response_metrics has no chunk_times (not a streamed request?)"
    return list(times)


def chunk_gaps(entry: Dict[str, Any]) -> List[float]:
    """Client-observed inter-chunk arrival gaps (len == chunks - 1)."""
    times = chunk_times(entry)
    return [t2 - t1 for t1, t2 in zip(times, times[1:], strict=False)]


def ttft(entry: Dict[str, Any]) -> float:
    times = chunk_times(entry)
    assert times, "no content chunks received"
    return times[0] - entry["start_time"]


def assert_streaming_bookkeeping(
    entry: Dict[str, Any],
    *,
    expected_chunks: int,
    expected_token_times: Optional[int] = None,
) -> None:
    """ITL bookkeeping must be internally consistent with what was received:

    - exactly ``expected_chunks`` content-bearing chunks were timestamped
    - ``output_token_times`` (chunk times expanded per token by reportgen)
      has exactly ``expected_token_times`` entries when given; its inter-token
      gaps must contain exactly chunks-1 nonzero values (the real chunk gaps,
      intra-chunk gaps are 0 by construction)
    """
    times = chunk_times(entry)
    assert len(times) == expected_chunks, f"expected {expected_chunks} content chunks, got {len(times)}"
    assert times == sorted(times), "chunk_times are not monotonically nondecreasing"

    token_times = response_metrics(entry).get("output_token_times")
    assert token_times, "response_metrics has no output_token_times"
    if expected_token_times is not None:
        assert len(token_times) == expected_token_times, (
            f"expected {expected_token_times} output_token_times, got {len(token_times)}"
        )
    token_gaps = [t2 - t1 for t1, t2 in zip(token_times, token_times[1:], strict=False)]
    nonzero = [g for g in token_gaps if g != 0.0]
    assert len(nonzero) == expected_chunks - 1, (
        f"expected {expected_chunks - 1} nonzero inter-token gaps (one per chunk boundary), got {len(nonzero)}"
    )
