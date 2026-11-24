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

"""Replaying traces from S3-compatible storage with real boto3 against an in-process S3 stub.

Covers what the FakeS3 unit tests cannot: boto3 and botocore over HTTP through
AWS_ENDPOINT_URL_S3, If-Match and Range on the wire, pagination, and forkserver workers
reading the cache the parent filled. Runs against the mock model server, offline.
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from inference_perf.datagen.replay.wire_trace_converter import _zstd
from utils.benchmark import run_benchmark_minimal
from utils.s3_stub import S3Stub
from utils.testdata import extract_tarball

BUCKET = "traces"
SESSIONS = 4
CALLS = 2
ACCESS_KEY = "e2e-key"
TOKENIZER = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
PROBE_RANGE = "bytes=0-262143"


# A zstd wire capture of CALLS Chat Completions calls with session header `session`.
def capture(session: str) -> bytes:
    rows = [
        {
            "path": "/v1/chat/completions",
            "wire": "openai",
            "start_unix": 1.0 + i,
            "headers": {"session_id": session},
            "request": json.dumps({"model": "m", "messages": [{"role": "user", "content": f"hi {i}"}]}),
            "model": "m",
            "status": 200,
            "response": json.dumps({"choices": [{"message": {"role": "assistant", "content": f"ok {i}"}}]}),
            "total_ms": 5.0,
        }
        for i in range(CALLS)
    ]
    return bytes(_zstd().compress("".join(json.dumps(r) + "\n" for r in rows).encode()))


# A stub bucket with SESSIONS captures under m/run/ plus a non-trace notes.txt, listed two keys
# a page, refusing any key but ACCESS_KEY and any second full download of an object. The
# benchmark inherits os.environ, so every AWS_* setting and proxy of the host is cleared first.
@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[S3Stub]:
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with S3Stub(page_size=2, access_key=ACCESS_KEY, full_gets_once=True) as stub:
        for i in range(SESSIONS):
            stub.put(BUCKET, f"m/run/task{i}-calls.jsonl.zst", capture(f"s{i}"))
        stub.put(BUCKET, "m/run/notes.txt", b"not a trace")
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", stub.url)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "e2e-secret")
        monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        yield stub


# A two-worker session replay of every capture in the stub bucket, caching under `cache`.
def config(cache: Path) -> Dict[str, Any]:
    return {
        "data": {
            "type": "otel_trace_replay",
            "otel_trace_replay": {
                "trace_files": [f"s3://{BUCKET}/m/*/*-calls.jsonl.zst"],
                "remote_cache_dir": str(cache),
                "use_static_model": True,
                "static_model_name": "mock_model",
            },
        },
        "load": {
            "type": "trace_session_replay",
            "stages": [{"concurrent_sessions": 2}],
            "num_workers": 2,
            "worker_max_concurrency": 4,
        },
        "api": {"type": "chat"},
        "server": {"type": "mock", "base_url": "http://0.0.0.0:8000"},
        "tokenizer": {"pretrained_model_name_or_path": str(extract_tarball(TOKENIZER))},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True}},
    }


# Cold run, 4 captures, 2 workers: all 8 calls succeed; listing paginates; every request is
# signed with ACCESS_KEY; every GET carries If-Match with the listed ETag; each object gets one
# ranged probe and one full download, both by the parent, so the workers read the cache (a
# second full GET would be refused). A warm run over the same cache downloads nothing whole.
@pytest.mark.asyncio
async def test_cold_then_warm_cache_with_two_workers(s3: S3Stub, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cold = await run_benchmark_minimal(config(cache), work_dir=tmp_path / "cold", timeout_sec=240)
    assert cold.success, cold.stdout
    assert cold.reports and cold.reports["summary_lifecycle_metrics.json"]["successes"]["count"] == SESSIONS * CALLS

    keys = {k for _, k in s3.objects if k.endswith(".zst")}
    lists = [r for r in s3.requests if not r.key]
    assert len(lists) >= 2 and any("continuation-token" in r.query for r in lists), "listing did not paginate"
    assert all(f"Credential={ACCESS_KEY}/" in r.headers["authorization"] for r in s3.requests)
    gets = s3.gets()
    assert all(r.headers.get("if-match") == s3.etag(BUCKET, r.key) for r in gets), "a GET was not pinned to its ETag"
    assert Counter(r.key for r in gets if r.headers.get("range") == PROBE_RANGE and r.status == 206) == Counter(keys)
    assert Counter(r.key for r in gets if "range" not in r.headers and r.status == 200) == Counter(keys)
    assert len(gets) == 2 * len(keys), [(r.key, r.headers.get("range"), r.status) for r in gets]
    assert "Waited" not in cold.stdout

    before = len(s3.requests)
    warm = await run_benchmark_minimal(config(cache), work_dir=tmp_path / "warm", timeout_sec=240)
    assert warm.success, warm.stdout
    assert all("range" in r.headers for r in s3.requests[before:] if r.key), "the warm run downloaded an object again"


# A wrong access key: the run fails at startup with the credentials hint and fetches nothing.
@pytest.mark.asyncio
async def test_wrong_access_key_fails_at_startup(s3: S3Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "wrong-key")
    result = await run_benchmark_minimal(config(tmp_path / "cache"), work_dir=tmp_path / "run", timeout_sec=120)
    assert not result.success
    assert "InvalidAccessKeyId" in result.stdout and "AWS_ACCESS_KEY_ID" in result.stdout
    assert not s3.gets()
